# Unified Robot Control Interface for HumanoidVerse
# Weiji Xie @ 2025.03.04

import mujoco, mujoco_viewer, mujoco.viewer
import glfw 
import os
import time
import signal
import sys
from pathlib import Path

import torch
from humanoidverse.deploy import URCIRobot
from scipy.spatial.transform import Rotation as R
import logging
from utils.config_utils import *  # noqa: E402, F403
# add argparse arguments

from typing import Dict, Optional, Tuple
from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger
from humanoidverse.utils.helpers import np2torch, torch2np
from humanoidverse.utils.real.rotation_helper import *
from humanoidverse.utils.noise_tool import noise_process_dict, RadialPerturbation
# from description.robots.dtype import RobotExitException

import numpy as np

from isaac_utils.rotations import (
    slerp,
    my_quat_rotate,
    calc_heading_quat,
    calc_heading_quat_inv,
    quat_mul,
    quat_rotate_inverse,
    xyzw_to_wxyz,
    wxyz_to_xyzw,
    get_euler_xyz_in_tensor,
    calc_yaw_heading_quat_inv
)


def wrap_to_pi_float(angles:float):
    angles %= 2*np.pi
    angles -= 2*np.pi * (angles > np.pi)
    return angles


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_connector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1,
                            point2)


class MViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    is_pause = False
    
    def _make_viewer(self):
        def _key_callback(key):
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------
                key_char = chr(key)
                
                print(key, key_char)
                
                if key == 27: # Esc
                    self.viewer.close()
                
                if key_char == ' ':
                    self.is_pause = not self.is_pause
                
                if key_char == 'w':
                    self.cmd[1] += 0.1
                elif key_char == 's':
                    self.cmd[1] -= 0.1
                elif key_char == 'a': #
                    self.cmd[0] += 0.1
                elif key_char == 'd':
                    self.cmd[0] -= 0.1
                elif key_char == 'e':
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/20)
                    else:
                        self.cmd[2] += 0.1
                elif key_char == 'q': 
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/20)
                    else:
                        self.cmd[2] -= 0.1
                elif key_char == 'r':
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == 257: # Enter
                    self.Reset()
                    self._ref_pid = -2
                elif key_char == '[':
                    self._ref_pid -= 1
                elif key_char == ']':
                    self._ref_pid += 1
                
                elif key == ord('n') or key == ord('N'):
                    # Don't call next_task() directly as it's too slow
                    # Just set a flag to be processed later
                    self._need_next_task = True
                    logger.info("Next task requested")

                print(self.cmd, self._ref_pid)
        # glfw.set_key_callback(self.viewer.window, _key_callback)
        ...
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, key_callback=_key_callback)
        for _ in range(50):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
        self.viewer.cam.lookat[:] = np.array([0,0,0.8])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 30                         # 可根据需要调整角度
        self.viewer.cam.elevation = -30                      # 负值表示从上往下看
        
    # ---- 力矩监控显示 --------------------------------------------------------
    _torque_display_enabled: bool = False
    _torque_display_interval: int = 5   # 每隔 N 个 render_step 更新一次
    _torque_display_counter: int = 0

    def _init_torque_display(self, num_dofs: int, tau_limit: np.ndarray,
                              joint_names=None):
        """初始化独立 matplotlib 窗口，用于实时显示各执行器力矩。

        Args:
            num_dofs:     执行器数量
            tau_limit:    每个关节的力矩上限（ndarray, shape=(num_dofs,)）
            joint_names:  关节名称列表（可选）
        """
        try:
            import matplotlib
            matplotlib.use('TkAgg')   # 若 TkAgg 不可用可改为 Qt5Agg / Agg
            import matplotlib.pyplot as plt
        except Exception as e:
            logger.warning(f"[TorqueDisplay] matplotlib 初始化失败，力矩显示已跳过: {e}")
            return

        self._td_plt = plt
        self._td_num_dofs = num_dofs
        self._td_tau_limit = np.asarray(tau_limit, dtype=np.float32)
        self._td_joint_names = joint_names or [f'DOF[{i}]' for i in range(num_dofs)]

        self._td_max_tau  = np.zeros(num_dofs, dtype=np.float32)
        self._td_cur_tau  = np.zeros(num_dofs, dtype=np.float32)

        self._td_fig, (self._td_ax_cur, self._td_ax_max) = plt.subplots(
            2, 1, figsize=(max(10, num_dofs * 0.55), 7))
        self._td_fig.suptitle('执行器力矩监控', fontsize=13)

        x = np.arange(num_dofs)
        lim_max = float(self._td_tau_limit.max()) * 1.25

        # 当前力矩（有符号）
        self._td_bars_cur = self._td_ax_cur.bar(
            x, np.zeros(num_dofs), color='steelblue', alpha=0.85)
        self._td_ax_cur.axhline(0, color='k', linewidth=0.6)
        self._td_ax_cur.set_ylim(-lim_max, lim_max)
        self._td_ax_cur.set_title('当前力矩 (Nm)', fontsize=11)
        self._td_ax_cur.set_xticks(x)
        self._td_ax_cur.set_xticklabels(
            self._td_joint_names, rotation=45, ha='right', fontsize=7)
        self._td_ax_cur.set_ylabel('Nm')
        # 画力矩上下限虚线
        for sign in (1, -1):
            self._td_ax_cur.step(
                np.append(x - 0.5, x[-1] + 0.5),
                np.append(sign * self._td_tau_limit, sign * self._td_tau_limit[-1]),
                where='post', linestyle='--', color='red', linewidth=0.8, alpha=0.6)

        # 历史最大力矩（绝对值）
        self._td_bars_max = self._td_ax_max.bar(
            x, np.zeros(num_dofs), color='tomato', alpha=0.85)
        self._td_ax_max.set_ylim(0, lim_max)
        self._td_ax_max.set_title('历史最大力矩 |τ|_max (Nm)', fontsize=11)
        self._td_ax_max.set_xticks(x)
        self._td_ax_max.set_xticklabels(
            self._td_joint_names, rotation=45, ha='right', fontsize=7)
        self._td_ax_max.set_ylabel('Nm')
        self._td_ax_max.step(
            np.append(x - 0.5, x[-1] + 0.5),
            np.append(self._td_tau_limit, self._td_tau_limit[-1]),
            where='post', linestyle='--', color='red', linewidth=0.8, alpha=0.6,
            label='limit')
        self._td_ax_max.legend(fontsize=8)

        # 百分比文字标注（预创建，后续更新文本内容）
        self._td_pct_texts = [
            self._td_ax_max.text(i, 0, '', ha='center', va='bottom', fontsize=6)
            for i in x
        ]

        plt.tight_layout()
        plt.ion()
        plt.show(block=False)
        self._torque_display_enabled = True
        logger.info("[TorqueDisplay] 力矩显示窗口已初始化")

    def _update_torque_display(self):
        """将 self._td_cur_tau / self._td_max_tau 刷新到 matplotlib 窗口。"""
        if not self._torque_display_enabled:
            return
        self._torque_display_counter += 1
        if self._torque_display_counter % self._torque_display_interval != 0:
            return

        cur  = self._td_cur_tau
        maxv = self._td_max_tau
        lim  = self._td_tau_limit

        # ---- 当前力矩柱状图 ----
        for bar, val, l in zip(self._td_bars_cur, cur, lim):
            bar.set_height(val)
            ratio = abs(val) / max(l, 1e-6)
            bar.set_color(
                'green'      if ratio < 0.80 else
                'darkorange' if ratio < 0.95 else
                'red')

        # ---- 历史最大力矩柱状图 ----
        for bar, val, l, txt in zip(self._td_bars_max, maxv, lim, self._td_pct_texts):
            bar.set_height(val)
            ratio = val / max(l, 1e-6)
            bar.set_color(
                'steelblue'  if ratio < 0.80 else
                'darkorange' if ratio < 0.95 else
                'red')
            txt.set_position((txt.get_position()[0], val + 0.3))
            txt.set_text(f'{ratio*100:.0f}%')

        self._td_fig.canvas.draw_idle()
        try:
            self._td_plt.pause(0.001)
        except Exception:
            pass
    # -------------------------------------------------------------------------

    def render_step(self,need_next_task=None):
        if need_next_task:
            self._need_next_task = need_next_task 
        # Handle next_task request outside of keyboard callback
        if hasattr(self, '_need_next_task') and self._need_next_task:
            logger.info("Processing next_task request...")
            self._need_next_task = False
            self.next_task()
            self._ref_pid = -2
            
        while self.is_pause:
            time.sleep(0.01)
        
        if self.is_render:
            if self.viewer.is_running():
                for i in range(self.ref_motion.shape[1]):
                    self.viewer.user_scn.geoms[i].pos = self.ref_motion[0, i]
                self.viewer.sync()
            
                # breakpoint()
            # else:
            #     raise RobotExitException("Mujoco Robot Exit")

        self._update_torque_display()
        ...


class ViewerPlugin:
    # TODO: visualize the motion keypoint in MujocoViewer
    
    is_recording = False
    fps = 30
    
    def _make_viewer(self):
        ...
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.lookat[:] = np.array([0,0,0.6])
        self.viewer.cam.distance = 3.0        
        self.viewer.cam.azimuth = 180                         # 可根据需要调整角度
        self.viewer.cam.elevation = -20                      # 负值表示从上往下看
        def _key_callback(window, key, scancode, action, mods):
            if action == glfw.PRESS:
                #  Keyboard mapping:
                #  ----------------
                #  |   K L ; '    |
                #  |   , . /      |
                #  ----------------

                if key == glfw.KEY_COMMA:
                    self.cmd[1] += 0.1
                elif key == glfw.KEY_SLASH:
                    self.cmd[1] -= 0.1
                elif key == glfw.KEY_L: #
                    self.cmd[0] += 0.1
                elif key == glfw.KEY_PERIOD:
                    self.cmd[0] -= 0.1
                elif key == glfw.KEY_K:
                    if self.heading_cmd:
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]+np.pi/10)
                    else:
                        self.cmd[2] += 0.1
                elif key == glfw.KEY_SEMICOLON:
                    if self.heading_cmd:                            
                        self.cmd[3] = wrap_to_pi_float(self.cmd[3]-np.pi/10)
                    else:
                        self.cmd[2] -= 0.1
                elif key == glfw.KEY_APOSTROPHE:
                    self.cmd = np.array(self.cfg.deploy.defcmd)
                elif key == glfw.KEY_ENTER:
                    # self.Reset()
                    self._ref_pid = -2
                elif key == glfw.KEY_LEFT_BRACKET:
                    self._ref_pid -= 1
                elif key == glfw.KEY_RIGHT_BRACKET:
                    self._ref_pid += 1
                elif key == glfw.KEY_R:
                    # Save video when R key is pressed
                    if self.is_recording and hasattr(self, '_video_buffer') and len(self._video_buffer) > 0:
                        logger.info("Saving video on R key press...")
                        model_path:Path = self.cfg.checkpoint
                        model_id = model_path.stem.replace('model_', 'ckpt_')
                        video_path:Path = model_path.parent.parent / 'renderings' / model_id / f'video_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        import imageio
                        imageio.mimsave(video_path, self._video_buffer, fps=self.fps, macro_block_size=self.macro_block_size)
                        logger.info(f"Video saved to {video_path}")
                    else:
                        logger.warning("No video buffer available for saving")
                elif key>=320 and key<=329:
                    # 小键盘的数字 0-9 ,控制 策略切换
                    self._ref_pid = key-320
                elif key >= 48 and key <= 57:
                    # 0-9 ,控制 策略切换
                    self._ref_pid = key-48
                else:
                    print('Press key: ',key)
                
                print(self.cmd, self._ref_pid)
            self.viewer._key_callback(window, key, scancode, action, mods)
        glfw.set_key_callback(self.viewer.window, _key_callback)
        
        if self.is_recording:
            logger.info("Recording is True")
            self.start_time = time.time()
            self._video_buffer = []
            viewport = self.viewer.viewport
            self._frame_buffer = np.zeros((viewport.height, viewport.width, 3), dtype=np.uint8)
            self._frame_size = (viewport.height // self.macro_block_size * self.macro_block_size,  #
                                viewport.width // self.macro_block_size * self.macro_block_size)
        
    macro_block_size = 16
    def render_step(self):
        if self.is_render:
            # if self.viewer.is_running():
            if self.viewer.is_alive:
                self.viewer.render()
                # self.viewer.sync()
                if self.is_recording and (time.time() - self.start_time >= 1/self.fps):
                    self.start_time = time.time()
                    mujoco.mjr_readPixels(self._frame_buffer, None, self.viewer.viewport, self.viewer.ctx)
                    # imageio.imwrite('frame.png', frame[::-1])
                    # breakpoint()
                    
                    self._video_buffer.append(self._frame_buffer[::-1].copy()[:self._frame_size[0], :self._frame_size[1]])
            else:
                if self.is_recording:
                    logger.info("Mujoco: Saving video ...")
                    
                    model_path:Path = self.cfg.checkpoint
                    model_id = model_path.stem.replace('model_', 'ckpt_')
                    video_path:Path = model_path.parent.parent / 'renderings' / model_id / f'video_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    import imageio
                    imageio.mimsave(video_path, self._video_buffer, fps=self.fps, macro_block_size=self.macro_block_size)
                    logger.info(f"Video saved to {video_path}")
                # raise RobotExitException("Mujoco Robot Exit")
        

class MujocoRobot(URCIRobot, MViewerPlugin):
    REAL=False
    
    HANG=False
    
    RAND_NOISE  : bool = False
    RAND_DELAY  : bool = False
    RAND_MASK   : bool = False
    RAND_IMU    : bool = False
    RAND_RADIAL : bool = False
    RAND_OFFSET : bool = False
    TANH_ATTACK : bool = False
    
    noise_imu_type:str = 'ou' # 'white' or 'ou' or 'pink'
    noise_imu_rpy:float = 3         # degree
    noise_imu_omega:float = 0.2     # rad/s
    # noise_imu_omega:float = 0.3     # rad/s
    noise_imu_ou_theta:float = 1    # 1/s
    # noise_imu_ou_theta:float = 0.3    # 1/s
    
    noise_ratio = 3e-2
    # delay_ratio = (4, 25) # unit: ms
    delay_ratio = (4, 20) # unit: ms
    mask_ratio = 0.7
    mk_rand_noise = lambda tens, ratio: (
                                            (tens * (1 + torch.randn_like(tens) * ratio) ) 
                                    if isinstance(tens, torch.Tensor) 
                                    else    (tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio)) )# type: ignore
    # mk_rand_noise = lambda tens, ratio: tens * (1 + np.random.randn(*tens.shape).astype(tens.dtype) * ratio) # type: ignore
    
    # print_torque = lambda tau: print(f"tau (norm, max) = {torch.norm(tau):.2f}, \t{torch.max(tau):.2f}", end='\r')
    print_torque = lambda tau: print(f"tau (norm, max) = {np.linalg.norm(tau):.2f}, \t{np.max(tau):.2f}", end='\r')
    
    def verify_dof_order(self):
        """验证并打印DOF顺序信息"""
        print("=== DOF Order Verification ===")
        print(f"data.qpos shape: {self.data.qpos.shape}")
        print(f"data.qpos[7:] shape: {self.data.qpos[7:].shape}")
        print(f"Expected DOF count: {self.num_dofs}")
        
        # 打印当前关节角度值和对应的关节名称
        print("\n当前关节状态:")
        for i in range(len(self.data.qpos[7:])):
            # 找到对应的关节名称
            joint_name = "Unknown"
            for j in range(self.model.njnt):
                joint_dof = self.model.jnt_dofadr[j]
                if joint_dof == i + 6:
                    joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                    break
            
            angle_deg = np.degrees(self.data.qpos[7 + i])
            print(f"  DOF[{i:2d}]: {joint_name:20s} = {self.data.qpos[7 + i]:8.4f} rad ({angle_deg:7.2f}°)")
        
        # 检查是否有期望的关节名称顺序
        expected_joints = [
            # 你可以在这里列出期望的关节顺序
            # 例如: "left_hip_yaw", "left_hip_roll", "left_hip_pitch", ...
        ]
        
        if expected_joints:
            print("\n期望顺序验证:")
            for i, expected_name in enumerate(expected_joints):
                if i < len(self.data.qpos[7:]):
                    actual_name = "Unknown"
                    for j in range(self.model.njnt):
                        joint_dof = self.model.jnt_dofadr[j]
                        if joint_dof == i + 7:
                            actual_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
                            break
                    
                    match = "✓" if expected_name == actual_name else "✗"
                    print(f"  {match} DOF[{i:2d}]: 期望 '{expected_name}' vs 实际 '{actual_name}'")
        
        print("===============================\n")
    
    def __init__(self, cfg):
        super().__init__(cfg)
        
        def signal_handler(sig, frame):
            logger.info("Ctrl+C  Exiting safely...")
            raise KeyboardInterrupt("Mujoco Robot Exiting")
        signal.signal(signal.SIGINT, signal_handler)
        
        self.decimation = cfg.simulator.config.sim.control_decimation
        self.sim_dt = 1/cfg.simulator.config.sim.fps
        assert self.dt == self.decimation * self.sim_dt
        # self._subtimer = 0
        
        
        
        self.model = mujoco.MjModel.from_xml_path(os.path.join(cfg.robot.asset.asset_root, cfg.robot.asset.xml_file)) # type: ignore
        print("XML", cfg.robot.asset.xml_file)
        
        # 输出所有body名称用于debug
        print("=== Body Names Debug ===")
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            print(f"Body {i}: {body_name}")
        print("========================")
        
        # Create body name to index mapping for ordered body selection
        if hasattr(self, 'body_names') and self.body_names is not None:
            self.body_indices = []
            for body_name in self.body_names:
                try:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    self.body_indices.append(body_id)
                except:
                    logger.warning(f"Body name '{body_name}' not found in MuJoCo model")
            self.body_indices = np.array(self.body_indices)
            logger.info(f"Body indices for ordered selection: {self.body_indices}")
        else:
            logger.warning("No body_names found in config, using default [1:] indexing")
            self.body_indices = None
        
        if "extend_config" in self.cfg.robot.motion:
            extend_parent_ids, extend_pos, extend_rot = [], [], []
            for extend_config in self.cfg.robot.motion.extend_config:
                parent_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, extend_config["parent_name"])
                extend_parent_ids.append(parent_id)
                extend_pos.append(extend_config["pos"])
                extend_rot.append(extend_config["rot"])
            self.extend_body_parent_ids = np.array(extend_parent_ids)
            self.extend_pos = torch.tensor(extend_pos, dtype=torch.float32)
            self.extend_rot_wxyz = torch.tensor(extend_rot, dtype=torch.float32)
            self.extend_rot_xyzw = self.extend_rot_wxyz[:, [1, 2, 3, 0]]
        self.data = mujoco.MjData(self.model) # type: ignore
        
        self.model.opt.timestep = self.sim_dt
        print("timestep", self.model.opt.timestep)
        if cfg.deploy.render:
            self.is_render = True
            if cfg.__contains__('recording') and cfg.recording:
                self.is_recording = True
            self._make_viewer()
            # 初始化力矩显示窗口（需在 _make_buffer / Reset 之后，kp/kd/tau_limit 已就绪）
        else:
            self.is_render = False
            
        self.num_ctrl = self.data.ctrl.shape[0]
        assert self.num_ctrl == self.num_actions, f"Number of control DOFs {self.num_ctrl} does not match number of actions {self.num_actions}"
        
        # noise init
        if self.RAND_IMU:
            if self.noise_imu_type == 'ou':
                sigma = np.sqrt(2*self.noise_imu_ou_theta)
                theta = self.noise_imu_ou_theta
            elif self.noise_imu_type == 'white':
                sigma = 1
                theta = 0
            else:
                raise ValueError(f"Invalid noise type: {self.noise_imu_type}")
            self.noise_imu = noise_process_dict[self.noise_imu_type]((6,), mu=0,sigma=sigma, theta=theta, dt=self.sim_dt)
            
            # self.noise_imu = noise_process_dict['empty']((3,))
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=250*np.sqrt(0.02), theta=25, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=250, theta=25, dt=self.sim_dt)
            
            # self.noise_imu = noise_process_dict['ou']((6,), mu=0, sigma=1, theta=0.15, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((6,), mu=0, sigma=1.825)
            
            # self.noise_imu = noise_process_dict['ou']((6,), mu=0, sigma=np.sqrt(2*3), theta=3, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((6,), mu=0, sigma=1)
            
            # self.noise_imu = noise_process_dict['ou']((3,), mu=0, sigma=np.sqrt(250), theta=25, dt=self.sim_dt)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=35.355)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=5)
            # self.noise_imu = noise_process_dict['white']((3,), mu=0, sigma=2.236)
            print(self.noise_imu)
            # print('OUProcess', 'mu', self.noise_imu.mu, 'sigma', self.noise_imu.sigma, 'theta', self.noise_imu.theta, 'dt', self.noise_imu.dt)
            # self.noise_imu = noise_process_dict[self.noise_imu_type]((2,), mu=0, sigma=1.0, theta=self.noise_imu_omega, dt=self.sim_dt)
        
        if self.RAND_RADIAL:
            self._radial_perturbation = RadialPerturbation(sigma=0.03, kappa=100)
            self.radial_perturbation = lambda x: self._radial_perturbation(torch.from_numpy(x).reshape(1, -1)).numpy().reshape(-1)
        
        logger.info("Initializing Mujoco Robot")
        logger.info("Task Name: {}".format(cfg.log_task_name))
        logger.info("Robot Type: {}".format(cfg.robot.asset.robot_type))
        
        
        logger.info(f"decimation: {self.decimation}, sim_dt: {self.sim_dt}, dt: {self.dt}")
        logger.info(f"xml_file: {cfg.robot.asset.xml_file}")
        # print(self.decimation, self.sim_dt, self.dt)

        self.Reset()
        
        # 验证DOF顺序
        self.verify_dof_order()

        # 初始化力矩显示（render 模式下）
        if self.is_render:
            # 从 MuJoCo 模型中提取执行器名称（actuator name）
            actuator_names = []
            for i in range(self.model.nu):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                actuator_names.append(name if name else f'act_{i}')
            # self._init_torque_display(
            #     num_dofs=self.num_actions,
            #     tau_limit=self.tau_limit,
            #     joint_names=actuator_names[:self.num_actions],
            # )
        
        mujoco.mj_step(self.model, self.data) # type: ignore


    def _reset(self):
        self.data.qpos[:3] = np.array(self.cfg.robot.init_state.pos)
        self.data.qpos[3:7] = np.array(self.cfg.robot.init_state.rot)[[3,0,1,2]] # XYZW to WXYZ
        
        self.data.qpos[7:] = self.dof_init_pose
        self.data.qvel[:]   = 0

        if hasattr(self, 'root_init_pos'):
            self.data.qpos[:3] = self.root_init_pos
            self.data.qpos[3:7] = self.root_init_rot[[3,0,1,2]]
            self.data.qvel[:3] = self.root_init_vel
            self.data.qvel[3:6] = self.root_init_ang_vel
            self.data.qvel[6:] = self.dof_init_vel
            self.data.qpos[7:] = self.dof_init_pos
            logger.info(f"init from motion_lib")

        self.cmd = np.array(self.cfg.deploy.defcmd)
        
        if self.RAND_IMU:
            self.noise_imu.reset()


    @staticmethod
    def pd_control(target_q, q, kp, target_dq, dq, kd):
        '''Calculates torques from position commands
        '''
        if MujocoRobot.RAND_NOISE:
            kp,kd = MujocoRobot.mk_rand_noise(np.array([kp, kd]), MujocoRobot.noise_ratio)
        return (target_q - q) * kp + (target_dq - dq) * kd

    def _get_state(self):
        '''Extracts physical states from the mujoco data structure
        '''
        data = self.data
        self.q_raw = data.qpos.astype(np.double)[7:] # 19 dim
        self.q = self.q_raw.copy()
            # WXYZ
            # 3 dim base pos + 4 dim quat + 12 dim actuation angles
        self.dq_raw = data.qvel.astype(np.double)[6:] # 18 dim ?????
        self.dq = self.dq_raw.copy()
            # 3 dim base vel + 3 dim omega + 12 dim actuation vel
        
        self.pos = data.qpos.astype(np.double)[:3]
        self.quat_raw = data.qpos.astype(np.double)[3:7][[1,2,3,0]] # WXYZ to XYZW
        self.quat = self.quat_raw.copy()
        self.vel = data.qvel.astype(np.double)[:3]
        self.omega_raw = data.qvel.astype(np.double)[3:6]
        self.omega = self.omega_raw.copy()
        
        # Get rigid body positions and rotations directly from MuJoCo
        self.rigid_body_pos = data.xpos.copy()  # Make a copy to avoid reference issues
        rigid_body_rot = data.xquat.copy()  # WXYZ format
        rigid_body_rot_xyzw = rigid_body_rot[:, [1, 2, 3, 0]]

        # Optimize: cache tensor conversions and avoid repeated allocations
        if not hasattr(self, '_rigid_body_pos_tensor') or self._rigid_body_pos_tensor.shape[0] != self.rigid_body_pos.shape[0]:
            # Initialize cached tensors only once or when size changes
            self._rigid_body_pos_tensor = torch.zeros_like(torch.from_numpy(self.rigid_body_pos)).float()
            self._rigid_body_rot_tensor = torch.zeros_like(torch.from_numpy(rigid_body_rot_xyzw)).float()
        
        # Update cached tensors in-place to avoid memory allocation
        self._rigid_body_pos_tensor.copy_(torch.from_numpy(self.rigid_body_pos))
        self._rigid_body_rot_tensor.copy_(torch.from_numpy(rigid_body_rot_xyzw))

        # Use cached tensors for computation
        parent_pos = self._rigid_body_pos_tensor[self.extend_body_parent_ids]
        parent_rot = self._rigid_body_rot_tensor[self.extend_body_parent_ids]

        rotated_pos_in_parent = my_quat_rotate(parent_rot, self.extend_pos.reshape(-1, 3))
        curr_pos = my_quat_rotate(self.extend_rot_xyzw.reshape(-1, 4), rotated_pos_in_parent) + parent_pos

        # Update rigid_body_pos_extend in-place if possible
        # Use body_names order if available, otherwise use default [1:] indexing
        if self.body_indices is not None:
            # Select body positions according to body_names order
            ordered_body_pos = self._rigid_body_pos_tensor[self.body_indices]
            new_rigid_body_pos_extend = torch.cat([ordered_body_pos, curr_pos], dim=0).unsqueeze(0)
        else:
            # Fallback to original [1:] indexing
            new_rigid_body_pos_extend = torch.cat([self._rigid_body_pos_tensor[1:], curr_pos], dim=0).unsqueeze(0)
        
        if hasattr(self, 'rigid_body_pos_extend') and self.rigid_body_pos_extend.shape == new_rigid_body_pos_extend.shape:
            self.rigid_body_pos_extend.copy_(new_rigid_body_pos_extend)
        else:
            self.rigid_body_pos_extend = new_rigid_body_pos_extend
        
    
        # print(self.omega_raw)
        if self.RAND_OFFSET:
            self.q = self.q_raw - self._motor_offset
        
        if self.RAND_IMU:
            step_noise_imu = self.noise_imu.x
            
            self.rpy = (quaternion_to_euler_array(self.quat)) + step_noise_imu[:3] * self.noise_imu_rpy *(np.pi/180)  
            self.quat = rpy_to_quaternion_array(self.rpy)
            quat_dist = np.arccos(np.clip(np.abs(np.dot(self.quat_raw, self.quat)), -1.0, 1.0))
            
            self.omega = self.omega + step_noise_imu[3:] * self.noise_imu_omega
            
            # print('quat_dist:', quat_dist)
            # print('rpy', self.rpy)
            # print('omega', self.omega, '\t| omega_raw', self.omega_raw, '\t| step_noise_imu', step_noise_imu[3:])
            # breakpoint()
            
        if self.RAND_RADIAL:
            # self.omega = self.radial_perturbation(self.omega)
            self.dq = self.radial_perturbation(self.dq)
            
            # print(f"{self.omega=}, {self.omega_raw=}")
            print(f"{self.dq=}, {self.dq_raw=}")
        
        r = R.from_quat(self.quat)  # R.from_quat: need xyzw
        self.rpy = quaternion_to_euler_array(self.quat) # need xyzw
        self.rpy[self.rpy > math.pi] -= 2 * math.pi
        self.gvec = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
    
        
    
    def _sanity_check(self, target_q):
        unsafe_dof = np.where((np.abs(target_q - self.q) > 220 ) | 
                              (np.abs(self.dq) > 600))[0]
        if len(unsafe_dof) > 0:
            for motor_idx in unsafe_dof:
                logger.error(f"Action of joint {motor_idx} is too large.\n"
                                f"target q\t: {target_q[motor_idx]} \n"
                                f"target dq\t: {0} \n"
                                f"q\t\t: {self.q[motor_idx]} \n"
                                f"dq\t\t: {self.dq[motor_idx]}\n")
                # breakpoint()  
        
    _motor_offset = np.array([
        3, 0.5, 2, -0.5, -1, 1,
        -2, 1, -.3, 1, 0.3, 0.1,
        0, 0, 0,
        0, 1, 0, -1,
        -2, 0, 0, 0
    ])*(np.pi/180)   # [23]
        
    def _apply_action(self, target_q):
        
        self._sanity_check(target_q)
        
        # logger.info(f"target_q: {target_q}")
        rand_mask = np.random.random(self.num_actions) < self.mask_ratio

        rand_delay = np.random.randint((1e-3*self.delay_ratio[0])//self.sim_dt, (1e-3*self.delay_ratio[1])//self.sim_dt) * self.RAND_DELAY
        step_delay = rand_delay//self.decimation
        substep_delay = rand_delay - step_delay * self.decimation
        
        if step_delay ==0:
            old_action = self.history_handler.history['actions'][0, step_delay]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = target_q
        else:
            old_action = self.history_handler.history['actions'][0, step_delay]
            cur_action = self.history_handler.history['actions'][0, step_delay+1]
            old_trg_q = np.clip(old_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
            cur_trg_q = np.clip(cur_action, -self.clip_action_limit, self.clip_action_limit) * self.action_scale + self.dof_init_pose
                
        for i in range(self.decimation):
            self.GetState()
            
            if self.RAND_DELAY and i < substep_delay:
                target_q = old_trg_q
            elif self.RAND_MASK:
                target_q = cur_trg_q * rand_mask + old_trg_q.numpy() * (1 - rand_mask)
            else:
                target_q = cur_trg_q
                
            # if self.RAND_OFFSET:
            #     target_q = target_q + self._motor_offset
                
            tau = self.pd_control(target_q, self.q, self.kp,
                            0, self.dq, self.kd)  # Calc torques
            
            if self.TANH_ATTACK:
                kappa = {
                    4:15,
                    10:15,
                    13:200,
                    # 14:200,
                }
                # tau = np.tanh(tau)
                for i in kappa.keys():
                    tau[i] = np.tanh(tau[i]/kappa[i]) * kappa[i]
            
            if self.RAND_NOISE: tau = MujocoRobot.mk_rand_noise(tau, MujocoRobot.noise_ratio)
            # print("tau",tau)
            # logger.info(f"tau: {tau}")
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)  # Clamp torques

            # 记录力矩用于可视化
            if self._torque_display_enabled:
                self._td_cur_tau[:] = tau
                np.maximum(self._td_max_tau, np.abs(tau), out=self._td_max_tau)
            
            
            # MujocoRobot.print_torque(tau)
            # tau*=0
            # print(np.linalg.norm(target_q-self.q), np.linalg.norm(self.dq), np.linalg.norm(tau))
            # self.data.qpos[:3] = np.array([0,0,1])
            self.data.ctrl[:] = tau
            if self.HANG:
                # self.data.ctrl[14] = 0.5
                self.data.qpos[:3] = np.array([0,0,1])
                self.data.qpos[3:7] = np.array([1,0,0,0])

            mujoco.mj_step(self.model, self.data) # type: ignore
    
            if self.RAND_IMU:
                self.noise_imu.step()
            
            self.tracking()
        motion_times = torch.tensor((self.timer+1) * self.dt, dtype=torch.float32)
        if motion_times > self.motion_length:
            need_next_task = True
        else:
            need_next_task = False
        self.render_step(need_next_task=need_next_task)
            # self._subtimer += 1
        
        
    # Noise Version ApplyAction & Obs
    def ApplyAction(self, action): 
        if self.RAND_NOISE: action = MujocoRobot.mk_rand_noise(action, MujocoRobot.noise_ratio)
        URCIRobot.ApplyAction(self, action)


            
        # breakpoint()

    # def Obs(self):
        
    #     # return {k: torch2np(v) for k, v in self.obs_buf_dict.items()}
        
    #     actor_obs = torch2np(self.obs_buf_dict['actor_obs']).reshape(1, -1)
    #     if self.RAND_NOISE:
    #         actor_obs = MujocoRobot.mk_rand_noise(actor_obs, MujocoRobot.noise_ratio)
    #     return {
    #         'actor_obs': actor_obs
    #         }
        

    def _get_motion_to_save_torch(self)->Tuple[float, Dict[str, torch.Tensor]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = torch.from_numpy(sRot.from_quat(root_rot).as_rotvec()).float() # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = torch.cat([root_rot_vec[..., None, :], 
                                torch.from_numpy(self._dof_axis * dof[..., None]), 
                                torch.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': torch.from_numpy(root_trans),
            'root_rot': torch.from_numpy(root_rot), # 统一save xyzw
            'dof': torch.from_numpy(dof),
            'pose_aa': pose_aa,
            'action': torch.from_numpy(self.act),
            # 'actor_obs': np2torch(self.obs_buf_dict['actor_obs']),
            'terminate': torch.zeros((1,)),
            'dof_vel': torch.from_numpy(self.dq),
            'root_lin_vel': torch.from_numpy(self.vel),
            'root_ang_vel': torch.from_numpy(self.omega),
            
            'clock_time': torch.from_numpy(np.array([time.time()])),
            'tau': torch.from_numpy(self.data.ctrl).clone(),
            'cmd': torch.from_numpy(self.cmd),
            'root_rot_raw': torch.from_numpy(self.quat_raw),
            'root_ang_vel_raw': torch.from_numpy(self.omega_raw),
        }
        
    def _get_motion_to_save_np(self)->Tuple[float, Dict[str, np.ndarray]]:
        
        from scipy.spatial.transform import Rotation as sRot
        
        motion_time = (self.timer) * self.dt 

        root_trans = self.pos
        root_rot = self.quat # XYZW
        root_rot_vec = np.array(sRot.from_quat(root_rot).as_rotvec(), dtype=np.float32) # type: ignore
        dof = self.q
        # T, num_env, J, 3
        # print(self._motion_lib.mesh_parsers.dof_axis)
        pose_aa = np.concatenate([root_rot_vec[..., None, :], 
                                np.array(self._dof_axis * dof[..., None]), 
                                np.zeros((self.num_augment_joint, 3))], axis = 0) # type: ignore
        
        return motion_time, {
            'root_trans_offset': (root_trans).copy(),
            'root_rot': (root_rot).copy(), # 统一save xyzw
            'dof': (dof).copy(),
            'pose_aa': pose_aa,
            'action': (self.act).copy(),
            # 'actor_obs': (self.obs_buf_dict['actor_obs']),
            'terminate': np.zeros((1,)),
            'dof_vel': (self.dq).copy(),
            'root_lin_vel': (self.vel).copy(),
            'root_ang_vel': (self.omega).copy(),
            
            'clock_time': (np.array([time.time()])),
            'tau': (self.data.ctrl).copy(),
            'cmd': (self.cmd).copy()
        }
        
    _get_motion_to_save = _get_motion_to_save_np


    def tracking(self):
        # breakpoint()
        # print(np.linalg.norm(self.data.xpos[6]-self.data.xpos[12]))
        if np.any(self.data.contact.pos[:,2] > 0.01):
            names_list = self.model.names.decode('utf-8').split('\x00')
            res = np.zeros((6,1),dtype=np.float64)
            
            def get_geom_name(geom_id):
                try:
                    body_id = self.model.geom_bodyid[geom_id]
                    if body_id + 1 < len(names_list):
                        return names_list[body_id + 1]
                    else:
                        return f"geom_{geom_id}"
                except IndexError:
                    return f"unknown_geom_{geom_id}"
            
            geom_force = lambda x:mujoco.mj_contactForce(self.model,self.data,x,res) #type:ignore
            
            for contact in self.data.contact:
                if contact.pos[2] > 0.01 and contact.geom1 != 0 and contact.geom2 != 0:
                    geom1_name = get_geom_name(contact.geom1)
                    geom2_name = get_geom_name(contact.geom2)
                    logger.warning(f"Warning!!! Collision between '{geom1_name,contact.geom1}' and '{geom2_name,contact.geom2}' at position {contact.pos}.")
                    # breakpoint()
    def _make_buffer(self):
        super()._make_buffer()
    
    def next_task(self):
        
        self.motion_start_idx += 1
        logger.info(f"DEBUG: motion_start_idx = {self.motion_start_idx}")
        
        # Get num_motions from motion_lib
        if hasattr(self, 'motion_lib') and self.motion_lib is not None:
            num_motions = self.motion_lib._num_unique_motions
            logger.info(f"DEBUG: num_motions = {num_motions}")
            # self.save_batch()
            if self.motion_start_idx >= num_motions:
                self.motion_start_idx = 0
            # self.motion_lib.load_motions(random_sample=False, start_idx=self.motion_start_idx)
            self.motion_ids = self.motion_lib.sample_motions(n=1, start_idx=self.motion_start_idx, random_sample=False)
            self._update_init_state()
            self.Reset()
        else:
            logger.warning("No motion_lib available, cannot switch to next task")
