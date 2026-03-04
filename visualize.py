import os
import sys
import time
sys.path.append(os.getcwd())
import numpy as np
import mujoco
import mujoco.viewer
import joblib
import typer

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

def key_call_back( keycode):
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    if chr(keycode) == "R":
        print("Reset")
        time_step = 0
    elif chr(keycode) == " ":
        print("Paused")
        paused = not paused
    elif chr(keycode) == "T":
        print("next")
        motion_id += 1
        curr_motion_key = motion_data_keys[motion_id]
        print(curr_motion_key)
    else:
        print("not mapped", chr(keycode))
    
    
        
def main(
    motion_file, asset_path
) -> None:
    global curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused, motion_data_keys
    humanoid_xml = asset_path

    motion_data = joblib.load(motion_file)
    motion_data_keys = list(motion_data.keys())

    fps = motion_data[motion_data_keys[0]]['fps']
    curr_start, num_motions, motion_id, motion_acc, time_step, dt, paused = 0, 1, 0, set(), 0, 1/fps, False
    
    assert motion_file!="", 'You have to give the path to the retargeted motion file you wanna visualize!'
    
    
    mj_model = mujoco.MjModel.from_xml_path(humanoid_xml)
    mj_data = mujoco.MjData(mj_model)
        
    mj_model.opt.timestep = dt
    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_call_back) as viewer:
        for _ in range(50):
            add_visual_capsule(viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([1, 0, 0, 1]))
   
        while viewer.is_running():
            step_start = time.time()
            curr_motion_key = motion_data_keys[motion_id]
            curr_motion = motion_data[curr_motion_key]
            curr_time = int(time_step/dt) % curr_motion['dof'].shape[0]
            
            mj_data.qpos[:3] = curr_motion['root_trans_offset'][curr_time]
            mj_data.qpos[3:7] = curr_motion['root_rot'][curr_time][[3, 0, 1, 2]]
            mj_data.qpos[7:] = curr_motion['dof'][curr_time]
                
            mujoco.mj_forward(mj_model, mj_data)
            if not paused:
                time_step += dt
            
            # joint_gt = motion_data[curr_motion_key]['smpl_joints']
            
            # for i in range(joint_gt.shape[1]):
            #     viewer.user_scn.geoms[i].pos = joint_gt[curr_time, i]
                
            viewer.sync()
            time_until_next_step = mj_model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    typer.run(main)
