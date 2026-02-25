import torch
from torch import nn, Tensor
from isaac_utils import rotations, maths
import humanoidverse.utils.torch_utils as torch_utils



def compute_returns(self, rewards, values, dones, last_values, gamma, lam):
    advantage = 0
    returns = torch.zeros_like(values)
    for step in reversed(range(self.num_transitions_per_env)):
        if step == self.num_transitions_per_env - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]

    # Compute and normalize the advantages
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

def swap_and_flatten01(arr: Tensor):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])
        
class RolloutStorage(nn.Module):

    def __init__(self, num_envs, num_transitions_per_env, device='cuda'):
        
        super().__init__()

        self.device = device

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # rnn
        # self.saved_hidden_states_a = None
        # self.saved_hidden_states_c = None

        self.step = 0
        self.stored_keys = list()
        
    def register_key(self, key: str, shape=(), dtype=torch.float):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not hasattr(self, key), key
        assert isinstance(shape, (list, tuple)), "shape must be a list or tuple"
        buffer = torch.zeros(
            (self.num_transitions_per_env, self.num_envs) + shape, dtype=dtype, device=self.device
        )
        self.register_buffer(key, buffer, persistent=False)
        self.stored_keys.append(key)
    
    def get_keys(self):
        return self.stored_keys
    
    def increment_step(self):
        self.step += 1

    def update_key(self, key: str, data: Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        assert self.step < self.num_transitions_per_env, "Rollout buffer overflow"
        getattr(self, key)[self.step].copy_(data)
        
    def batch_update_data(self, key: str, data: Tensor):
        # This class was partially copied from https://github.com/NVlabs/ProtoMotions/blob/94059259ba2b596bf908828cc04e8fc6ff901114/phys_anim/agents/utils/data_utils.py
        assert not data.requires_grad
        getattr(self, key)[:] = data
        # self.store_dict[key] += self.total_sum()

    def _save_hidden_states(self, hidden_states):
        assert NotImplementedError
        if hidden_states is None or hidden_states==(None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed 
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))]
            self.saved_hidden_states_c = [torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0


    def get_statistics(self):
        raise NotImplementedError
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0]))
        trajectory_lengths = (done_indices[1:] - done_indices[:-1])
        return trajectory_lengths.float().mean(), self.rewards.mean()
    
    def query_key(self, key: str):
        assert hasattr(self, key), key
        return getattr(self, key)

    # def mini_batch_generator(self, num_mini_batches, num_epochs=8):
    #     batch_size = self.num_envs * self.num_transitions_per_env
    #     mini_batch_size = batch_size // num_mini_batches
    #     indices = torch.randperm(num_mini_batches*mini_batch_size, requires_grad=False, device=self.device)
        
    #     _buffer_dict = {key: getattr(self, key)[:].flatten(0, 1) for key in self.stored_keys}

    #     for epoch in range(num_epochs):
    #         for i in range(num_mini_batches):

    #             start = i*mini_batch_size
    #             end = (i+1)*mini_batch_size
    #             batch_idx = indices[start:end]

    #             _batch_buffer_dict = {key: _buffer_dict[key][batch_idx] for key in self.stored_keys}
    #             yield _batch_buffer_dict

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches

#         # 确保数据内存连续并预加载到GPU
#         _buffer_dict = {
#             key: getattr(self, key)[:].flatten(0, 1).contiguous() 
#             for key in self.stored_keys
#         }
#         # 在CPU生成随机索引后传输到GPU（更高效）
#         indices = torch.randperm(batch_size,device=self.device)
# 
#         # 每个epoch整体打乱数据
#         shuffled_data = {
#             key: _buffer_dict[key][indices] 
#             for key in self.stored_keys
#         }

        indices = torch.randperm(batch_size,device=self.device)
        
        
        shuffled_data = {
            key: getattr(self, key)[:].flatten(0, 1)[indices].contiguous()
            for key in self.stored_keys
        }
        
        for epoch in range(num_epochs):
            
            # 按顺序生成连续的小批量
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                _batch_buffer_dict = {
                    key: shuffled_data[key][start:end] 
                    for key in self.stored_keys
                }
                yield _batch_buffer_dict

@torch.jit.script
def compute_humanoid_observations_max(
    body_pos: Tensor,
    body_rot: Tensor,
    body_vel: Tensor,
    body_ang_vel: Tensor,
    w_last: bool,
    dof_pos_ref,
    default_dof_pos
) -> Tensor:
    root_pos = body_pos[..., 0, :]
    root_rot = body_rot[..., 0, :]

    root_h = root_pos[..., 2:3]
    root_h_obs = root_h.reshape(root_h.shape[0], -1)
    heading_rot = rotations.calc_heading_quat_inv(root_rot.reshape(-1, 4), w_last)

    # if not root_height_obs:
    #     root_h_obs = torch.zeros_like(root_h)
    # else:
    #     root_h_obs = root_h - ground_height

    heading_rot_expand = heading_rot.reshape(root_rot.shape[0], root_rot.shape[1], 1, root_rot.shape[2])
    heading_rot_expand = heading_rot_expand.repeat((1, 1, body_pos.shape[2], 1))
    flat_heading_rot = heading_rot_expand.reshape(-1, 4)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(-1, 3)
    flat_local_body_pos = rotations.my_quat_rotate(
        flat_heading_rot, flat_local_body_pos
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2] * local_body_pos.shape[3]
    )
    # local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(-1, 4)
    flat_local_body_rot = rotations.quat_mul(flat_heading_rot, flat_body_rot, w_last)
    # flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot, w_last)
    local_body_rot_obs = flat_local_body_rot.reshape(
        body_rot.shape[0], body_rot.shape[1] * body_rot.shape[2] * body_rot.shape[3]
    )

    # if not local_root_obs:
    #     root_rot_obs = torch_utils.quat_to_tan_norm(root_rot, w_last)
    #     local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(-1, 3)
    flat_local_body_vel = rotations.my_quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2] * body_vel.shape[3]
    )

    flat_body_ang_vel = body_ang_vel.reshape(-1, 3)
    flat_local_body_ang_vel = rotations.my_quat_rotate(
        flat_heading_rot, flat_body_ang_vel
    )
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2] * body_ang_vel.shape[3]
    )

    dof_pos_obs = dof_pos_ref - default_dof_pos.unsqueeze(1)
    dof_pos_obs = dof_pos_obs.reshape(dof_pos_obs.shape[0], -1)

    obs = torch.cat(
        (
            root_h_obs,
            dof_pos_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel,
        ),
        dim=-1,
    )
    return obs