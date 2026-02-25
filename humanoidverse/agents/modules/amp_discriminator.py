import torch
from torch import nn
from typing import List
from hydra.utils import instantiate
from .modules import MLP_WithNorm
import inspect

class Discriminator(MLP_WithNorm):
    def __init__(self, config, num_out: int, obs_dim_dict: dict):
        self.obs_dim_dict = obs_dim_dict
        self._calculate_input_dim(config)
        super().__init__(config, self.input_dim, num_out)

    def forward(self, input_dict: dict) -> torch.Tensor:
        outs = super().forward(input_dict)
        return torch.sigmoid(outs)

    def _calculate_input_dim(self, config):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in config['input_dim_names']:
            if each_input in self.obs_dim_dict:
                # atomic observation type
                input_dim += self.obs_dim_dict[each_input]
            elif isinstance(each_input, (int, float)):
                # direct numeric input
                input_dim += each_input
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown input type: {each_input}")
        
        self.input_dim = input_dim

    def compute_logits(
        self, input_dict: dict, return_norm_obs: bool = False
    ) -> torch.Tensor:
        outs = super().forward(input_dict, return_norm_obs=return_norm_obs)
        if return_norm_obs:
            outs["outs"] = 50 * (torch.sigmoid(outs["outs"]) - 0.5)
        else:
            outs = 50 * (torch.sigmoid(outs) - 0.5)
        return outs

    def compute_reward(self, input_dict: dict, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(input_dict)
        s = torch.clamp(s, eps, 1 - eps)
        reward = -(1 - s).log()
        reward = torch.clamp(reward, 0, 1)
        return reward

    def all_discriminator_weights(self):
        weights: list[nn.Parameter] = []
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                weights.append(mod.weight)
        return weights

    def logit_weights(self) -> List[nn.Parameter]:
        return [self.mlp[-1].weight]


# class AMPModel(PPOModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self._discriminator: Discriminator = instantiate(
#             self.config.discriminator,
#         )