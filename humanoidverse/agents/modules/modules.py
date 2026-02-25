import torch
import torch.nn as nn
import inspect
import torch.nn.functional as F
import math
from torch import Tensor
from humanoidverse.utils.running_mean_std import RunningMeanStd
from humanoidverse.utils import model_utils

class BaseModule(nn.Module):
    def __init__(self, obs_dim_dict, module_config_dict):
        super(BaseModule, self).__init__()
        self.obs_dim_dict = obs_dim_dict
        self.module_config_dict = module_config_dict

        self._calculate_input_dim()
        self._calculate_output_dim()
        self._build_network_layer(self.module_config_dict.layer_config)

    def _calculate_input_dim(self):
        # calculate input dimension based on the input specifications
        input_dim = 0
        for each_input in self.module_config_dict['input_dim']:
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

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim

    def _build_network_layer(self, layer_config):
        if layer_config['type'] == 'MLP':
            self._build_mlp_layer(layer_config)
        else:
            raise NotImplementedError(f"Unsupported layer type: {layer_config['type']}")
        
    def _build_mlp_layer(self, layer_config):
        layers = []
        hidden_dims = layer_config['hidden_dims']
        output_dim = self.output_dim
        activation = getattr(nn, layer_config['activation'])()

        layers.append(nn.Linear(self.input_dim, hidden_dims[0]))
        layers.append(activation)

        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(nn.LayerNorm(hidden_dims[l + 1]))
                layers.append(activation)

        self.module = nn.Sequential(*layers)

    def forward(self, input):
        return self.module(input)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 优化：单次QKV投影
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        
        # 单次投影获取Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.d_k)
        q, k, v = qkv.unbind(2)
        
        # 转置
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 使用PyTorch高效注意力
        context = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False
        )
        
        # 重塑输出
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.w_o(context)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        # 使用Sequential优化
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        
    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # 预归一化（更稳定）
        x_norm = self.norm1(x)
        attn_output = self.self_attention(x_norm, mask)
        x = x + attn_output
        
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + ff_output
        
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Transformer(nn.Module):
    def __init__(self, history_num, future_num, module_config_dict, obs_slices):
        super(Transformer, self).__init__()
        self.module_config_dict = module_config_dict
        self.obs_slices = obs_slices
        self.history_num = history_num
        self.future_num = future_num
        self._calculate_input_dim()
        self._calculate_output_dim()

        d_model = self.module_config_dict.net_config['d_model']
        max_seq_length = self.module_config_dict.net_config['max_seq_length']
        num_heads = self.module_config_dict.net_config['num_heads']
        d_ff = self.module_config_dict.net_config['d_ff']
        num_layers = self.module_config_dict.net_config['num_layers']
        
        self.input_projection_past = nn.Linear(self.input_dim_past, d_model)
        self.input_projection_current = nn.Linear(self.input_dim_current, d_model)
        self.input_projection_future = nn.Linear(self.input_dim_future, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, self.output_dim)
        
    
    def _calculate_input_dim(self):
        self.input_dim_past = 0
        self.input_dim_future = 0
        self.input_dim_current = 0
        for obs_key in self.obs_slices.keys():
            if 'history' in obs_key:
                self.input_dim_past += self.obs_slices[obs_key][1] - self.obs_slices[obs_key][0]
            elif 'future' in obs_key:
                self.input_dim_future += self.obs_slices[obs_key][1] - self.obs_slices[obs_key][0]
            else:
                self.input_dim_current += self.obs_slices[obs_key][1] - self.obs_slices[obs_key][0]
        self.input_dim_past = int(self.input_dim_past/ self.history_num)
        self.input_dim_future = int(self.input_dim_future/ self.future_num)

    def _calculate_output_dim(self):
        output_dim = 0
        for each_output in self.module_config_dict['output_dim']:
            if isinstance(each_output, (int, float)):
                output_dim += each_output
            else:
                current_function_name = inspect.currentframe().f_code.co_name
                raise ValueError(f"{current_function_name} - Unknown output type: {each_output}")
        self.output_dim = output_dim
        
    def forward(self, x1, x2, x3, mask=None):
        x1_proj = self.input_projection_current(x1)
        x2_proj = self.input_projection_past(x2)
        x3_proj = self.input_projection_future(x3)
        
        x = torch.cat([x1_proj, x2_proj, x3_proj], dim=1)
        x = self.positional_encoding(x)
        
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        output = self.output_projection(x[:,0,:])
        return output

def build_mlp(config, num_in: int, num_out: int):
    indim = num_in
    layers = []
    for i, layer in enumerate(config.layers):
        layers.append(nn.Linear(indim, layer.units))
        if layer.use_layer_norm and i == 0:
            layers.append(nn.LayerNorm(layer.units))
        layers.append(model_utils.get_activation_func(layer.activation))
        indim = layer.units

    layers.append(nn.Linear(indim, num_out))
    return nn.Sequential(*layers)

class NormObsBase(nn.Module):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__()
        self.config = config
        self.num_in = num_in
        self.num_out = num_out
        self.build_norm()

    def build_norm(self):
        if self.config.normalize_obs:
            self.running_obs_norm = RunningMeanStd(
                shape=(self.num_in,),
                device="cpu",
                clamp_value=self.config.norm_clamp_value,
            )

    def forward(self, obs, *args, **kwargs):
        if torch.isnan(obs).any():
            raise ValueError("NaN in obs")
        if self.config.normalize_obs:
            # Only update obs during training
            if self.training:
                self.running_obs_norm.update(obs)
            obs = self.running_obs_norm.normalize(obs)
        if torch.isnan(obs).any():
            raise ValueError("NaN in obs")
        return obs

class MLP_WithNorm(NormObsBase):
    def __init__(self, config, num_in: int, num_out: int):
        super().__init__(config, num_in, num_out)
        self.mlp = build_mlp(self.config, num_in, num_out)

    def forward(self, input_dict, return_norm_obs=False):
        obs = super().forward(input_dict[self.config.obs_key])
        outs: Tensor = self.mlp(obs)

        if return_norm_obs:
            return {"outs": outs, f"norm_{self.config.obs_key}": obs}
        else:
            return outs
