# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two types of filters which can be applied to policy output sequences.

1. Simple exponential filter
2. Butterworth filter - lowpass or bandpass

The implementation of the butterworth filter follows scipy's lfilter
https://github.com/scipy/scipy/blob/v1.2.1/scipy/signal/signaltools.py

We re-implement the logic in order to explicitly manage the y states

The filter implements::
       a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1] + ... + b[M]*x[n-M]
                             - a[1]*y[n-1] - ... - a[N]*y[n-N]

We assume M == N.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import collections
from loguru import logger
import numpy as np
from scipy.signal import butter

ACTION_FILTER_ORDER = 2
ACTION_FILTER_LOW_CUT = 0.0
ACTION_FILTER_HIGH_CUT = 4.0

class ActionFilter(object):
  """Implements a generic lowpass or bandpass action filter."""

  def __init__(self, a, b, order, num_joints, ftype='lowpass'):
    """Initializes filter.

    Either one per joint or same for all joints.

    Args:
      a: filter output history coefficients
      b: filter input coefficients
      order: filter order
      num_joints: robot DOF
      ftype: filter type. 'lowpass' or 'bandpass'
    """
    self.num_joints = num_joints
    if isinstance(a, list):
      self.a = a
      self.b = b
    else:
      self.a = [a]
      self.b = [b]

    # Either a set of parameters per joint must be specified as a list
    # Or one filter is applied to every joint
    if not ((len(self.a) == len(self.b) == num_joints) or (
        len(self.a) == len(self.b) == 1)):
      raise ValueError('Incorrect number of filter values specified')

    # Normalize by a[0]
    for i in range(len(self.a)):
      self.b[i] /= self.a[i][0]
      self.a[i] /= self.a[i][0]

    # Convert single filter to same format as filter per joint
    if len(self.a) == 1:
      self.a *= num_joints
      self.b *= num_joints
    self.a = np.stack(self.a)
    self.b = np.stack(self.b)

    if ftype == 'bandpass':
      assert len(self.b[0]) == len(self.a[0]) == 2 * order + 1
      self.hist_len = 2 * order
    elif ftype == 'lowpass':
      assert len(self.b[0]) == len(self.a[0]) == order + 1
      self.hist_len = order
    else:
      raise ValueError('%s filter type not supported' % (ftype))

    logger.info('Filter shapes: a: %s, b: %s', self.a.shape, self.b.shape)
    logger.info('Filter type:%s', ftype)

    self.yhist = collections.deque(maxlen=self.hist_len)
    self.xhist = collections.deque(maxlen=self.hist_len)
    self.reset()

  def reset(self):
    """Resets the history buffers to 0."""
    self.yhist.clear()
    self.xhist.clear()
    for _ in range(self.hist_len):
      self.yhist.appendleft(np.zeros((self.num_joints, 1)))
      self.xhist.appendleft(np.zeros((self.num_joints, 1)))

  def filter(self, x):
    """Returns filtered x."""
    xs = np.concatenate(list(self.xhist), axis=-1)
    ys = np.concatenate(list(self.yhist), axis=-1)
    y = np.multiply(x, self.b[:, 0]) + np.sum(
        np.multiply(xs, self.b[:, 1:]), axis=-1) - np.sum(
            np.multiply(ys, self.a[:, 1:]), axis=-1)
    # import pdb; pdb.set_trace()
    self.xhist.appendleft(x.reshape((self.num_joints, 1)).copy())
    self.yhist.appendleft(y.reshape((self.num_joints, 1)).copy())
    return y

  def init_history(self, x):
    x = np.expand_dims(x, axis=-1)
    for i in range(self.hist_len):
      self.xhist[i] = x
      self.yhist[i] = x

class ActionFilterButter(ActionFilter):
  """Butterworth filter."""

  def __init__(self,
               lowcut=None,
               highcut=None,
               sampling_rate=None,
               order=ACTION_FILTER_ORDER,
               num_joints=None):
    """Initializes a butterworth filter.

    Either one per joint or same for all joints.

    Args:
      lowcut: list of strings defining the low cutoff frequencies.
        The list must contain either 1 element (same filter for all joints)
        or num_joints elements
        0 for lowpass, > 0 for bandpass. Either all values must be 0
        or all > 0
      highcut: list of strings defining the high cutoff frequencies.
        The list must contain either 1 element (same filter for all joints)
        or num_joints elements
        All must be > 0
      sampling_rate: frequency of samples in Hz
      order: filter order
      num_joints: robot DOF
    """
    # import pdb; pdb.set_trace()
    self.lowcut = ([float(x) for x in lowcut]
                   if lowcut is not None else [ACTION_FILTER_LOW_CUT])
    self.highcut = ([float(x) for x in highcut]
                    if highcut is not None else [ACTION_FILTER_HIGH_CUT])
    if len(self.lowcut) != len(self.highcut):
      raise ValueError('Number of lowcut and highcut filter values should '
                       'be the same')

    if sampling_rate is None:
      raise ValueError('sampling_rate should be provided.')

    if num_joints is None:
      raise ValueError('num_joints should be provided.')

    if np.any(self.lowcut):
      if not np.all(self.lowcut):
        raise ValueError('All the filters must be of the same type: '
                         'lowpass or bandpass')
      self.ftype = 'bandpass'
    else:
      self.ftype = 'lowpass'

    a_coeffs = []
    b_coeffs = []
    for i, (l, h) in enumerate(zip(self.lowcut, self.highcut)):
      if h <= 0.0:
        raise ValueError('Highcut must be > 0')

      b, a = self.butter_filter(l, h, sampling_rate, order)
      logger.info(
          'Butterworth filter: joint: {}, lowcut: {}, highcut: {}, '
          'sampling rate: {}, order: {}, num joints: {}', i, l, h,
          sampling_rate, order, num_joints)
      b_coeffs.append(b)
      a_coeffs.append(a)

    super(ActionFilterButter, self).__init__(
        a_coeffs, b_coeffs, order, num_joints, self.ftype)

  def butter_filter(self, lowcut, highcut, fs, order=5):
    """Returns the coefficients of a butterworth filter.

    If lowcut = 0, the function returns the coefficients of a low pass filter.
    Otherwise, the coefficients of a band pass filter are returned.
    Highcut should be > 0

    Args:
      lowcut: low cutoff frequency
      highcut: high cutoff frequency
      fs: sampling rate
      order: filter order
    Return:
      b, a: parameters of a butterworth filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if low:
      b, a = butter(order, [low, high], btype='band')
    else:
      b, a = butter(order, [high], btype='low')
    return b, a
  
  def reset_by_ids(self, action_ids):
    """Resets the history buffers to 0."""
    x_hist_backup = self.xhist.copy()
    y_hist_backup = self.yhist.copy()

    self.yhist.clear()
    self.xhist.clear()
    for _ in range(self.hist_len):
      x, y = x_hist_backup.popleft(), y_hist_backup.popleft()
      x[action_ids] = 0.
      y[action_ids] = 0.
      self.yhist.append(y)
      self.xhist.append(x)

class ActionFilterButterTorch(ActionFilterButter):
  """ Utilizes pytorch for filtering. """
  def __init__(self,
               lowcut=None,
               highcut=None,
               sampling_rate=None,
               order=ACTION_FILTER_ORDER,
               num_joints=None,
               device='cpu'):
    super(ActionFilterButterTorch, self).__init__(
      lowcut, highcut, sampling_rate, order, num_joints
    )
    
    # 确保设备类型正确
    if isinstance(device, str):
        self.device = torch.device(device)
    else:
        self.device = device
    
    # 初始化张量时明确指定设备
    self.a_torch = torch.tensor(self.a, dtype=torch.float32, device=self.device)
    self.b_torch = torch.tensor(self.b, dtype=torch.float32, device=self.device)
    self.xhist_torch = torch.zeros((self.hist_len, self.num_joints), 
                                  dtype=torch.float32, device=self.device)
    self.yhist_torch = torch.zeros((self.hist_len, self.num_joints), 
                                  dtype=torch.float32, device=self.device)
    
    # 调试信息
    logger.info(f"Torch滤波器初始化: 设备={self.device}, 关节数={self.num_joints}")
    logger.info(f"a_torch形状: {self.a_torch.shape}, b_torch形状: {self.b_torch.shape}")

  def filter_old(self, x):
    return super(ActionFilterButterTorch, self).filter(x)
  
  def reset_old(self, action_ids):
    return super(ActionFilterButterTorch, self).reset_by_ids(action_ids)
  
  def filter(self, x):
    """Returns filtered x. """
    try:
        # 设备一致性检查
        if x.device != self.device:
            logger.warning(f"设备不匹配: 输入x在{x.device}, 滤波器在{self.device}, 自动转换")
            x = x.to(self.device)
        
        # 形状检查和调整
        original_shape = x.shape
        if x.dim() == 1:
            x = x.unsqueeze(1)  # 确保是2D: (num_joints, 1) 或 (batch, num_joints)
        
        # 检查形状兼容性
        if x.shape[-1] != 1 and x.shape[0] != self.num_joints:
            # 尝试自动reshape
            if x.numel() == self.num_joints:
                x = x.reshape(self.num_joints, 1)
                logger.info(f"自动reshape输入: {original_shape} -> {x.shape}")
            else:
                raise ValueError(f"输入形状{x.shape}与滤波器关节数{self.num_joints}不兼容")
        
        # 数值检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            logger.error(f"输入包含非法数值: NaN={torch.isnan(x).any()}, Inf={torch.isinf(x).any()}")
            # 安全处理：清理非法数值
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 确保内存连续性
        if not x.is_contiguous():
            x = x.contiguous()
        if not self.b_torch.is_contiguous():
            self.b_torch = self.b_torch.contiguous()
        if not self.a_torch.is_contiguous():
            self.a_torch = self.a_torch.contiguous()
        if not self.xhist_torch.is_contiguous():
            self.xhist_torch = self.xhist_torch.contiguous()
        if not self.yhist_torch.is_contiguous():
            self.yhist_torch = self.yhist_torch.contiguous()
        
        # 核心滤波计算
        # 注意：这里假设x的形状是(num_joints, 1)或(batch, num_joints)
        if x.dim() == 2 and x.shape[1] == 1:
            # 单个样本: (num_joints, 1)
            x_flat = x.squeeze(1)
        else:
            # 批量处理或其他形状
            x_flat = x
        
        # 执行滤波计算
        y = x_flat * self.b_torch[:, 0] \
              + torch.sum(self.xhist_torch.T * self.b_torch[:, 1:], dim=1) \
              - torch.sum(self.yhist_torch.T * self.a_torch[:, 1:], dim=1)
        
        # 更新历史记录
        self.xhist_torch = torch.cat([x_flat.unsqueeze(0), self.xhist_torch[:-1]], dim=0)
        self.yhist_torch = torch.cat([y.unsqueeze(0), self.yhist_torch[:-1]], dim=0)
        
        # 输出验证
        if torch.isnan(y).any() or torch.isinf(y).any():
            logger.error("滤波器输出包含非法数值，使用安全回退")
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return y
        
    except Exception as e:
        logger.error(f"滤波器计算错误: {e}")
        logger.error(f"调试信息: x.shape={x.shape if 'x' in locals() else 'N/A'}")
        logger.error(f"a_torch.shape={self.a_torch.shape}, b_torch.shape={self.b_torch.shape}")
        logger.error(f"xhist_torch.shape={self.xhist_torch.shape}, yhist_torch.shape={self.yhist_torch.shape}")
        
        # 安全回退：返回输入值（跳过滤波）
        if 'x' in locals():
            return x.squeeze() if x.dim() > 1 else x
        else:
            return torch.zeros(self.num_joints, device=self.device)
  
  def reset_hist(self, action_ids):
    """重置历史记录"""
    if action_ids is not None:
        self.xhist_torch[:, action_ids] = 0.
        self.yhist_torch[:, action_ids] = 0.
    else:
        # 完全重置
        self.xhist_torch.zero_()
        self.yhist_torch.zero_()
    
    logger.info("滤波器历史记录已重置")

  def get_debug_info(self):
    """获取调试信息"""
    return {
        'device': str(self.device),
        'num_joints': self.num_joints,
        'a_torch_shape': str(self.a_torch.shape),
        'b_torch_shape': str(self.b_torch.shape),
        'xhist_shape': str(self.xhist_torch.shape),
        'yhist_shape': str(self.yhist_torch.shape),
        'a_torch_device': str(self.a_torch.device),
        'b_torch_device': str(self.b_torch.device)
    }

class ActionFilterExp(ActionFilter):
  """Filter by way of simple exponential smoothing.

  y = alpha * x + (1 - alpha) * previous_y
  """

  def __init__(self, alpha, num_joints):
    """Initialize the filter.

    Args:
      alpha: list of strings defining the alphas.
        The list must contain either 1 element (same filter for all joints)
        or num_joints elements
        0 < alpha <= 1
      num_joints: robot DOF
    """
    self.alphas = [float(x) for x in alpha]
    logger.info('Exponential filter: alpha: %d', self.alphas)

    a_coeffs = []
    b_coeffs = []
    for a in self.alphas:
      a_coeffs.append(np.asarray([1., a - 1.]))
      b_coeffs.append(np.asarray([a, 0]))

    order = 1
    self.ftype = 'lowpass'

    super(ActionFilterExp, self).__init__(
        a_coeffs, b_coeffs, order, num_joints, self.ftype)