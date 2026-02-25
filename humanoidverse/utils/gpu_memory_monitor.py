import torch
import time
import os
import subprocess
import re
import json
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import gc
from datetime import datetime
import psutil


class GPUMemoryMonitor:
    """
    GPU内存监控类 - 保持原有接口
    
    使用方式：
    monitor = GPUMemoryMonitor(log_interval=10, detail_interval=100)
    
    在训练循环中：
    1. monitor.before_iteration(iteration)
    2. monitor.after_rollout(iteration) 
    3. monitor.after_training(iteration)
    """
    
    def __init__(
        self,
        log_interval: int = 10,
        detail_interval: int = 100,
        threshold_mb: float = 10.0,
        log_dir: str = "./gpu_monitor_logs",
        check_warnings: bool = True
    ):
        """
        初始化GPU内存监控器
        
        Args:
            log_interval: 日志打印间隔（迭代次数）
            detail_interval: 详细分析间隔
            threshold_mb: 大张量阈值（MB）
            log_dir: 日志保存目录
            check_warnings: 是否检查警告
        """
        self.log_interval = log_interval
        self.detail_interval = detail_interval
        self.threshold_mb = threshold_mb
        self.log_dir = log_dir
        self.check_warnings = check_warnings
        
        # 状态变量
        self.iteration = 0
        self.memory_history = []
        self.gpu_stats_history = []
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 检查CUDA可用性
        self.cuda_available = torch.cuda.is_available()
        if not self.cuda_available:
            print("警告: CUDA不可用，GPU监控功能受限")
        else:
            self.num_gpus = torch.cuda.device_count()
            print(f"GPU监控器初始化: 检测到 {self.num_gpus} 个GPU设备")
        
        # 禁用不需要的警告
        if check_warnings:
            import warnings
            warnings.filterwarnings('ignore', message='To copy construct from a tensor')
    
    def _get_nvidia_smi_memory(self):
        """获取nvidia-smi内存信息"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                gpu_memories = []
                lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                
                for i, line in enumerate(lines):
                    if ',' in line:
                        used, total = line.split(',')
                        gpu_memories.append({
                            'device_id': i,
                            'used_mb': float(used.strip()),
                            'total_mb': float(total.strip())
                        })
                return gpu_memories
        except Exception as e:
            if self.check_warnings:
                print(f"获取nvidia-smi信息失败: {e}")
        return []
    
    def _get_pytorch_memory_stats(self, device_id: int):
        """获取PyTorch内存统计"""
        try:
            allocated = torch.cuda.memory_allocated(device_id) / 1024**2
            reserved = torch.cuda.memory_reserved(device_id) / 1024**2
            max_allocated = torch.cuda.max_memory_allocated(device_id) / 1024**2
            
            return {
                'allocated_mb': allocated,
                'reserved_mb': reserved,
                'max_allocated_mb': max_allocated
            }
        except Exception as e:
            if self.check_warnings:
                print(f"获取GPU {device_id}的PyTorch内存统计失败: {e}")
            return {
                'allocated_mb': 0,
                'reserved_mb': 0,
                'max_allocated_mb': 0
            }
    
    def _get_gpu_comparison(self):
        """获取GPU对比信息"""
        if not self.cuda_available:
            return []
        
        nvidia_info = self._get_nvidia_smi_memory()
        comparison = []
        
        for i in range(self.num_gpus):
            pytorch_stats = self._get_pytorch_memory_stats(i)
            nvidia_mem = next((info['used_mb'] for info in nvidia_info if info['device_id'] == i), 0)
            nvidia_total = next((info['total_mb'] for info in nvidia_info if info['device_id'] == i), 0)
            
            torch_mem = pytorch_stats['allocated_mb']
            diff = nvidia_mem - torch_mem
            diff_ratio = diff / nvidia_mem if nvidia_mem > 0 else 0
            
            comparison.append({
                'device_id': i,
                'nvidia_used_mb': nvidia_mem,
                'nvidia_total_mb': nvidia_total,
                'torch_allocated_mb': torch_mem,
                'torch_reserved_mb': pytorch_stats['reserved_mb'],
                'torch_max_allocated_mb': pytorch_stats['max_allocated_mb'],
                'difference_mb': diff,
                'difference_ratio': diff_ratio,
                'usage_percentage': (nvidia_mem / nvidia_total * 100) if nvidia_total > 0 else 0
            })
        
        return comparison
    
    def _print_memory_summary(self, step_name: str = ""):
        """打印内存摘要"""
        if not self.cuda_available:
            return
        
        comparison = self._get_gpu_comparison()
        
        if step_name:
            print(f"\n[{step_name}] 迭代 {self.iteration}")
        else:
            print(f"\n迭代 {self.iteration}")
        
        print("PyTorch统计:")
        torch_allocated = sum(c['torch_allocated_mb'] for c in comparison) / len(comparison) if comparison else 0
        torch_reserved = sum(c['torch_reserved_mb'] for c in comparison) / len(comparison) if comparison else 0
        torch_max = sum(c['torch_max_allocated_mb'] for c in comparison) / len(comparison) if comparison else 0
        print(f"  分配: {torch_allocated:.1f}MB / 预留: {torch_reserved:.1f}MB / 峰值: {torch_max:.1f}MB")
        
        print("nvidia-smi统计:")
        for comp in comparison:
            print(f"  GPU {comp['device_id']}: {comp['nvidia_used_mb']:.0f}MB / {comp['nvidia_total_mb']:.0f}MB "
                  f"({comp['usage_percentage']:.1f}%)")
        
        print("内存差异分析:")
        for comp in comparison:
            if comp['torch_allocated_mb'] > 0:
                print(f"  GPU {comp['device_id']}: nvidia-smi - PyTorch = {comp['difference_mb']:.1f}MB "
                      f"({comp['nvidia_used_mb']/comp['torch_allocated_mb']:.1f}x)")
    
    def _print_detailed_analysis(self):
        """打印详细分析"""
        if not self.cuda_available:
            return
        
        comparison = self._get_gpu_comparison()
        
        print("\n" + "="*80)
        print("GPU内存详细对比分析")
        print("="*80)
        
        total_torch = sum(c['torch_allocated_mb'] for c in comparison)
        total_nvidia = sum(c['nvidia_used_mb'] for c in comparison)
        total_diff = total_nvidia - total_torch
        
        print(f"\n总计 ({self.num_gpus}个GPU):")
        print(f"  nvidia-smi总内存: {total_nvidia:.1f} MB ({total_nvidia/1024:.2f} GB)")
        print(f"  PyTorch分配内存: {total_torch:.1f} MB ({total_torch/1024:.2f} GB)")
        print(f"  总差异: {total_diff:.1f} MB ({total_diff/1024:.2f} GB)")
        
        if total_diff > 0:
            print(f"\n内存差异分析 ({total_diff/1024:.1f}GB):")
            print("  1. CUDA上下文: 每个GPU约1-2GB")
            print("  2. 内存碎片: 分配释放产生的空洞")
            print("  3. CUDA内核缓存: 已编译的内核代码")
            print("  4. PyTorch内存池: 为提高性能保留")
            print("  5. 其他CUDA运行时组件")
    
    def _analyze_tensor_memory(self, summary_only=False):
        """分析张量内存"""
        if not self.cuda_available:
            return {}
        
        print("\n" + "="*60)
        print("GPU张量内存详细分析")
        print("="*60)
        
        # 获取内存统计
        comparison = self._get_gpu_comparison()
        if comparison:
            torch_allocated = sum(c['torch_allocated_mb'] for c in comparison) / len(comparison)
            torch_reserved = sum(c['torch_reserved_mb'] for c in comparison) / len(comparison)
            print(f"当前分配内存: {torch_allocated:.2f} MB")
            print(f"预留内存: {torch_reserved:.2f} MB")
        
        if summary_only:
            print("="*60)
            return {}
        
        # 分析所有张量
        tensor_stats = defaultdict(lambda: {
            'count': 0,
            'total_size_mb': 0.0,
            'shapes': [],
            'dtypes': set()
        })
        
        total_tensors = 0
        total_size_mb = 0
        large_tensors = []
        
        # 收集所有GPU张量
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    tensor_size_bytes = obj.element_size() * obj.nelement()
                    tensor_size_mb = tensor_size_bytes / 1024**2
                    total_tensors += 1
                    total_size_mb += tensor_size_mb
                    
                    # 按张量类型统计
                    tensor_type = type(obj).__name__
                    tensor_stats[tensor_type]['count'] += 1
                    tensor_stats[tensor_type]['total_size_mb'] += tensor_size_mb
                    tensor_stats[tensor_type]['shapes'].append(obj.shape)
                    tensor_stats[tensor_type]['dtypes'].add(str(obj.dtype))
                    
                    # 记录大张量
                    if tensor_size_mb > self.threshold_mb:
                        large_tensors.append({
                            'type': tensor_type,
                            'size_mb': tensor_size_mb,
                            'shape': obj.shape,
                            'dtype': obj.dtype,
                            'device': obj.device
                        })
            
            except Exception as e:
                continue
        
        # 按类型显示统计
        print(f"\n总计: {total_tensors} 个GPU张量, 总大小: {total_size_mb:.2f} MB")
        
        if tensor_stats:
            print("\n按类型统计:")
            print("-"*60)
            print(f"{'类型':<20} {'数量':<8} {'总大小(MB)':<12} {'平均大小(MB)':<12}")
            print("-"*60)
            
            for tensor_type, stats in sorted(tensor_stats.items(), 
                                             key=lambda x: x[1]['total_size_mb'], 
                                             reverse=True):
                avg_size = stats['total_size_mb'] / stats['count'] if stats['count'] > 0 else 0
                print(f"{tensor_type:<20} {stats['count']:<8} {stats['total_size_mb']:<12.2f} {avg_size:<12.2f}")
        
        # 显示大张量详情
        if large_tensors:
            print(f"\n大于 {self.threshold_mb}MB 的张量详情:")
            print("-"*60)
            print(f"{'序号':<6} {'类型':<15} {'大小(MB)':<12} {'形状':<25} {'数据类型':<10}")
            print("-"*60)
            
            large_tensors.sort(key=lambda x: x['size_mb'], reverse=True)
            for i, tensor_info in enumerate(large_tensors[:10], 1):  # 只显示前10个
                shape_str = str(tensor_info['shape'])
                if len(shape_str) > 25:
                    shape_str = shape_str[:22] + "..."
                
                print(f"{i:<6} {tensor_info['type']:<15} {tensor_info['size_mb']:<12.2f} "
                      f"{shape_str:<25} {str(tensor_info['dtype']):<10}")
        
        print("="*60)
        
        return {
            'total_tensors': total_tensors,
            'total_size_mb': total_size_mb,
            'tensor_stats': dict(tensor_stats),
            'large_tensors': large_tensors[:10]
        }
    
    def _optimize_memory(self):
        """优化内存使用"""
        if not self.cuda_available:
            return
        
        print("\n执行内存优化:")
        
        # 清理缓存
        torch.cuda.empty_cache()
        print("  ✓ 清理GPU缓存")
        
        # 重置峰值内存统计
        for i in range(self.num_gpus):
            torch.cuda.reset_peak_memory_stats(i)
        print("  ✓ 重置峰值内存统计")
        
        # 垃圾回收
        gc.collect()
        print("  ✓ 执行垃圾回收")
    
    def before_iteration(self, iteration: int):
        """
        迭代开始前的内存检查
        
        Args:
            iteration: 当前迭代次数
        """
        self.iteration = iteration
        
        if iteration % self.log_interval == 0:
            print(f"\n=== 迭代 {iteration} 开始 ===")
            self._print_memory_summary("开始前")
            
        if iteration % self.detail_interval == 0:
            self._analyze_tensor_memory(summary_only=(iteration % (self.detail_interval * 2) != 0))
    
    def after_rollout(self, iteration: int):
        """
        rollout后的内存检查
        
        Args:
            iteration: 当前迭代次数
        """
        self.iteration = iteration
        
        if iteration % self.log_interval == 0:
            self._print_memory_summary("rollout后")
    
    def after_training(self, iteration: int):
        """
        训练后的内存检查
        
        Args:
            iteration: 当前迭代次数
        """
        self.iteration = iteration
        
        if iteration % self.log_interval == 0:
            self._print_memory_summary("训练后")
            
        if iteration % self.detail_interval == 0:
            # 详细分析
            self._print_detailed_analysis()
            
            # 张量分析
            self._analyze_tensor_memory()
            
            # 重置峰值内存统计
            for i in range(self.num_gpus):
                torch.cuda.reset_peak_memory_stats(i)
            
            # 清理缓存
            if iteration % 50 == 0:
                self._optimize_memory()
    
    def get_current_memory_info(self):
        """获取当前内存信息"""
        if not self.cuda_available:
            return {}
        
        comparison = self._get_gpu_comparison()
        if not comparison:
            return {}
        
        info = {
            'iteration': self.iteration,
            'timestamp': time.time(),
            'gpus': comparison
        }
        
        # 计算总计
        info['total_nvidia_mb'] = sum(c['nvidia_used_mb'] for c in comparison)
        info['total_torch_mb'] = sum(c['torch_allocated_mb'] for c in comparison)
        info['total_difference_mb'] = info['total_nvidia_mb'] - info['total_torch_mb']
        
        return info
    
    def save_memory_history(self):
        """保存内存历史"""
        if not self.memory_history:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.log_dir, f"memory_history_{timestamp}.json")
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.memory_history, f, indent=2, default=str)
            print(f"内存历史已保存到: {filename}")
        except Exception as e:
            if self.check_warnings:
                print(f"保存内存历史失败: {e}")
    
    def cleanup(self):
        """清理资源"""
        print("\nGPU内存监控结束")
        self.save_memory_history()
        
        if self.cuda_available:
            self._optimize_memory()


# 保持原有接口的辅助函数
def analyze_tensor_memory(detail_threshold_mb=10.0, summary_only=False):
    """
    分析GPU上所有张量的内存占用（保持原有接口）
    
    Args:
        detail_threshold_mb: 只显示大于此值的张量详情（MB）
        summary_only: 是否只显示摘要
    """
    monitor = GPUMemoryMonitor(
        log_interval=1,
        detail_interval=1,
        threshold_mb=detail_threshold_mb,
        check_warnings=False
    )
    
    return monitor._analyze_tensor_memory(summary_only)


def print_memory_summary(step_name=""):
    """
    打印内存使用摘要（保持原有接口）
    
    Args:
        step_name: 步骤名称
    """
    monitor = GPUMemoryMonitor(
        log_interval=1,
        detail_interval=1,
        check_warnings=False
    )
    
    monitor._print_memory_summary(step_name)


def get_memory_summary():
    """
    获取内存使用摘要（保持原有接口）
    """
    monitor = GPUMemoryMonitor(
        log_interval=1,
        detail_interval=1,
        check_warnings=False
    )
    
    info = monitor.get_current_memory_info()
    if not info:
        return {}
    
    summary = {
        'allocated_mb': info.get('total_torch_mb', 0),
        'reserved_mb': 0,  # 需要单独计算
        'max_allocated_mb': 0,  # 需要单独计算
    }
    
    # 计算预留内存
    if torch.cuda.is_available():
        reserved_sum = 0
        max_allocated_sum = 0
        for i in range(torch.cuda.device_count()):
            reserved_sum += torch.cuda.memory_reserved(i) / 1024**2
            max_allocated_sum += torch.cuda.max_memory_allocated(i) / 1024**2
        
        summary['reserved_mb'] = reserved_sum
        summary['max_allocated_mb'] = max_allocated_sum
    
    return summary