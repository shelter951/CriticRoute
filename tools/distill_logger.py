"""
DistillLogger: 用于记录教师模型的蒸馏数据
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import threading


class DistillLogger:
    """用于记录教师模型蒸馏数据的日志类"""
    
    def __init__(self, output_dir: str, task_name: str, split: str, enabled: bool = True):
        """
        Args:
            output_dir: 输出目录
            task_name: 任务名称 (如 'CVDN', 'R2R')
            split: 数据集划分 (如 'train', 'val_seen', 'val_unseen')
            enabled: 是否启用日志记录
        """
        self.enabled = enabled
        if not enabled:
            return
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建输出文件路径
        filename = f"distill_{task_name.lower()}_{split}.jsonl"
        self.output_file = self.output_dir / filename
        
        # 线程锁，确保多进程/多线程安全
        self.lock = threading.Lock()
        
        # 打开文件（追加模式）
        self.file_handle = open(self.output_file, 'a', encoding='utf-8')
        
    def log_episode(self, episode_data: Dict):
        """
        记录一个完整的episode数据
        
        Args:
            episode_data: episode数据字典，格式：
            {
                "task": "CVDN",
                "split": "train",
                "episode_id": "CVDN_12345",
                "success": 1,
                "final_sr": 1,
                "final_spl": 0.73,
                "steps": [...]
            }
        """
        if not self.enabled:
            return
            
        try:
            with self.lock:
                # 写入JSONL格式（一行一个JSON）
                json_str = json.dumps(episode_data, ensure_ascii=False)
                self.file_handle.write(json_str + '\n')
                self.file_handle.flush()  # 立即刷新到磁盘
        except Exception as e:
            print(f"Error writing distill log: {e}")
    
    def close(self):
        """关闭文件句柄"""
        if self.enabled and hasattr(self, 'file_handle'):
            self.file_handle.close()
    
    def __del__(self):
        """析构函数，确保文件被关闭"""
        self.close()


