# agcn/dataset.py (真实数据版本)

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle
import os

class SkeletonDataset(Dataset):
    """
    加载预处理后的骨骼动作识别数据集。
    """
    def __init__(self, config, mode='train'):
        """
        初始化数据集。
        
        Args:
            config (dict): 从 config.yaml 加载的配置字典。
            mode (str): 'train' 或 'val'。
        """
        print(f"Initializing SkeletonDataset in '{mode}' mode from REAL data files.")
        
        data_path = config['data']['data_path'] # e.g., './generated_data'
        
        # 根据模式选择加载的文件
        if mode == 'train':
            data_file = os.path.join(data_path, 'train_data.npy')
            label_file = os.path.join(data_path, 'train_label.pkl')
        elif mode == 'val':
            data_file = os.path.join(data_path, 'val_data.npy')
            label_file = os.path.join(data_path, 'val_label.pkl')
        else:
            raise ValueError("Mode must be 'train' or 'val'")

        # --- 加载真实数据和标签 ---
        print(f"Loading data from: {data_file}")
        self.data = np.load(data_file)
        
        print(f"Loading labels from: {label_file}")
        with open(label_file, 'rb') as f:
            # 假设 pickle 文件中存储的是一个元组 (sample_names, labels)
            self.sample_name, self.labels = pickle.load(f)

        # 检查是否有M（人数）维度，如果有，则简化
        if self.data.ndim == 5 and self.data.shape[-1] > 1:
            print("Data has 5 dimensions, selecting data for the first person.")
            self.data = self.data[:, :, :, :, 0]
        
        print("Data loading complete.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 将Numpy数据转换为Tensor
        data_sample = torch.from_numpy(self.data[index]).float()
        label_sample = torch.tensor(self.labels[index]).long()
        
        return data_sample, label_sample