# agcn/dataset.py (真实数据版本)

import torch
from torch.utils.data import Dataset
import numpy as np
import pickle

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
        print(f"Initializing SkeletonDataset in '{mode}' mode from REAL data.")
        
        data_path = config['data']['data_path'] # e.g., './data/ntu60_xsub'
        label_path = config['data']['label_path'] # e.g., './data/ntu60_xsub'
        
        # 根据模式选择加载的文件
        if mode == 'train':
            data_file = f"{data_path}/train_data.npy"
            label_file = f"{label_path}/train_label.pkl"
        elif mode == 'val':
            data_file = f"{data_path}/val_data.npy"
            label_file = f"{label_path}/val_label.pkl"
        else:
            raise ValueError("Mode must be 'train' or 'val'")

        # --- 加载真实数据和标签 ---
        print(f"Loading data from: {data_file}")
        self.data = np.load(data_file)
        
        print(f"Loading labels from: {label_file}")
        with open(label_file, 'rb') as f:
            self.sample_name, self.labels = pickle.load(f)

        # 数据形状通常是 (N, C, T, V, M) -> [样本数, 通道, 帧, 关节, 人数]
        # 我们只使用第一个人 (M=0) 的数据来简化
        if self.data.shape[-1] > 1:
            self.data = self.data[:, :, :, :, 0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # 将Numpy数据转换为Tensor
        data_sample = torch.from_numpy(self.data[index]).float()
        label_sample = torch.tensor(self.labels[index]).long()
        
        return data_sample, label_sample