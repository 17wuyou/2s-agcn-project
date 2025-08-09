# agcn/dataset.py (模拟数据版本)

import torch
from torch.utils.data import Dataset
import numpy as np

class SkeletonDataset(Dataset):
    """
    骨骼动作识别数据集的模拟版本。
    
    这个类目前不从文件中加载数据，而是动态生成随机数据。
    这对于在没有真实数据的情况下测试模型的训练流程非常有用。
    
    当你准备好使用真实数据时，你需要修改这个类的内部实现来加载你的数据文件。
    """
    def __init__(self, config, mode='train'):
        """
        初始化数据集。
        
        Args:
            config (dict): 从 config.yaml 加载的配置字典。
            mode (str): 'train' 或 'val'，用于区分训练集和验证集。
        """
        print(f"Initializing SkeletonDataset in '{mode}' mode with MOCK data.")
        
        # 从配置中获取关键参数
        self.num_joints = config['data']['num_joints']
        self.num_classes = config['data']['num_classes']
        self.num_frames = 300 # 假设所有样本填充到300帧
        in_channels = config['model']['in_channels']

        # 根据模式确定样本数量
        if mode == 'train':
            self.num_samples = 100 # 生成100个训练样本
        else:
            self.num_samples = 30  # 生成30个验证样本
        
        # --- 生成模拟数据和标签 ---
        # data 的维度: [样本数, 通道数, 帧数, 关节点数]
        print(f"Generating {self.num_samples} random samples...")
        self.data = torch.randn(self.num_samples, in_channels, self.num_frames, self.num_joints)
        
        # labels 的维度: [样本数]
        self.labels = torch.randint(0, self.num_classes, (self.num_samples,))
        
        print("Mock data generation complete.")

    def __len__(self):
        """返回数据集中的样本总数"""
        return self.num_samples

    def __getitem__(self, index):
        """
        根据索引获取一个样本。
        这是PyTorch DataLoader调用的核心接口。
        """
        # 从模拟数据中获取一个样本和其对应的标签
        data_sample = self.data[index]
        label_sample = self.labels[index]
        
        return data_sample, label_sample