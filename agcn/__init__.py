# agcn/__init__.py

# 导入核心模型类，使其可以直接从 'agcn' 包导入
from .model import Model_2sAGCN, AGCN_Single_Stream, AGCN_Block

# 导入数据加载和工具函数
from .dataset import SkeletonDataset
from .utils import load_config, get_adj_matrix
from .engine import train_one_epoch, evaluate

# 定义 __all__，指定 `from agcn import *` 时导入的模块
__all__ = [
    'Model_2sAGCN',
    'AGCN_Single_Stream',
    'AGCN_Block',
    'SkeletonDataset',
    'load_config',
    'get_adj_matrix',
    'train_one_epoch',
    'evaluate'
]