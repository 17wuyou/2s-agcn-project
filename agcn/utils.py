import yaml
import torch
import numpy as np

def load_config(path='config.yaml'):
    """加载YAML配置文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_adj_matrix(num_joints):
    """
    生成邻接矩阵 A_k (k=3)。
    
    !!! 重要提示 !!!
    这是一个演示函数。在实际应用中，你需要根据你使用的数据集
    （如NTU-RGBD）的骨架定义来精确构建这三个矩阵：
    1. A_k[0]: 单位矩阵（自身连接）
    2. A_k[1]: 向心邻接矩阵
    3. A_k[2]: 离心邻接矩阵
    
    这里我们只生成一个代表物理连接的邻接矩阵，并复制3次作为演示。
    """
    # 示例：一个简单的链状骨架连接
    edges = [(i, i + 1) for i in range(num_joints - 1)]
    
    adj = np.zeros((num_joints, num_joints))
    for i, j in edges:
        adj[i, j] = 1
        adj[j, i] = 1
    
    # 加上自身连接
    adj_self = adj + np.eye(num_joints)
    
    # 归一化（可选，但推荐）
    D = np.diag(np.sum(adj_self, axis=1)**(-0.5))
    adj_normalized = D @ adj_self @ D

    # 复制3次作为三个子集的占位符
    # A_k[0] 应该是单位阵，这里为了演示简化
    A_k = np.stack([adj_normalized] * 3)
    
    return torch.from_numpy(A_k).float()