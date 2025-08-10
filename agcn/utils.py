import yaml
import torch
import numpy as np

def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

class Graph:
    """
    这个类用于为NTU-RGB+D数据集生成邻接矩阵。
    代码改编自：https://github.com/lshiwjx/2s-AGCN
    """
    def __init__(self, layout='ntu-rgb+d', strategy='spatial'):
        self.num_node = 25
        self.self_link = [(i, i) for i in range(self.num_node)]
        self.inward = [(2, 1), (21, 2), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                       (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15),
                       (17, 1), (18, 17), (19, 18), (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
        self.outward = [(j, i) for (i, j) in self.inward]
        self.neighbor = self.inward + self.outward

        # 1. 生成A_k[0]：单位矩阵（自身连接）
        A_root = self.get_adjacency_matrix(labeling_mode='spatial', t_kernel=1, max_graph_distance=1)
        # 2. 生成A_k[1]和A_k[2]：向心和离心矩阵
        A_close = self.get_adjacency_matrix(labeling_mode='distance', t_kernel=1, max_graph_distance=1)
        A_further = self.get_adjacency_matrix(labeling_mode='distance', t_kernel=1, max_graph_distance=2)

        # 3. 组合成最终的A
        self.A = np.stack([A_root, A_close, A_further])

    def get_adjacency_matrix(self, labeling_mode, t_kernel, max_graph_distance):
        if labeling_mode == 'spatial':
            A = self.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        elif labeling_mode == 'distance':
            A = self.get_distance_graph(self.num_node, self.self_link, self.inward + self.outward)
        else:
            raise ValueError()
        return A

    def get_spatial_graph(self, num_node, self_link, inward, outward):
        I = np.eye(num_node)
        Inward = self.get_adetermined_graph(num_node, self_link, inward)
        Outward = self.get_adetermined_graph(num_node, self_link, outward)
        A = Inward + Outward + I
        return A

    def get_adetermined_graph(self, num_node, self_link, link):
        A = np.zeros((num_node, num_node))
        for i, j in link:
            A[j, i] = 1
        return A

    def get_distance_graph(self, num_node, self_link, neighbor):
        A = np.zeros((num_node, num_node))
        for i in range(num_node):
            for j in range(num_node):
                if self.get_hop_distance(num_node, neighbor, i, j) <= 1:
                    A[i, j] = 1
        return A

    def get_hop_distance(self, num_node, neighbor, start, end):
        # BFS 算法计算跳数距离
        q = [(start, 0)]
        visited = {start}
        while q:
            node, dist = q.pop(0)
            if node == end:
                return dist
            for n_node in [j for i, j in neighbor if i == node]:
                if n_node not in visited:
                    visited.add(n_node)
                    q.append((n_node, dist + 1))
        return float('inf')


def get_adj_matrix(num_joints):
    """主接口函数，返回最终的邻接矩阵Tensor"""
    if num_joints != 25:
        # 这个实现是针对NTU的25个节点的
        raise ValueError("This adjacency matrix implementation is specific to NTU-RGB+D with 25 joints.")
    
    graph = Graph()
    return torch.from_numpy(graph.A).float()