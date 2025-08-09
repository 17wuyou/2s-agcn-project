# agcn/model.py

import torch
import torch.nn as nn

# =================================================================================
# 零件1: 自适应图卷积层 (核心创新)
# =================================================================================
class AdaptiveGraphConv(nn.Module):
    """
    实现了论文中 `A_k + B_k + C_k` 逻辑的自适应图卷积层。
    """
    def __init__(self, in_channels, out_channels, num_nodes, num_subsets=3):
        super(AdaptiveGraphConv, self).__init__()
        
        self.num_subsets = num_subsets
        self.num_nodes = num_nodes

        # 权重函数 W_k，用一个1x1卷积实现
        self.conv_W = nn.Conv2d(in_channels, out_channels * num_subsets, kernel_size=1)

        # B_k (全局自适应图): 可学习的参数矩阵
        # 初始化为0，以确保训练初期等价于ST-GCN
        self.B = nn.Parameter(torch.zeros(num_subsets, num_nodes, num_nodes))

        # C_k (样本自适应图): 通过嵌入函数θ和φ生成
        # 使用1x1卷积作为嵌入函数
        # 为了简化，假设嵌入后的通道数与out_channels相同
        embed_channels = out_channels 
        self.conv_theta = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.conv_phi = nn.Conv2d(in_channels, embed_channels, kernel_size=1)

    def forward(self, x, A):
        """
        x: 输入特征图, 尺寸 (N, C_in, T, V)
        A: 固定的物理邻接矩阵, 尺寸 (num_subsets, V, V)
        """
        N, C_in, T, V = x.shape

        # --- 计算 C_k (样本自适应图) ---
        # 1. 嵌入
        theta_x = self.conv_theta(x).permute(0, 2, 3, 1).contiguous().view(N, T * V, -1) # (N, T*V, C_embed)
        phi_x = self.conv_phi(x).view(N, -1, T * V) # (N, C_embed, T*V)
        
        # 2. 计算相似度矩阵
        # (N, T*V, C_embed) @ (N, C_embed, T*V) -> (N, T*V, T*V)
        # 这会产生巨大的计算量，论文中的实现技巧是只在空间维度上做
        # 让我们遵循更高效的实现，只在V上计算
        theta_x_spatial = self.conv_theta(x).mean(dim=2) # (N, C_embed, V)
        phi_x_spatial = self.conv_phi(x).mean(dim=2)   # (N, C_embed, V)
        
        similarity_matrix = torch.einsum('ncv, ncw -> nvw', theta_x_spatial, phi_x_spatial) # (N, V, V)

        # 3. 归一化得到 C_k
        C = torch.softmax(similarity_matrix, dim=-1) # (N, V, V)
        C = C.unsqueeze(1).repeat(1, self.num_subsets, 1, 1) # (N, num_subsets, V, V)

        # --- 组合最终的图 ---
        # A和B需要广播到batch维度
        # A: (num_subsets, V, V) -> (1, num_subsets, V, V)
        # B: (num_subsets, V, V) -> (1, num_subsets, V, V)
        adaptive_graph = A.unsqueeze(0) + self.B.unsqueeze(0) + C

        # --- 执行图卷积 ---
        # 1. 聚合邻居特征
        aggregated_features = torch.einsum('nctv, nsvw -> nsctw', x, adaptive_graph)

        # 2. 应用权重函数 W_k (1x1卷积)
        z = aggregated_features.contiguous().view(N, self.num_subsets * C_in, T, V)
        feature_maps = self.conv_W(z)

        # 3. 将不同子集的结果相加
        feature_maps = feature_maps.view(N, self.num_subsets, -1, T, V)
        out = torch.sum(feature_maps, dim=1) # (N, C_out, T, V)

        return out

# =================================================================================
# 零件2: 时间卷积层
# =================================================================================
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9):
        super(TemporalConv, self).__init__()
        
        padding = (kernel_size - 1) // 2
        self.conv_t = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=(kernel_size, 1), 
                                padding=(padding, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        # ReLU在块级别应用
        return x

# =================================================================================
# 模块: 自适应图卷积块
# =================================================================================
class AGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A, stride=1):
        super(AGCN_Block, self).__init__()

        self.A = A

        # 空间卷积部分
        self.conv_s = AdaptiveGraphConv(in_channels, out_channels, num_nodes)
        self.bn_s = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5, inplace=True)

        # 时间卷积部分
        self.conv_t = TemporalConv(out_channels, out_channels)

        # 残差连接
        if stride != 1 or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        
        x = self.conv_s(x, self.A)
        x = self.bn_s(x)
        x = self.relu(x)
        
        x = self.conv_t(x)
        x = self.dropout(x) # 遵循一些实现，dropout放在时间卷积后
        
        x = x + res
        x = self.relu(x)

        return x

# =================================================================================
# 组件: 单流AGCN网络
# =================================================================================
class AGCN_Single_Stream(nn.Module):
    def __init__(self, in_channels, num_nodes, num_classes, A):
        super(AGCN_Single_Stream, self).__init__()
        
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        
        channel_configs = [64, 64, 64, 128, 128, 128, 256, 256, 256]
        strides = [1, 1, 1, 2, 1, 1, 2, 1, 1]
        
        self.layers = nn.ModuleList()
        last_channels = in_channels
        for channels, stride in zip(channel_configs, strides):
            self.layers.append(AGCN_Block(last_channels, channels, num_nodes, A, stride))
            last_channels = channels

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channels, num_classes)

    def forward(self, x):
        N, C, T, V = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()

        for layer in self.layers:
            x = layer(x)

        x = self.global_pool(x).squeeze(-1).squeeze(-1) # (N, C_out)
        x = self.fc(x)

        return x

# =================================================================================
# 成品: 最终的双流模型
# =================================================================================
class Model_2sAGCN(nn.Module):
    def __init__(self, num_joints, num_classes, A, bone_pairs):
        super(Model_2sAGCN, self).__init__()

        self.j_stream = AGCN_Single_Stream(in_channels=3, num_nodes=num_joints, num_classes=num_classes, A=A)
        self.b_stream = AGCN_Single_Stream(in_channels=3, num_nodes=num_joints, num_classes=num_classes, A=A)
        self.bone_pairs = bone_pairs

    def forward(self, x):
        # x 是关节数据 (N, C, T, V)
        j_logits = self.j_stream(x)

        bone_data = torch.zeros_like(x)
        for v1, v2 in self.bone_pairs:
            bone_data[:, :, :, v2] = x[:, :, :, v2] - x[:, :, :, v1]
        
        b_logits = self.b_stream(bone_data)

        # 融合: 直接相加logits，这在实践中比相加scores更常见且数值稳定
        final_logits = j_logits + b_logits
        
        # 返回logits，让CrossEntropyLoss处理softmax
        return final_logits