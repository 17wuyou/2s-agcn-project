# agcn/model.py (Corrected Version)

import torch
import torch.nn as nn

# =================================================================================
# 零件1: 自适应图卷积层 (核心创新) - 无需修改
# =================================================================================
class AdaptiveGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, num_subsets=3):
        super(AdaptiveGraphConv, self).__init__()
        
        self.num_subsets = num_subsets
        self.num_nodes = num_nodes

        self.conv_W = nn.Conv2d(in_channels * num_subsets, out_channels * num_subsets, kernel_size=1)

        self.B = nn.Parameter(torch.zeros(num_subsets, num_nodes, num_nodes))

        embed_channels = out_channels 
        self.conv_theta = nn.Conv2d(in_channels, embed_channels, kernel_size=1)
        self.conv_phi = nn.Conv2d(in_channels, embed_channels, kernel_size=1)

    def forward(self, x, A):
        N, C_in, T, V = x.shape

        theta_x_spatial = self.conv_theta(x).mean(dim=2)
        phi_x_spatial = self.conv_phi(x).mean(dim=2)
        
        similarity_matrix = torch.einsum('ncv, ncw -> nvw', theta_x_spatial, phi_x_spatial)
        C = torch.softmax(similarity_matrix, dim=-1)
        C = C.unsqueeze(1).repeat(1, self.num_subsets, 1, 1)

        adaptive_graph = A.unsqueeze(0) + self.B.unsqueeze(0) + C
        
        aggregated_features = torch.einsum('nctv, nsvw -> nsctw', x, adaptive_graph)
        z = aggregated_features.contiguous().view(N, self.num_subsets * C_in, T, V)
        feature_maps = self.conv_W(z)
        
        feature_maps = feature_maps.view(N, self.num_subsets, -1, T, V)
        out = torch.sum(feature_maps, dim=1)
        return out

# =================================================================================
# 零件2: 时间卷积层 - *** 这里是修改的部分 ***
# =================================================================================
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1): # 1. 增加 stride 参数
        super(TemporalConv, self).__init__()
        
        padding = (kernel_size - 1) // 2
        # 2. 在卷积层中使用 stride 参数
        self.conv_t = nn.Conv2d(in_channels, out_channels, 
                                kernel_size=(kernel_size, 1), 
                                padding=(padding, 0),
                                stride=(stride, 1)) # <-- 应用 stride
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv_t(x)
        x = self.bn(x)
        return x

# =================================================================================
# 模块: 自适应图卷积块 - *** 这里是修改的部分 ***
# =================================================================================
class AGCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A, stride=1):
        super(AGCN_Block, self).__init__()

        self.A = A

        self.conv_s = AdaptiveGraphConv(in_channels, out_channels, num_nodes)
        self.bn_s = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5, inplace=True)

        # 3. 将 stride 传递给 TemporalConv
        self.conv_t = TemporalConv(out_channels, out_channels, stride=stride)

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
        x = self.dropout(x)
        
        x = x + res
        x = self.relu(x)

        return x

# =================================================================================
# 组件和成品部分 - 无需修改
# =================================================================================
class AGCN_Single_Stream(nn.Module):
    # ... (此部分代码保持不变)
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
        x = self.global_pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x

class Model_2sAGCN(nn.Module):
    # ... (此部分代码保持不变)
    def __init__(self, num_joints, num_classes, A, bone_pairs):
        super(Model_2sAGCN, self).__init__()
        self.j_stream = AGCN_Single_Stream(in_channels=3, num_nodes=num_joints, num_classes=num_classes, A=A)
        self.b_stream = AGCN_Single_Stream(in_channels=3, num_nodes=num_joints, num_classes=num_classes, A=A)
        self.bone_pairs = bone_pairs
    def forward(self, x):
        j_logits = self.j_stream(x)
        bone_data = torch.zeros_like(x)
        for v1, v2 in self.bone_pairs:
            bone_data[:, :, :, v2] = x[:, :, :, v2] - x[:, :, :, v1]
        b_logits = self.b_stream(bone_data)
        final_logits = j_logits + b_logits
        return final_logits