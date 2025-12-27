
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMC_Encoder(nn.Module):
    """渐進式降维处理结构"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1152, 768, 1),  # 第1阶段降维
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(768, 512, 1),   # 过渡层
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(512, 256, 1),   # 最终维度
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.encoder(x.permute(0,2,1))  # [B,256,L]


class DSSP_Encoder(nn.Module):
    """二级结构类别特征嵌入"""
    def __init__(self):
        super().__init__()
        self.dssp_embed = nn.Embedding(8, 64)            # 类别嵌入
        self.dssp_proj = nn.Sequential(
            nn.Conv1d(8, 32, 1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3)
        )        # 原始特征保留
        
    def forward(self, x):
        # 输入shape: [B, L, 8] (one-hot)
        indices = torch.argmax(x, dim=-1)  # 获取类别索引 [B, L]
        embed_feat = self.dssp_embed(indices).permute(0, 2, 1)  # [B, 32, L]
        proj_feat = self.dssp_proj(x.permute(0, 2, 1))          # [B, 32, L]
        return F.gelu(embed_feat + proj_feat)
        
#class MC_Encoder(nn.Module):
#    """特征嵌入"""
#    def __init__(self):
#        super().__init__()
#        self.mc_embed = nn.Embedding(17, 32)            # 类别嵌入
#        self.mc_proj = nn.Sequential(
#            nn.Conv1d(17, 32, 1),
#            nn.BatchNorm1d(32),
#            nn.GELU(),
#            nn.Dropout(0.3)
#        )        # 原始特征保留        
#    def forward(self, x):
        # 输入shape: [B,L,8] (one-hot)
#        embed_feat = self.mc_embed(torch.argmax(x, dim=-1)).permute(0,2,1)  # [B,24,L]
#        proj_feat = self.mc_proj(x.permute(0,2,1)) # [B,24,L]
#        return F.gelu(embed_feat + proj_feat)       # 残差融合


class Bfactor_Gate(nn.Module):
    """基于B因子的稳定性门控机制"""
    def __init__(self):
        super().__init__()
        self.b_gate = nn.Sequential(
            nn.Conv1d(1, 32, 1),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(0.3)# 映射到0~1区间
        )
        
        self.b_proj = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
    def forward(self, b):
        # 输入shape: [B,L,1] 
        return self.b_gate(b.permute(0,2,1))*self.b_proj(b.permute(0,2,1))  # [B,16,L]

        
class SF_Encoder(nn.Module):
    """处理概率型特征的公共组件"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),  # 维持概率特性
            nn.Dropout(0.3),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),  # 维持概率特性
            nn.Dropout(0.3)
        )
        
    def forward(self, x):
        return self.net(x.permute(0,2,1))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.proj(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class MultiChannelResCNN(nn.Module):
    def __init__(self, init_dims):
        super().__init__()

        self.esmc_branch = ESMC_Encoder()
        self.dssp_branch = DSSP_Encoder()
        self.b_gate = Bfactor_Gate()
        #self.mc_processor = MC_Encoder()
        self.sf_processor = SF_Encoder()

        self.post_concat_dropout = nn.Dropout1d(0.3)
        
        # 残差主干结构
        self.resblock1 = ResidualBlock(init_dims, 256, 5)  # 输入通道为拼接后大小
        self.pool1 = nn.MaxPool1d(2, ceil_mode=True)
        self.resblock2 = ResidualBlock(256, 128, 3)

        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )


    def forward(self, esmc=None, dssp=None, b=None, sf=None, mask=None, return_features=False):
        # 动态收集存在的特征
        feat_list = []
        if esmc is not None:
            esmc_feat = self.esmc_branch(esmc)  # [B,256,L]
            feat_list.append(esmc_feat)
        if dssp is not None:
            dssp_feat = self.dssp_branch(dssp)  # [B,64,L]
            feat_list.append(dssp_feat)
        if b is not None:
            b_feat = self.b_gate(b)  # [B,32,L]
            feat_list.append(b_feat)
        if sf is not None:
            sf_feat = self.sf_processor(sf)  # [B,32,L]
            feat_list.append(sf_feat)
        
        # 检查至少有一个输入
        if not feat_list:
            raise ValueError("必须至少提供一个输入")
        
        # 拼接存在的特征（通道数动态变化）
        x = torch.cat(feat_list, dim=1)  # [B, _, L]
        x = self.post_concat_dropout(x)
        
        # 主干网络
        x = self.resblock1(x)        # [B, 256, L]
        x = self.pool1(x)            # [B, 256, L//2]
        x = self.resblock2(x)        # [B, 128, L//2]

        # === Mask-aware Pooling ===
        if mask is not None:
            mask = mask[:, None, :]  # [B, 1, L]
            mask = F.avg_pool1d(mask.float(), kernel_size=2, ceil_mode=True)  # 下采样掩码
            x = x * mask  # 用连续掩码值加权特征
            x = torch.max(x, dim=-1).values  # 加权后的最大池化
        else:
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1) # [B, 128]

        if return_features:
            return x
            
        return self.classifier(x)

