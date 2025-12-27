
import torch
import torch.nn as nn
import torch.nn.functional as F


class ESMC_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1152, 768, 1), 
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(768, 512, 1),  
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Conv1d(512, 256, 1),  
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return self.encoder(x.permute(0,2,1))  # [B,256,L]


class DSSP_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dssp_embed = nn.Embedding(8, 64)
        self.dssp_proj = nn.Sequential(
            nn.Conv1d(8, 32, 1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, 1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.3)
        )       
        
    def forward(self, x):

        indices = torch.argmax(x, dim=-1)  #  [B, L]
        embed_feat = self.dssp_embed(indices).permute(0, 2, 1)  # [B, 32, L]
        proj_feat = self.dssp_proj(x.permute(0, 2, 1))          # [B, 32, L]
        return F.gelu(embed_feat + proj_feat)
        


class Bfactor_Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.b_gate = nn.Sequential(
            nn.Conv1d(1, 32, 1),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),
            nn.Dropout(0.3)
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
        return self.b_gate(b.permute(0,2,1))*self.b_proj(b.permute(0,2,1))  # [B,16,L]

        
class SF_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 1),
            nn.BatchNorm1d(16),
            nn.Sigmoid(),  
            nn.Dropout(0.3),
            nn.Conv1d(16, 32, 1),
            nn.BatchNorm1d(32),
            nn.Sigmoid(),  
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
        self.sf_processor = SF_Encoder()

        self.post_concat_dropout = nn.Dropout1d(0.3)
        
        self.resblock1 = ResidualBlock(init_dims, 256, 5)
        self.pool1 = nn.MaxPool1d(2, ceil_mode=True)
        self.resblock2 = ResidualBlock(256, 128, 3)

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )


    def forward(self, esmc=None, dssp=None, b=None, sf=None, mask=None, return_features=False):
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

        if not feat_list:
            raise ValueError("必须至少提供一个输入")
        
        x = torch.cat(feat_list, dim=1)  # [B, _, L]
        x = self.post_concat_dropout(x)

        x = self.resblock1(x)        # [B, 256, L]
        x = self.pool1(x)            # [B, 256, L//2]
        x = self.resblock2(x)        # [B, 128, L//2]

        # === Mask-aware Pooling ===
        if mask is not None:
            mask = mask[:, None, :]  # [B, 1, L]
            mask = F.avg_pool1d(mask.float(), kernel_size=2, ceil_mode=True) 
            x = x * mask 
            x = torch.max(x, dim=-1).values 
        else:
            x = F.adaptive_max_pool1d(x, 1).squeeze(-1) # [B, 128]

        if return_features:
            return x
            
        return self.classifier(x)

