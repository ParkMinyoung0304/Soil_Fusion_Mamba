import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
import math
import time
from engine.logger import get_logger
from models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock

from torch_geometric.nn import GCNConv, GATConv, LayerNorm
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

logger = get_logger()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义GCN模块
class GCNBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNBlock, self).__init__()
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, out_channels)
        self.bn = LayerNorm(out_channels)

    def forward(self, x, edge_index):
        x = self.gcn1(x, edge_index)
        x = F.relu(x)
        x = self.gcn2(x, edge_index)
        x = self.bn(x)
        return x

class RGBXTransformer(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 gcn_dims=[96, 192, 384, 768],
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.ape = ape
        
        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        self.gcn_dims = gcn_dims  # 保存GCN使用的维度
        # 为GCN的每个可能的dims创建GCNBlock
        self.gcn_blocks_rgb = nn.ModuleList([GCNBlock(in_channels=d, hidden_channels=256, out_channels=d) for d in gcn_dims])
        self.gcn_blocks_x = nn.ModuleList([GCNBlock(in_channels=d, hidden_channels=256, out_channels=d) for d in gcn_dims])

        # 将GCN模块移动到设备上
        self.gcn_blocks_rgb.to(DEVICE)
        self.gcn_blocks_x.to(DEVICE)

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=dims * (2 ** i),
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(4)
        )
        
        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

        self.multi_scale_fusion_layers = nn.ModuleList(
            [MultiScaleFeatureFusion(
                in_channels=dims * (2 ** i),
                out_channels=dims * (2 ** i))
                for i in range(4)]
        )
        
    def forward_features(self, x_rgb, x_e):
        B = x_rgb.shape[0]
        outs_fused = []

        outs_rgb = self.vssm(x_rgb)
        outs_x = self.vssm(x_e)

        for i in range(4):
            if self.ape:
                out_rgb = self.absolute_pos_embed[i].to(outs_rgb[i].device) + outs_rgb[i]
                out_x = self.absolute_pos_embed_x[i].to(outs_x[i].device) + outs_x[i]
            else:
                out_rgb = outs_rgb[i]
                out_x = outs_x[i]
            
            # # Apply multi-scale fusion
            out_rgb = self.multi_scale_fusion_layers[i](out_rgb)
            out_x = self.multi_scale_fusion_layers[i](out_x)

            # # # 打印多尺度融合后的维度
            # # print(f"out_rgb after multi-scale fusion: {out_rgb.shape}")

            # edge_index_rgb = self.create_edge_index(out_rgb).to(DEVICE)
            # edge_index_x = self.create_edge_index(out_x).to(DEVICE)
            
            # # Flatten and permute
            # flattened_out_rgb = out_rgb.flatten(2).permute(0, 2, 1)
            # flattened_out_x = out_x.flatten(2).permute(0, 2, 1)

            # # print(f"flattened_out_rgb shape: {flattened_out_rgb.shape}")

            # reshaped_out_rgb = flattened_out_rgb.reshape(-1, flattened_out_rgb.size(2))
            # reshaped_out_x = flattened_out_x.reshape(-1, flattened_out_x.size(2))

            # # print(f"reshaped_out_rgb shape: {reshaped_out_rgb.shape}")

            # out_rgb = self.gcn_blocks_rgb[i](reshaped_out_rgb, edge_index_rgb)
            # out_x = self.gcn_blocks_x[i](reshaped_out_x, edge_index_x)

            # # 打印GCN输出维度
            # # print(f"out_rgb after GCN: {out_rgb.shape}")

            # # 更新形状
            # size = out_rgb.size(0)
            # dim = int((size / 4) ** 0.5)
            # out_rgb = out_rgb.view(4, out_rgb.size(1), dim, dim)
            # out_x = out_x.view(4, out_x.size(1), dim, dim)

            # # print(f"out_rgb after view: {out_rgb.shape}")

            # cross attention
            cma = True
            cam = True
            if cma and cam:
                # print("out_rgb.shape:",out_rgb.shape)  # 输出 out_rgb 的形状
                cross_rgb, cross_x = self.cross_mamba[i](out_rgb.permute(0, 2, 3, 1).contiguous(), out_x.permute(0, 2, 3, 1).contiguous()) # B x H x W x C
                x_fuse = self.channel_attn_mamba[i](cross_rgb, cross_x).permute(0, 3, 1, 2).contiguous()
            elif cam and not cma:
                x_fuse = self.channel_attn_mamba[i](out_rgb.permute(0, 2, 3, 1).contiguous(), out_x.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
            elif not cam and not cma:
                x_fuse = (out_rgb + out_x)
            outs_fused.append(x_fuse)
            # print("运行一次结束")
        return outs_fused, outs_rgb


    def create_edge_index(self, x):
        # Create a simple grid structure for the edges, based on the input feature map size
        B, C, H, W = x.shape
        row, col = torch.meshgrid(torch.arange(H), torch.arange(W))
        row, col = row.flatten(), col.flatten()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index

    def forward(self, x_rgb, x_e):
        out, outs_rgb = self.forward_features(x_rgb, x_e)
        return out, outs_rgb

class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.scale1_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.scale2_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.scale3_conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 根据需要调整scale_factor
        self.downsample = nn.AdaptiveAvgPool2d((out_channels, out_channels))
        
        self.adjust_channels = nn.Conv2d(out_channels * 3, out_channels, 1)  # 将合并的通道数调整回原始尺寸

    def forward(self, x):
        x1 = self.scale1_conv(x)
        x2 = self.scale2_conv(x)
        x3 = self.scale3_conv(x)

        # Assume x1 is the target size
        x2 = self.upsample(x2) if x2.size()[2:] != x1.size()[2:] else x2
        x3 = self.downsample(x3) if x3.size()[2:] != x1.size()[2:] else x3

        x_fused = torch.cat([x1, x2, x3], dim=1)
        x_fused = self.adjust_channels(x_fused)
        return x_fused

# 下面是模型实例
class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='/home/czh/Soil_Fusion_Mamba/pretrained/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='pretrained/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='pretrained/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )
