import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torchsummary import summary
from torchstat import stat
import os
import numpy as np
import random
from scipy.stats import truncnorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lf2disp.BpCNet.models.warp import FourierWarpSingleImage


os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

class ResidualBlock3D(nn.Module):
    expansion = 1

    def __init__(self, inchannel, outchannel, stride=1, kernel_size=2):
        super(ResidualBlock3D, self).__init__()
        self.left = nn.Sequential(
            nn.Conv3d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv3d(outchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=0, bias=False),
            nn.BatchNorm3d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inchannel, outchannel, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(outchannel)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class BasicBlock(nn.Module):

    def __init__(self, input_dim, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_channels, kernel_size=3, stride=stride, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or out_channels != 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_dim, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.downsample(x)
        out = x1 + x2
        return out


class feature_extraction(nn.Module):
    def __init__(self, input_dim, output_dim=16, device=None):
        super(feature_extraction, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )
        self.layers = list()
        input_dim = 4
        numblock = [2, 8, 2, 2]
        hidden_dim = [4, 8, 16, 16]
        for i in range(0, 4):
            temp = self._make_layer(input_dim, hidden_dim[i], numblock[i], 1)
            self.layers.append(temp)
            input_dim = hidden_dim[i]
        self.layers = nn.Sequential(*self.layers)
        # SPP Module
        self.branchs = list()
        hidden_dim = [4, 4, 4, 4]
        size = [2, 4, 8, 16]
        for i in range(0, 4):
            temp = nn.Sequential(
                nn.AvgPool2d((size[i], size[i]), (size[i], size[i])),
                nn.Conv2d(input_dim, hidden_dim[i], kernel_size=1, stride=1, dilation=1),
                nn.BatchNorm2d(hidden_dim[i]),
                nn.ReLU(inplace=True),
            )
            self.branchs.append(temp)
        self.branchs = nn.Sequential(*self.branchs)

        input_dim = np.array(hidden_dim).sum() + 8 + 16
        self.last = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1,dilation=1),
        )

    def _make_layer(self, input_dim, out_channels, blocks, stride):
        layers = list()
        layers.append(BasicBlock(input_dim, out_channels, stride))
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x):
        x = self.conv1(x)
        layers_out = [x]
        for i in range(len(self.layers)):
            layers_out.append(self.layers[i](layers_out[-1]))
        layer4_size = layers_out[-1].shape  # B,C,H,W
        branchs_out = []
        for i in range(len(self.branchs)):
            temp = self.branchs[i](layers_out[-1])
            temp = nn.UpsamplingBilinear2d(size=(int(layer4_size[-2]), int(layer4_size[-1])))(temp)
            branchs_out.append(temp)
        cat_f = [layers_out[2], layers_out[4]] + branchs_out
        feature = torch.cat([i for i in cat_f], dim=1)
        out = self.last(feature)
        return out


class ViewAttention(nn.Module):
    def __init__(self, input_dim, n_views=9, device=None):
        super(ViewAttention, self).__init__()
        self.n_views = n_views
        self.global_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_dim * self.n_views * self.n_views, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.n_views * self.n_views, kernel_size=1),
            nn.BatchNorm2d(self.n_views * self.n_views),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W, MM = x.shape
        x = x.permute(0, 1, 4, 2, 3)
        x = x.reshape(B, -1, H, W)
        x_g = self.global_attention(x)  # B,1,MM,H,W
        wei = self.sigmoid(x_g)
        wei = wei.reshape(B, 1, 1, 1, MM)
        return wei


class BuildVolume(nn.Module):
    def __init__(self, input_dim, output_dim, warp_method=False, n_views=9, device=None):
        super(BuildVolume, self).__init__()
        self.device = device
        self.n_views = n_views
        self.fourierwarp = FourierWarpSingleImage(device=self.device).to(self.device)
        self.warp_method = warp_method

        # this is for fourierwarp
        if self.warp_method:
            self.last_true = nn.Sequential(
                    nn.Conv2d(2*input_dim * n_views ** 2, output_dim, kernel_size=1),
                )
        # this is for bilinearwarp
        else:
            self.last = nn.Sequential(
                    nn.Conv2d(input_dim * n_views ** 2, output_dim, kernel_size=1),
                )

    def forward(self, deltmap, imageMxM, x_g):
        if self.warp_method:
            return self.use_fourier(deltmap,imageMxM,x_g)
        else:
            return self.use_bilinear(deltmap,imageMxM,x_g)

    def use_bilinear(self, deltmap, imageMxM, x_g):
        B, H, W, N = deltmap.shape
        coords_x = torch.linspace(-1, 1, W)
        coords_y = torch.linspace(-1, 1, H)
        coords_x = coords_x.repeat(H, 1).reshape(H, W, 1)
        coords_y = coords_y.repeat(W, 1).permute(1, 0).reshape(H, W, 1)
        coords = torch.cat([coords_x, coords_y], dim=2)
        coords = coords.reshape(1, H, W, 2).to(self.device)
        B, H, W, C, M, M = imageMxM.shape
        center_view = int(M / 2)
        pixel_feats = list()
        for h in range(0, N):
            cost_list = list()
            for v in range(0, M):
                for u in range(0, M):
                    offsetx = 2 * (center_view - u) * deltmap[:, :, :, h] / W
                    offsety = 2 * (center_view - v) * deltmap[:, :, :, h] / H
                    coords_x = coords[:, :, :, 0] + offsetx
                    coords_y = coords[:, :, :, 1]  + offsety
                    coords_uv = torch.cat([coords_x.unsqueeze(dim=-1), coords_y.unsqueeze(dim=-1)], dim=-1)
                    temp = F.grid_sample(imageMxM[:, :, :, :, v, u].permute(0, 3, 1, 2), coords_uv[:, :, :, :])
                    cost_list.append(temp.unsqueeze(dim=-1))  # B,C,H,W,MM
            cost = torch.cat([i for i in cost_list], dim=-1)  # B,C,H,W,MM
            cost = (cost * x_g).permute(0, 1, 4, 2, 3)
            cost = cost.reshape(B, -1, H, W)
            cost = self.last(cost)     
            pixel_feats.append(cost.unsqueeze(dim=-1))
        pixel_feats = torch.cat([i for i in pixel_feats], dim=-1)  # B,C,H,W,N
        return pixel_feats

    def use_fourier(self, deltmap, imageMxM, x_g):
        B, H, W, C, M, M = imageMxM.shape
        B, H, W, N = deltmap.shape
        center_view = int(M / 2)
        pixel_feats = list()
        for h in range(0, N):
            feat = list()
            for v in range(0, M):
                for u in range(0, M):
                    offsety = (center_view - v) * deltmap[:, :, :, h]
                    offsetx = (center_view - u) * deltmap[:, :, :, h]
                    temp = self.fourierwarp(imageMxM[:, :, :, :, v, u], offsetx=offsetx, offsety=offsety).permute(0,3,1,2)
                    feat.append(temp.unsqueeze(dim=-1))
            cost = torch.cat([i for i in feat], dim=-1)
            cost = (cost * x_g).permute(0, 1, 4, 2, 3)
            cost = cost.reshape(B, -1, H, W)
            cost = self.last_true(cost)
            pixel_feats.append(cost.unsqueeze(dim=-1))
        pixel_feats = torch.cat([i for i in pixel_feats], dim=-1)  # B,C,H,W,N 
        return pixel_feats


class RefineNet(nn.Module):
    def __init__(self, input_dim, n_views, hidden_dim=140,warp_method=False, scope=1.0, times=8,
                 device=None):
        super(RefineNet, self).__init__()

        self.device = device
        self.scope = scope
        scope = self.scope
        step = 2 * scope / times
        self.delta = torch.range(-scope, scope, step).to(self.device)
        self.n_views = n_views
        self.warp_method = warp_method

        # feature_extraction
        self.feature_extraction = feature_extraction(input_dim=input_dim, output_dim=12, device=self.device)

        # view attention
        self.view_attention = ViewAttention(input_dim=12, n_views=9, device=device).to(self.device)

        self.build_volume = BuildVolume(input_dim=12, output_dim=hidden_dim, warp_method=self.warp_method, n_views=self.n_views,
                                        device=self.device)

        self.cost_conv1 = ResidualBlock3D(hidden_dim,hidden_dim)
        self.cost_conv2 = ResidualBlock3D(hidden_dim,hidden_dim)
        self.cost_conv3 = ResidualBlock3D(hidden_dim,hidden_dim)
        self.cls = nn.Sequential(
            nn.Conv3d(hidden_dim, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, depthmap, imageMxM, scale=1.0):
        B, H, W, C = depthmap.shape
        
        delta = self.delta.reshape(1, 1, -1).repeat(H, W, 1)
        deltamap = depthmap + delta*scale   # B*H*W*N

        B, H, W, C, M, M = imageMxM.shape
        imageMxM = imageMxM.permute(0, 4, 5, 3, 1, 2).reshape(B * M * M, C, H, W)
        x = self.feature_extraction(imageMxM)
        _, C, H, W = x.shape
        x = x.reshape(B, M * M, C, H, W).permute(0, 2, 3, 4, 1)  # B,C,H,W,MM
        # view_attention
        x_g = self.view_attention(x)
        # 代价体
        x = x.reshape(B, C, H, W, M, M).permute(0, 2, 3, 1, 4, 5)
        x = self.build_volume(deltamap, x, x_g)
        _, C, H, W, N = x.shape
        x = self.cost_conv1(x)
        x = self.cost_conv2(x)
        x = self.cost_conv3(x)
        x = self.cls(x)
        score = nn.functional.softmax(x, dim=-1).reshape(B, H, W, -1)
        refine_map = torch.sum(score * deltamap, dim=-1, keepdim=True)
        return refine_map


if __name__ == "__main__":
    depthmap = torch.ones(2, 32, 32, 1)
    imageMxM = torch.ones(2, 32, 32, 1, 9, 9)
    encoder = RefineNet(input_dim=1, n_views=9, hidden_dim=140, img_size=(32, 32), device='cpu')
    print(encoder)

    total_num = sum(p.numel() for p in encoder.parameters())
    trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)
    out = encoder(depthmap, imageMxM)
    print(out.shape)
    pass
