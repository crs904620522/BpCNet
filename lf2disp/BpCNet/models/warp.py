# -*- coding: utf-8 -*-
"""
@Time: 2021/10/14 16:47
@Auth: Rongshan Chen
"""
import torch
from torch import nn, einsum
import torch.nn.functional as F

import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import math

class FourierWarpSingleImage(nn.Module):
    '''
    input: B, C1, H, W
    output: B, C2, H, W
    '''
    def __init__(self, strict=False, device=None):
        super(FourierWarpSingleImage, self).__init__()
        self.device = device
        self.strict = strict # False for network train, True for image visulization


    def forward(self, image, offsetx, offsety):
        B, H, W = offsetx.shape
        Nh = torch.fft.ifftshift(torch.tensor([-1, 0, 1]))
        Nw = torch.fft.ifftshift(torch.tensor([-1, 0, 1]))
        Nw, Nh = torch.meshgrid(Nh, Nw)
        img_x = np.array([i for i in range(0, H)])
        img_y = np.array([j for j in range(0, W)])
        img_x, img_y = np.meshgrid(img_x, img_y)
        a = 1
        all_grid = [[img_y - a, img_x - a], [img_y, img_x - a], [img_y + a, img_x - a],
                    [img_y - a, img_x], [img_y, img_x], [img_y + a, img_x],
                    [img_y - a, img_x + a], [img_y, img_x + a], [img_y + a, img_x + a], ]
        all_grid = torch.from_numpy(np.array(all_grid)).to(self.device)

        all_grid = all_grid.permute(0, 2, 3, 1)

        Nw = Nw.to(self.device)
        Nh = Nh.to(self.device)

        Batch_index = torch.from_numpy(np.array([i for i in range(B)])).to(self.device)
        Batch_index = Batch_index.reshape(B, 1, 1, 1, 1).repeat(repeats=(1, 9 * H * W, 1, 1, 1)).reshape(B, 9, H, W, 1)

        int_y = all_grid[:, :, :, 0].reshape(1, 9, H, W) + offsety.reshape(B, 1, H, W).to(
            torch.int64) 
        int_x = all_grid[:, :, :, 1].reshape(1, 9, H, W) + offsetx.reshape(B, 1, H, W).to(torch.int64)
        int_x = int_x.reshape(B, 9, H, W, 1)
        int_y = int_y.reshape(B, 9, H, W, 1)

        # sample_coord_int
        sample_coord_int = torch.cat([Batch_index, int_y, int_x], dim=-1).reshape(B, 3, 3, H, W, 3)

        padding_error = torch.zeros((B, 9, H, W, 1)).to(self.device)
        padding_error[int_x >= W] = 1
        padding_error[int_y >= H] = 1
        padding_error[int_x < 0] = 1
        padding_error[int_y < 0] = 1
        padding_error = padding_error.reshape(B, 3, 3, H, W)

        sample_coord_int[padding_error == 1] = 0
        sample_coord_int = sample_coord_int.to(torch.int64)

        sample_patch = image[sample_coord_int[:, :, :, :, :, 0], sample_coord_int[:, :, :, :, :, 1],
                       sample_coord_int[:, :, :, :, :, 2], :]
        sample_patch[padding_error == 1] = 0

        offsetx -= offsetx.to(torch.int64)
        offsety -= offsety.to(torch.int64)
        offsetx = offsetx.reshape(1, 1, -1)
        offsety = offsety.reshape(1, 1, -1)
        temp_Nh = Nh.reshape(3, 3, 1)
        temp_Nw = Nw.reshape(3, 3, 1)
        sample_patch = sample_patch.permute(1, 2, 5, 0, 3, 4)  # 3*3*3*B*512*512


        fft_image = torch.fft.fftn(sample_patch, dim=(0, 1, 2))
        kernel = torch.exp(
            1j * 2 * math.pi * (
                    offsety * temp_Nh / 3 + offsetx * temp_Nw / 3))

        kernel = kernel.reshape(3, 3, 1, B, H, W)

        image_back = torch.fft.ifftn(kernel * fft_image, dim=(0, 1, 2)) * np.exp(-1j * 2)
        pixel_feats = image_back[1, 1].permute(1, 2, 3, 0)

        if self.strict: # output [image]  for visulization
            pixel_feats = torch.abs(pixel_feats)
        else: # output [real, imag] for train 
            f_real = (pixel_feats.real).type_as(image).reshape(B,H,W,-1)
            f_imag = (pixel_feats.imag).type_as(image).reshape(B,H,W,-1)
            pixel_feats = torch.cat([f_real,f_imag],dim=-1)

        return pixel_feats
