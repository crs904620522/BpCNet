import torch
import torch.nn as nn

import os
from lf2disp.BpCNet.models.RefineNet import RefineNet
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class BpCNet(nn.Module):
    def __init__(self, cfg, device=None):
        super().__init__()
        n_views = cfg['data']['views']
        input_dim = cfg['RefineNet']['input_dim']
        scope = cfg['RefineNet']['scope']
        times = cfg['RefineNet']['times']
        warp_method = cfg['data']['warp_method']
        self.refine_net = RefineNet(input_dim=input_dim, n_views=n_views, scope=scope,
                                        times=times,warp_method=warp_method, device=device).to(device)

    def forward(self, coarse_map, imageMxM, scale=1.0):
        B1, B2, H, W, M, M = imageMxM.shape
        imageMxM = imageMxM.reshape(B1 * B2, H, W, M, M,1).permute(0, 1, 2, 5, 3, 4)
        coarse_map = coarse_map.reshape(B1*B2,H,W,-1)
        refine_map = self.refine_net(coarse_map, imageMxM,scale=scale)
        refine_map = refine_map.reshape(B1*B2, 1, H, W).permute(0, 2, 3, 1)
        return refine_map

    def refine(self, coarse_map, imageMxM, scale=1.0):
        B1, B2, H, W, M, M = imageMxM.shape
        imageMxM = imageMxM.reshape(B1 * B2, H, W, M, M,1).permute(0, 1, 2, 5, 3, 4)
        coarse_map = coarse_map.reshape(B1*B2,H,W,-1)
        refine_map = self.refine_net(coarse_map, imageMxM,scale=scale)
        refine_map = refine_map.reshape(B1*B2, 1, H, W).permute(0, 2, 3, 1)
        return refine_map

    def to(self, device):
        ''' Puts the model to the device.
        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model


