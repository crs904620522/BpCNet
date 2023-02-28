import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import trimesh
import time
import os
import cv2
from lf2disp.utils.utils import depth_metric, write_pfm


class GeneratorDepth(object):

    def __init__(self, model, cfg=None, device=None):
        self.model = model.to(device)
        self.device = device
        self.scope = cfg['RefineNet']['scope']
        self.iteration = cfg['data']['iteration']
        self.generate_dir = cfg['generation']['generation_dir']
        self.name =  cfg['generation']['name']
        if not os.path.exists(self.generate_dir):
            os.makedirs(self.generate_dir)

    def generate_depth(self, data, id=0):
        ''' Generates the output depthmap
        '''
        self.model.eval()
        device = self.device
        with torch.no_grad():
            label = data.get('label').to(device)
            B1, B2, H, W = label.shape
            imageMxM = data.get('imageMxM').to(device)
            label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
            with torch.no_grad():
                refine_map = data.get('coarse').to(device)
                coarse_map = refine_map.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
                print("coarse",depth_metric(label[15:-15, 15:-15], coarse_map[15:-15, 15:-15]))
                scale = 1.0
                for i in range(self.iteration):
                    refine_map = self.model.refine(refine_map, imageMxM,scale)
                    scale /= 2
                    depthmap = refine_map.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
                    print("refine:"+str(i),depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15]))
                    torch.cuda.empty_cache()
                save_path = os.path.join(self.generate_dir, self.name[id] + '_fine.png')
                metric = depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15])
                pfm_path = os.path.join(self.generate_dir, self.name[id] + '_fine.pfm')
                write_pfm(depthmap, pfm_path, scale=1.0)
                depthmap = (depthmap - depthmap.min()) / (depthmap.max() - depthmap.min())
                cv2.imwrite(save_path, depthmap.copy() * 255.0)
                print('save fine depthmap in', save_path)
            return metric
