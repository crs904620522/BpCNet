import os
from tqdm import trange
import torch
from torch.nn import functional as F
from torch import distributions as dist
from lf2disp.training import BaseTrainer
import torch.nn as nn
import cv2
import math
import numpy as np
from lf2disp.utils.utils import depth_metric


class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion=nn.MSELoss, device=None, cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion()
        self.vis_dir = cfg['vis']['vis_dir']
        self.test_dir = cfg['test']['test_dir']
        self.iteration = cfg['data']['iteration']

        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def train_step(self, data, iter=0):
        self.model.train()
        
        self.optimizer.zero_grad()
        loss, scale, depth_map = self.compute_loss(data, iter)
        loss.backward()
        self.optimizer.step()
        for i in range(self.iteration - 1):
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            loss, scale, depth_map = self.compute_loss(data, iter,depth_map,scale/2)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def eval_step(self, data, imgid=0, val_dir=None):
        device = self.device
        val_dir = self.test_dir
        torch.cuda.empty_cache()
        self.model.eval()
        label = data.get('label').to(device)
        imageMxM = data.get('imageMxM').to(device)
        B1, B2, H, W = label.shape
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
        metric = depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15])
        metric['id'] = imgid
        if val_dir is not None:
            self.visualize(data, id=imgid, vis_dir=val_dir)
        return metric

    def visualize(self, data, id=0, vis_dir=None):
        self.model.eval()
        torch.cuda.empty_cache()
        device = self.device
        if vis_dir == None:
            vis_dir = self.vis_dir
        self.model.eval()

        label = data.get('label').to(device)
        imageMxM = data.get('imageMxM').to(device)
        B1, B2, H, W = label.shape
        with torch.no_grad():
            refine_map = data.get('coarse').to(device)
            coarse_map = refine_map.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
            scale = 1.0
            for i in range(self.iteration):
                refine_map = self.model.refine(refine_map, imageMxM,scale)
                scale /= 2
            depthmap = refine_map.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]

        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
        depthmap = (depthmap - label.min()) / (label.max() - label.min())
        coarse_map = (coarse_map - label.min()) / (label.max() - label.min())
        label = (label - label.min()) / (label.max() - label.min())
        path = os.path.join(vis_dir, str(id) + '.png')
        labelpath = os.path.join(vis_dir, '%03d_label.png' % id)
        coarse_path = os.path.join(vis_dir, str(id) + '_coarse.png')

        cv2.imwrite(path, depthmap.copy() * 255.0)
        print('save depth map in', path)
        cv2.imwrite(labelpath, label.copy() * 255.0)
        print('save label in', labelpath)
        cv2.imwrite(coarse_path, coarse_map.copy() * 255.0)
        print('save coarse in', coarse_path)


    def compute_loss(self, data, iter=0,coarse_map=None,scale=1.0):
        device = self.device
        imageMxM = data.get('imageMxM').to(device)
        label = data.get('label').to(device)
        B1, B2, H, W = label.shape

        if coarse_map == None:
            coarse_map = data.get('coarse').to(device)
            scale = 1.0
        else:
            coarse_map = coarse_map
            scale = scale
        fine_map = self.model.refine(coarse_map, imageMxM,scale)
        next_fine = fine_map.reshape(B1*B2,H,W,1)

        coarse_map = coarse_map.reshape(B1*B2,-1)
        fine_map = fine_map.reshape(B1*B2,-1)
        label = label.reshape(B1*B2,-1)

        if iter % 100 == 0:
            print("------------", self.criterion(coarse_map, fine_map).sum())
            print("------fine------", self.criterion(fine_map, label).sum())
            print("------coarse------", self.criterion(coarse_map, label).sum())
        loss_fine = self.criterion(fine_map, label).mean()
        loss = loss_fine.mean()
            
        return loss, scale, next_fine.detach()