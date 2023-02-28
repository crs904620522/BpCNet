# coding:utf-8
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import os
import cv2
import csv
from PIL import Image
import random
from scipy.stats import truncnorm
from lf2disp.utils import utils
import imageio
from skimage import io
import time

np.random.seed(160)


class HCInew(Dataset):
    def __init__(self, cfg, mode='train'):
        super(HCInew, self).__init__()
        self.datadir = cfg['data']['path']
        self.mode = mode
        self.coarse_dir = cfg['data']['coarse_dir']  # coarse data path
        self.coarse_list = list()
        if mode == 'train':
            self.imglist = []
            self.batch_size = cfg['training']['image_batch_size']
            with open(os.path.join(self.datadir, 'train.txt'), "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
                    self.coarse_list.append(os.path.join(self.coarse_dir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['training']['input_size']
            self.imageMxM_size = cfg['training']['imageMxM_size']
            self.augmentation = cfg['training']['augmentation']
        elif mode == 'test':  # val or test
            self.imglist = []
            self.batch_size = cfg['test']['image_batch_size']
            datafile = os.path.join(self.datadir, 'test.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
                    self.coarse_list.append(os.path.join(self.coarse_dir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['test']['input_size']
        elif mode == 'vis':
            self.imglist = []
            self.batch_size = cfg['vis']['image_batch_size']
            datafile = os.path.join(self.datadir, 'vis.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
                    self.coarse_list.append(os.path.join(self.coarse_dir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['vis']['input_size']
        elif mode == 'generate':
            self.imglist = []
            self.batch_size = 1
            datafile = os.path.join(self.datadir, 'generate.txt')
            with open(datafile, "r") as f:
                for line in f.readlines():
                    imgdir = line.strip("\n")
                    self.imglist.append(os.path.join(self.datadir, imgdir))
                    self.coarse_list.append(os.path.join(self.coarse_dir, imgdir))
            self.number = len(self.imglist)
            self.views = cfg['data']['views']
            self.inputsize = cfg['generation']['input_size']

        self.invalidpath = []
        with open(os.path.join(self.datadir, 'invalid.txt'), "r") as f:
            for line in f.readlines():
                imgpath = line.strip("\n")
                self.invalidpath.append(os.path.join(self.datadir, imgpath))

        self.traindata_all = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
        self.traindata_label = np.zeros((len(self.imglist), 512, 512), np.float32)
        self.boolmask_data = np.zeros((len(self.invalidpath), 512, 512), np.float32)
        self.imgPreloading()

    def __len__(self):
        return self.number

    def __getitem__(self, idx):
        if self.mode == 'train':
            label, coarse, imageMxM = self.train_data()
            label, coarse, imageMxM = self.data_aug(label, coarse, imageMxM)
        else:
            label, coarse, imageMxM = self.val_data(idx)
        out = {'label': np.float32(label),
               'coarse': np.float32(coarse),
               'imageMxM': np.float32(np.clip(imageMxM,0.0,1.0)),
               }
        return out

    def imgPreloading(self):
        for idx in range(0, len(self.imglist)):
            imgdir = self.imglist[idx]
            # Load image
            for i in range(0, self.views ** 2):
                imgname = 'input_Cam' + str(i).zfill(3) + '.png'
                imgpath = os.path.join(imgdir, imgname)
                img = np.uint8(imageio.imread(imgpath))
                self.traindata_all[idx, :, :, i // 9, i - 9 * (i // 9), :] = img
            labelname = 'gt_disp_lowres.pfm'
            labelpath = os.path.join(imgdir, labelname)
            if os.path.exists(labelpath):
                imgLabel = utils.read_pfm(labelpath)
            else:
                imgLabel = np.zeros((512, 512))

            self.traindata_label[idx] = imgLabel

        # load mask
        for idx in range(0, len(self.invalidpath)):
            boolmask_img = np.uint8(imageio.imread(self.invalidpath[idx]))
            boolmask_img = 1.0 * boolmask_img[:, :, 3] > 0
            self.boolmask_data[idx] = boolmask_img
        # load coarse data
        self.coarse_path_list = []
        for idx in range(0, len(self.imglist)):
            coarse_label_dir = self.coarse_list[idx]
            temp = []
            for root,dirs,files in os.walk(coarse_label_dir):
                for file in files:
                    if file.endswith('.pfm'):
                        labelpath = os.path.join(coarse_label_dir, file)
                        temp.append(labelpath)
            self.coarse_path_list.append(temp)
        # shift
        if self.mode == 'train':
            self.traindata_all_add1 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub1 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_add2 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            self.traindata_all_sub2 = np.zeros((len(self.imglist), 512, 512, 9, 9, 3), np.uint8)
            center = int(self.views / 2)
            for batch_i in range(0, len(self.imglist)):
                for v in range(0, self.views):  # v
                    for u in range(0, self.views):  # u
                        offsety = (center - v)
                        offsetx = (center - u)
                        mat_translation = np.float32([[1, 0, 1 * offsetx], [0, 1, 1 * offsety]])
                        self.traindata_all_add1[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -1 * offsetx], [0, 1, -1 * offsety]])
                        self.traindata_all_sub1[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, 2 * offsetx], [0, 1, 2 * offsety]])
                        self.traindata_all_add2[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
                        mat_translation = np.float32([[1, 0, -2 * offsetx], [0, 1, -2 * offsety]])
                        self.traindata_all_sub2[batch_i, :, :, v, u, :] = cv2.warpAffine(
                            self.traindata_all[batch_i, :, :, v, u, :],
                            mat_translation,
                            (512, 512))
        print('Image Preloading Finish!')
        return

    def train_data(self):
        """ initialize image_stack & label """
        batch_size = self.batch_size
        imageMxM_size, label_size, input_size = self.imageMxM_size, self.inputsize, self.inputsize
        traindata_batch_MxM = np.zeros((batch_size, imageMxM_size, imageMxM_size, self.views, self.views),
                                       dtype=np.uint8)
        traindata_batch_label = np.zeros((batch_size, label_size, label_size))
        traindata_batch_coarse = np.zeros((batch_size, label_size, label_size))

        """ inital variable """
        crop_half1 = int(0.5 * (imageMxM_size - label_size))
        crop_half2 = int(0.5 * (imageMxM_size - input_size))

        """ Generate image stacks """
        for ii in range(0, batch_size):
            sum_diff = 0
            valid = 0

            while sum_diff < 0.01 * input_size * input_size or valid < 1:
                rand_3color = 0.05 + np.random.rand(3)
                rand_3color = rand_3color / np.sum(rand_3color)
                R = rand_3color[0]
                G = rand_3color[1]
                B = rand_3color[2]

                aa_arr = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14,
                                   0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ])

                image_id = np.random.choice(aa_arr)
                traindata_all, traindata_label,traindata_coarse = self.choose_delta(image_id)
                if self.views == 9:
                    ix_rd = 0
                    iy_rd = 0

                kk = np.random.randint(17)
                if (kk < 8):
                    scale = 1
                elif (kk < 14):
                    scale = 2
                elif (kk < 17):
                    scale = 3
                idx_start = np.random.randint(0, 512 - scale * imageMxM_size)  # random_size
                idy_start = np.random.randint(0, 512 - scale * imageMxM_size)
                valid = 1

                """
                    boolmask: reflection masks for images(4,6,15)
                """

                # 这是去除高光
                if image_id in [4, 6, 15]:
                    if image_id == 4:
                        a_tmp = self.boolmask_data[0]
                    if image_id == 6:
                        a_tmp = self.boolmask_data[1]
                    if image_id == 15:
                        a_tmp = self.boolmask_data[2]

                    if (np.sum(a_tmp[
                               idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                               idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]) > 0
                            or np.sum(a_tmp[
                                      idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                                      idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale]) > 0):
                        valid = 0
                if valid > 0:
                    image_center = (1 / 255) * np.squeeze(
                        R * traindata_all[
                            idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                            idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale,
                            4 + ix_rd, 4 + iy_rd, 0].astype(
                            'float32') +
                        G * traindata_all[
                            idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                            idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale,
                            4 + ix_rd, 4 + iy_rd, 1].astype(
                            'float32') +
                        B * traindata_all[
                            idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                            idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale,
                            4 + ix_rd, 4 + iy_rd, 2].astype('float32'))
                    sum_diff = np.sum(
                        np.abs(image_center - np.squeeze(image_center[int(0.5 * input_size), int(0.5 * input_size)])))
                    
                    traindata_batch_MxM[ii, :, :, :, :] = np.squeeze(
                        R * traindata_all[idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                                idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale, :, :, 0].astype(
                            'float32') +
                        G * traindata_all[idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                                idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale, :, :, 1].astype(
                            'float32') +
                        B * traindata_all[idx_start + scale * crop_half2: idx_start + scale * crop_half2 + scale * input_size:scale,
                                idy_start + scale * crop_half2: idy_start + scale * crop_half2 + scale * input_size:scale, :, :, 2].astype(
                            'float32'))

                    if len(traindata_label.shape) == 5:
                        traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[
                                                                          idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                          idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale,
                                                                          4 + ix_rd, 4 + iy_rd]
                        traindata_batch_coarse[ii, :, :] = (1.0 / scale) * traindata_coarse[
                                                                           idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                           idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale,
                                                                           4 + ix_rd, 4 + iy_rd]                                                
                    else:
                        traindata_batch_label[ii, :, :] = (1.0 / scale) * traindata_label[
                                                                          idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                          idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]
                        traindata_batch_coarse[ii, :, :] = (1.0 / scale) * traindata_coarse[
                                                                          idx_start + scale * crop_half1: idx_start + scale * crop_half1 + scale * label_size:scale,
                                                                          idy_start + scale * crop_half1: idy_start + scale * crop_half1 + scale * label_size:scale]                                    

        traindata_batch_MxM = np.float32((1 / 255) * traindata_batch_MxM)

        return traindata_batch_label, traindata_batch_coarse, traindata_batch_MxM

    def choose_delta(self, image_id):
        traindata_coarse = utils.read_pfm(self.coarse_path_list[image_id][0])
        trans = np.random.randint(25)
        if trans < 15:  # 0
            traindata_all = self.traindata_all[image_id]
            traindata_label = self.traindata_label[image_id]
            traindata_coarse = traindata_coarse
        elif trans < 18:
            traindata_all = self.traindata_all_add1[image_id]
            traindata_label = self.traindata_label[image_id] + 1
            traindata_coarse +=1
        elif trans < 21:
            traindata_all = self.traindata_all_sub1[image_id]
            traindata_label = self.traindata_label[image_id] - 1
            traindata_coarse -=1
        elif trans < 23:
            traindata_all = self.traindata_all_add2[image_id]
            traindata_label = self.traindata_label[image_id] + 2
            traindata_coarse +=2
        elif trans < 25:
            traindata_all = self.traindata_all_sub2[image_id]
            traindata_label = self.traindata_label[image_id] - 2
            traindata_coarse -=2

        return traindata_all, traindata_label,traindata_coarse

    def data_aug(self,  traindata_label_batchNxN, traindata_batch_coarse, traindata_batch_MxM):

        traindata_batch_MxM = traindata_batch_MxM

        for batch_i in range(self.batch_size):
            gray_rand = 0.4 * np.random.rand() + 0.8

            traindata_batch_MxM[batch_i] = pow(traindata_batch_MxM[batch_i],gray_rand)

            transp_rand = np.random.randint(0, 2)

            if transp_rand == 1:
                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.transpose(traindata_label_batchNxN[batch_i, :, :], (1, 0)))
                traindata_batch_coarse[batch_i, :, :] = np.copy(
                    np.transpose(traindata_batch_coarse[batch_i, :, :], (1, 0)))
                # H,W转置
                traindata_batch_MxM_tmp6 = np.copy(
                    np.transpose(np.squeeze(traindata_batch_MxM[batch_i, :, :, :, :]), (1, 0, 2, 3)))
                # MxM转置
                traindata_batch_MxM[batch_i, :, :, :, :] = np.copy(
                    np.transpose(np.squeeze(traindata_batch_MxM_tmp6), (0, 1, 3, 2)))

            rotation_or_transp_rand = np.random.randint(0, 4)
            if rotation_or_transp_rand == 1:  # 90도

                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN[batch_i], 1, (0, 1)))
                traindata_batch_coarse[batch_i, :, :] = np.copy(
                    np.rot90(traindata_batch_coarse[batch_i, :, :], 1, (0, 1)))
                # 每张图像旋转90
                traindata_batch_MxM_tmp6 = np.copy(
                    np.rot90(traindata_batch_MxM[batch_i], 1, (0, 1)))
                # MxM旋转90
                traindata_batch_MxM[batch_i] = np.copy(np.rot90(traindata_batch_MxM_tmp6, 1, (2, 3)))

            if rotation_or_transp_rand == 2:  # 180도

                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN[batch_i, :, :], 2, (0, 1)))
                traindata_batch_coarse[batch_i, :, :] = np.copy(
                    np.rot90(traindata_batch_coarse[batch_i, :, :], 2, (0, 1)))
                # 每张图像旋转90
                traindata_batch_MxM_tmp6 = np.copy(
                    np.rot90(traindata_batch_MxM[batch_i], 2, (0, 1)))
                # MxM旋转90
                traindata_batch_MxM[batch_i] = np.copy(np.rot90(traindata_batch_MxM_tmp6, 2, (2, 3)))

            if rotation_or_transp_rand == 3:  # 270도
                traindata_label_batchNxN[batch_i, :, :] = np.copy(
                    np.rot90(traindata_label_batchNxN[batch_i, :, :], 3, (0, 1)))
                traindata_batch_coarse[batch_i, :, :] = np.copy(
                    np.rot90(traindata_batch_coarse[batch_i, :, :], 3, (0, 1)))

                # 每张图像旋转90
                traindata_batch_MxM_tmp6 = np.copy(
                    np.rot90(traindata_batch_MxM[batch_i], 3, (0, 1)))
                # MxM旋转90
                traindata_batch_MxM[batch_i] = np.copy(np.rot90(traindata_batch_MxM_tmp6, 3, (2, 3)))


        # image noise
        for batch_i in range(self.batch_size):
            traindata_batch_MxM[batch_i] = self.add_noise(traindata_batch_MxM[batch_i])

        # coarse noise
        times = np.random.randint(3)
        for i in range(0,times):
            random_scale_noise = truncnorm(a=0.80, b=1.20, scale=1.).rvs(size=[traindata_batch_coarse[batch_i, :, :].shape[0], traindata_batch_coarse[batch_i, :, :].shape[1]])    
            traindata_batch_coarse[batch_i, :, :] =  traindata_label_batchNxN[batch_i, :, :] +random_scale_noise * (traindata_batch_coarse[batch_i, :, :] - traindata_label_batchNxN[batch_i, :, :])
            random_translation_noise = 0.2*(np.random.rand(traindata_batch_coarse[batch_i, :, :].shape[0], traindata_batch_coarse[batch_i, :, :].shape[1])-0.5)
            traindata_batch_coarse[batch_i, :, :] += random_translation_noise

        scale_noise = 2*(traindata_batch_coarse[batch_i, :, :] - traindata_label_batchNxN[batch_i, :, :]) * 2 * (random.random() - 0.5)
        traindata_batch_coarse[batch_i, :, :] = traindata_label_batchNxN[batch_i, :, :] + scale_noise
        translation_noise = 0.4 * (random.random() - 0.5)
        traindata_batch_coarse[batch_i, :, :] += translation_noise  

        return traindata_label_batchNxN,traindata_batch_coarse, traindata_batch_MxM

    def add_noise(self, traindata_MxM):
        noise_rand = np.random.randint(0, 40)
        if noise_rand in [0, 1, 2, 3]: 
            gauss = np.random.normal(0.0, np.random.uniform() * np.sqrt(0.2), (
                traindata_MxM.shape[0], traindata_MxM.shape[1], traindata_MxM.shape[2],
                traindata_MxM.shape[3]))
            traindata_MxM = np.clip(traindata_MxM + gauss, 0.0, 1.0)
        return traindata_MxM

    def val_data(self, idx):
        batch_size = 1
        label_size, input_size = self.inputsize, self.inputsize

        test_data_MxM = np.zeros((batch_size, input_size, input_size, self.views, self.views),
                                 dtype=np.uint8)

        test_data_label = np.zeros((batch_size, label_size, label_size))
        test_data_coarse = np.zeros((batch_size, label_size, label_size))
        crop_half1 = int(0.5 * (input_size - label_size))

        R = 0.299
        G = 0.587
        B = 0.114
        test_image = self.traindata_all[idx]
        test_label = self.traindata_label[idx]
        test_data_coarse[0] =  utils.read_pfm(self.coarse_path_list[idx][0])



        test_data_MxM[0] = np.squeeze(
                        R * test_image[:,:,:,:,0].astype('float32') +
                        G * test_image[:,:,:,:,1].astype('float32') +
                        B * test_image[:,:,:,:,2].astype('float32'))

        test_data_label[0] = test_label[crop_half1: crop_half1 + label_size,
                             crop_half1:crop_half1 + label_size]
        test_data_MxM = np.float32((1 / 255) * test_data_MxM)
        return test_data_label,test_data_coarse, test_data_MxM


import math


def testData():
    cfg = {'data': {'path': 'D:/code/LFdepth/LFData/HCInew',
                    'views': 9},
           'training': {'input_size': 25, 'transform': False,
                        'image_batch_size': 8,
                        'imageMxM_size': 45,
                        'augmentation': True,
                        'mode': 'coarse'},

           }
    mydataset = HCInew(cfg)
    train_loader = DataLoader(mydataset, batch_size=1, shuffle=True)
    for epoch in range(5):
        for i, data in enumerate(train_loader):
            out = data
            print(out['image'].shape, out['label'].shape, out['imageMxM'].shape, type(out['image']))


if __name__ == '__main__':
    testData()
