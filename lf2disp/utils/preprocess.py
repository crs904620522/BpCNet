# -*- coding: utf-8 -*-
"""
@Time: 2021/9/11 13:14
@Auth: Rongshan Chen
@File: utils.py
@IDE:PyCharm
@Motto: Happy coding, Thick hair
@Email: 904620522@qq.com
"""
import os
import sys
import numpy as np
import numpy as np
import imageio
import json

import cv2


# Positional encoding

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** np.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = np.linspace(2. ** 0., 2. ** max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                        freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        out = list()
        for fn in self.embed_fns:
            temp = fn(inputs)
            out.append(temp)

        return np.concatenate([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    # if i == -1:
    #     return tf.identity, 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [np.sin, np.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


def call_Fourier_transform(input, config=None):
    if config == None:
        config = {"multires_views": 4,
                  "i_embed": 0
                  }
    embed, out_dim = get_embedder(config["multires_views"], config["i_embed"])
    out = embed(input)
    return out, out_dim


import utils


def mix_occlude(data1, label1, data2, label2):
    # 混合

    traindata_all = data1
    traindata_label = label1
    another_data = data2
    another_label = label2
    temp = (another_label > traindata_label)
    out_label = np.where(temp, another_label, traindata_label)
    out_data = np.where(temp.reshape(512, 512, 9, 9, 1).repeat(3, axis=-1), another_data, traindata_all)
    return out_data, out_label


def trans_or_rot(data, label):  # 旋转或翻折
    """ transpose """
    traindata_batch_MxM = data
    traindata_label_batchNxN = label

    transp_rand = np.random.randint(0, 2)

    if transp_rand == 1:  # 这个是转置

        traindata_label_batchNxN_tmp6 = np.copy(
            np.transpose(np.squeeze(traindata_label_batchNxN[:, :, :, :]), (1, 0, 2, 3)))
        traindata_label_batchNxN = np.copy(
            np.transpose(np.squeeze(traindata_label_batchNxN_tmp6), (0, 1, 3, 2)))
        # H,W转置
        traindata_batch_MxM_tmp6 = np.copy(
            np.transpose(np.squeeze(traindata_batch_MxM[:, :, :, :, :]), (1, 0, 2, 3, 4)))
        # MxM转置
        traindata_batch_MxM[:, :, :, :, :] = np.copy(
            np.transpose(np.squeeze(traindata_batch_MxM_tmp6), (0, 1, 3, 2, 4)))

    rotation_or_transp_rand = np.random.randint(0, 4)
    if rotation_or_transp_rand == 1:  # 90도

        traindata_label_batchNxN_tmp6 = np.copy(
            np.rot90(traindata_label_batchNxN, 1, (0, 1)))
        traindata_label_batchNxN = np.copy(np.rot90(traindata_label_batchNxN_tmp6, 1, (2, 3)))

        # 每张图像旋转90
        traindata_batch_MxM_tmp6 = np.copy(
            np.rot90(traindata_batch_MxM[:, :, :, :, :], 1, (0, 1)))
        # MxM旋转90
        traindata_batch_MxM[:, :, :, :, :] = np.copy(np.rot90(traindata_batch_MxM_tmp6, 1, (2, 3)))

    if rotation_or_transp_rand == 2:  # 180도

        traindata_label_batchNxN_tmp6 = np.copy(
            np.rot90(traindata_label_batchNxN, 2, (0, 1)))
        traindata_label_batchNxN = np.copy(np.rot90(traindata_label_batchNxN_tmp6, 2, (2, 3)))

        # 每张图像旋转90
        traindata_batch_MxM_tmp6 = np.copy(
            np.rot90(traindata_batch_MxM[:, :, :, :, :], 2, (0, 1)))
        # MxM旋转90
        traindata_batch_MxM[:, :, :, :, :] = np.copy(np.rot90(traindata_batch_MxM_tmp6, 2, (2, 3)))

    if rotation_or_transp_rand == 3:  # 270도

        traindata_label_batchNxN_tmp6 = np.copy(
            np.rot90(traindata_label_batchNxN, 3, (0, 1)))
        traindata_label_batchNxN = np.copy(np.rot90(traindata_label_batchNxN_tmp6, 3, (2, 3)))

        # 每张图像旋转90
        traindata_batch_MxM_tmp6 = np.copy(
            np.rot90(traindata_batch_MxM[:, :, :, :, :], 3, (0, 1)))
        # MxM旋转90
        traindata_batch_MxM[:, :, :, :, :] = np.copy(np.rot90(traindata_batch_MxM_tmp6, 3, (2, 3)))

    print(transp_rand, rotation_or_transp_rand)
    return traindata_batch_MxM, traindata_label_batchNxN


def delta_or_not(data, label):
    pass


def read_data(path):
    data_folder = path
    imageMxM = np.zeros((512, 512, 9, 9, 3),
                        dtype=np.uint8)
    deltmap = np.zeros((512, 512, 9, 9), dtype=np.float32)

    for i in range(0, 81):
        imgname = 'input_Cam' + str(i).zfill(3) + '.png'
        imgpath = os.path.join(data_folder, imgname)
        img = np.uint8(imageio.imread(imgpath))
        imageMxM[:, :, i // 9, i - 9 * (i // 9), :] = img
        # labelname = 'gt_disp_lowres_Cam' + str(i).zfill(3) + '.pfm'
        # labelpath = os.path.join(data_folder, labelname)
        # imgLabel = utils.read_pfm(labelpath)
        # deltmap[:, :, i // 9, i - 9 * (i // 9)] = imgLabel
    return imageMxM, deltmap


def root_to_npy(root_dir):
    name = ["antinous", "boardgames", "dishes", "greek", "medieval2",
            "pens", "pillows", "platonic", "rosemary", "table", "tomb", "tower", "town"]
    for i in range(len(name)):
        path1 = os.path.join(root_dir, name[i])
        data, label = read_data(path1)
        npy_path = os.path.join(path1, 'image.npy')
        np.save(npy_path, data)


def generate(data_path1, data_path2, target_path):
    data1, label1 = read_data(data_path1)
    data2, label2 = read_data(data_path2)

    data1, label1 = trans_or_rot(data1, label1)
    data2, label2 = trans_or_rot(data2, label2)

    mix_data, mix_label = mix_occlude(data1, label1, data2, label2)
    for i in range(0, 9):
        for j in range(0, 9):
            img_path = os.path.join(target_path, 'input_Cam' + str(i * 9 + j).zfill(3) + '.png')
            imageio.imsave(img_path, np.uint8(mix_data[:, :, i, j, :]))

            labelname = 'depth_img_' + str(i * 9 + j).zfill(3) + '.png'
            labelpath = os.path.join(target_path, labelname)
            depth = (mix_label[:, :, i, j] - mix_label[:, :, i, j].min()) / (
                        mix_label[:, :, i, j].max() - mix_label[:, :, i, j].min())

            depth = np.expand_dims((depth * 255.0), axis=-1)
            # depth = 0.9*depth + 0.1*mix_data[:, :, i, j, :]
            imageio.imsave(labelpath, np.uint8(depth))
            labelname = 'gt_disp_lowres_Cam' + str(i * 9 + j).zfill(3) + '.pfm'
            labelpath = os.path.join(target_path, labelname)
            utils.write_pfm(mix_label[:, :, i, j], labelpath)

    npy_path = os.path.join(target_path, 'image.npy')
    np.save(npy_path, mix_data)
    center = mix_label[:, :, 4, 4]
    labelname = 'gt_disp_lowres.pfm'
    labelpath = os.path.join(target_path, labelname)
    utils.write_pfm(center, labelpath)

    depth = (center - center.min()) / (center.max() - center.min())
    depth_path = os.path.join(target_path, 'depth_img.png')
    imageio.imsave(depth_path, np.uint8(depth * 255.0))


def random_data(root_dir, target_dir, num=5):
    name = ["antinous", "boardgames", "dishes", "greek", "medieval2",
            "pens", "pillows", "platonic", "rosemary", "table", "tomb", "tower", "town"]
    for i in range(num):
        one = np.random.randint(13)
        two = np.random.randint(13)
        while two == one:
            two = np.random.randint(13)
        path1 = os.path.join(root_dir, name[one])
        path2 = os.path.join(root_dir, name[two])
        print(path1)
        print(path2)
        target_path = os.path.join(target_dir, str(i).zfill(3))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        generate(path1, path2, target_path)


def delta_data_bilinear(root_dir, target_dir):
    names = ["antinous", "boardgames", "dishes", "greek", "medieval2",
             "pens", "pillows", "platonic", "rosemary", "table", "tomb", "tower", "town"]
    deltas = [0.2,0.5]
    for name in names:
        path = os.path.join(root_dir, name)
        data, label = read_data(path)
        for d in deltas:
            data_add1 = np.zeros_like(data)
            data_sub1 = np.zeros_like(data)
            print(name,d)
            for v in range(0, 9):  # v  双线性插值
                for u in range(0, 9):  # u
                    offsety = (4 - v)
                    offsetx = (4 - u)
                    mat_translation = np.float32([[1, 0, d * offsetx], [0, 1, d * offsety]])
                    data_add1[:, :, v, u, :] = cv2.warpAffine(
                        data[:, :, v, u, :],
                        mat_translation,
                        (512, 512))
                    mat_translation = np.float32([[1, 0, -d * offsetx], [0, 1, -d * offsety]])
                    data_sub1[:, :, v, u, :] = cv2.warpAffine(
                        data[:, :, v, u, :],
                        mat_translation,
                        (512, 512))

            add1_path = target_dir + '/' + name + '_add' + str(d) + '/'
            sub1_path = target_dir + '/' + name + '_sub' + str(d) + '/'
            if not os.path.exists(add1_path):
                os.makedirs(add1_path)
            if not os.path.exists(sub1_path):
                os.makedirs(sub1_path)
            for i in range(0, 9):
                for j in range(0, 9):
                    img_path = os.path.join(add1_path, 'input_Cam' + str(i * 9 + j).zfill(3) + '.png')
                    imageio.imsave(img_path, np.uint8(data_add1[:, :, i, j, :]))

                    img_path = os.path.join(sub1_path, 'input_Cam' + str(i * 9 + j).zfill(3) + '.png')
                    imageio.imsave(img_path, np.uint8(data_sub1[:, :, i, j, :]))

            npy_path = os.path.join(add1_path, 'image.npy')
            np.save(npy_path, data_add1)
            center_add1 = label[:, :, 4, 4] + d
            labelname = 'gt_disp_lowres.pfm'
            labelpath = os.path.join(add1_path, labelname)
            utils.write_pfm(center_add1, labelpath)
            depth = (center_add1 - center_add1.min()) / (center_add1.max() - center_add1.min())
            depth_path = os.path.join(add1_path, 'depth_img.png')
            imageio.imsave(depth_path, np.uint8(depth * 255.0))

            npy_path = os.path.join(sub1_path, 'image.npy')
            np.save(npy_path, data_sub1)
            center_sub1 = label[:, :, 4, 4] - d
            labelname = 'gt_disp_lowres.pfm'
            labelpath = os.path.join(sub1_path, labelname)
            utils.write_pfm(center_sub1, labelpath)
            depth = (center_sub1 - center_sub1.min()) / (center_sub1.max() - center_sub1.min())
            depth_path = os.path.join(sub1_path, 'depth_img.png')
            imageio.imsave(depth_path, np.uint8(depth * 255.0))


# 混合混合
def mixed_mixed_data(root_dir, start, num=5):
    # root = target
    for i in range(start, start + num):
        name = os.listdir(root_dir)
        one = np.random.randint(len(name))
        two = np.random.randint(len(name))
        while two == one:
            two = np.random.randint(len(name))
        path1 = os.path.join(root_dir, name[one])
        path2 = os.path.join(root_dir, name[two])
        print(path1)
        print(path2)
        target_path = os.path.join(root_dir, str(i).zfill(3))
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        generate(path1, path2, target_path)


def delta_data_phase_shift(root_dir, target_dir):
    # 写反了......
    names = ["antinous", "boardgames", "dishes", "greek", "medieval2",
             "pens", "pillows", "platonic", "rosemary", "table", "tomb", "tower", "town"]
    deltas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
    for name in names:
        path = os.path.join(root_dir, name)
        data, label = read_data(path)
        for d in deltas:
            # data_add1 = np.zeros_like(data)
            # data_sub1 = np.zeros_like(data)
            # print(name,d)
            # data_add1 = Image2FFt(data,d)
            # data_sub1 = Image2FFt(data,-d)
            #
            add1_path = target_dir + '/' + name + '_add' + str(d) + '/'
            sub1_path = target_dir + '/' + name + '_sub' + str(d) + '/'
            # if not os.path.exists(add1_path):
            #     os.makedirs(add1_path)
            # if not os.path.exists(sub1_path):
            #     os.makedirs(sub1_path)
            # for i in range(0, 9):
            #     for j in range(0, 9):
            #         img_path = os.path.join(add1_path, 'input_Cam' + str(i * 9 + j).zfill(3) + '.png')
            #         imageio.imsave(img_path, np.uint8(data_add1[:, :, i, j, :]))
            #
            #         img_path = os.path.join(sub1_path, 'input_Cam' + str(i * 9 + j).zfill(3) + '.png')
            #         imageio.imsave(img_path, np.uint8(data_sub1[:, :, i, j, :]))
            #
            # npy_path = os.path.join(add1_path, 'image.npy')
            # np.save(npy_path, data_add1)
            center_add1 = label[:, :, 4, 4] - d
            labelname = 'gt_disp_lowres.pfm'
            labelpath = os.path.join(add1_path, labelname)
            utils.write_pfm(center_add1, labelpath)
            depth = (center_add1 - center_add1.min()) / (center_add1.max() - center_add1.min())
            depth_path = os.path.join(add1_path, 'depth_img.png')
            imageio.imsave(depth_path, np.uint8(depth * 255.0))

            # npy_path = os.path.join(sub1_path, 'image.npy')
            # np.save(npy_path, data_sub1)
            center_sub1 = label[:, :, 4, 4] + d
            labelname = 'gt_disp_lowres.pfm'
            labelpath = os.path.join(sub1_path, labelname)
            utils.write_pfm(center_sub1, labelpath)
            depth = (center_sub1 - center_sub1.min()) / (center_sub1.max() - center_sub1.min())
            depth_path = os.path.join(sub1_path, 'depth_img.png')
            imageio.imsave(depth_path, np.uint8(depth * 255.0))

def Image2FFt(image, d):
    image = image
    H, W, M, M,C = image.shape
    Nh = np.fft.ifftshift([i for i in range(-int(np.fix(H / 2)), int(np.ceil(H / 2)))])  # 他是相当于用这个来移动了
    Nw = np.fft.ifftshift([i for i in range(-int(np.fix(W / 2)), int(np.ceil(W / 2)))])
    Nh, Nw = np.meshgrid(Nh, Nw)
    center_view = int(M / 2)
    pixel_feats = np.zeros((H, W, M, M,C))
    for v in range(0, M):
        for u in range(0, M):
            fft_image = np.fft.fftn(image[:, :, v, u,:])
            # print(v,u)
            offsetx = (center_view - u) * d
            offsety = (center_view - v) * d
            grid = np.exp(1j * 2 * np.pi * (offsetx * Nh / H + offsety * Nw / W))  # 这个grid是采样频率
            grid = grid.reshape(H, W, 1)
            image_back = np.fft.ifftn(fft_image * grid) * np.exp(-1j * 2)
            image_back = np.abs(image_back)
            image_back[image_back > 255] = 255
            image_back[image_back < 0] = 0
            image_back = image_back  # .real
            pixel_feats[:, :,v, u,:] = image_back

    return pixel_feats
