import numpy
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def huber_loss(gt, pred, delta=0.10):
    loss = torch.where(torch.abs(gt - pred) < delta, 0.5 * ((gt - pred) ** 2),
                       delta * torch.abs(gt - pred) - 0.5 * (delta ** 2))
    return torch.sum(loss)


def edge_aware_per_pixel_loss(gt, pred,device=None):
    device = device
    def gradient_y(img):
        weight = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(device)
        gy = torch.cat([F.conv2d(img[batch_i, :, :, :].unsqueeze(0),
                                 weight.view((1, 1, 3, 3)), padding=1).to(device) for
                        batch_i in range(img.shape[0])], 0)
        return gy

    def gradient_x(img):
        weight = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).to(device)
        gx = torch.cat([F.conv2d(img[batch_i, :, :, :].unsqueeze(0),
                                 weight.view((1, 1, 3, 3)), padding=1).to(device) for
                        batch_i in range(img.shape[0])], 0)
        return gx

    pred = pred.to(device)
    gt = gt.to(device)
    pred_gradients_x = gradient_x(pred)  # B,1,H,W
    pred_gradients_y = gradient_y(pred)

    gt_gradients_x = gradient_x(gt)
    gt_gradients_y = gradient_y(gt)

    weights_x = torch.exp(-gt_gradients_x)  # 这是求image_X的梯度
    weights_y = torch.exp(-gt_gradients_y)  # 这是求image_Y的梯度

    smoothness_x = torch.abs(pred_gradients_x) * weights_x
    smoothness_y = torch.abs(pred_gradients_y) * weights_y

    return torch.mean(smoothness_x) + torch.mean(smoothness_y)




def uarl_loss(y_true, y_pred, device=None):

    y_true_label = y_true
    B, H, W, channel = y_pred.shape
    disparity_values = torch.linspace(-4, 4, channel).to(device)
    disparity_values = disparity_values.reshape(1, 1, 1, channel)
    x = disparity_values.repeat(B, H, W, 1)
    y_pred_label = (x * y_pred).sum(dim=-1).reshape(B, H, W, 1)
    # y_pred_label = y_pred

    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1, channel)

    y_true = torch.add(y_true, 4.)
    disl = y_true // 0.5
    disr = disl + 1

    wl = (0.5 * disr - y_true) / 0.5
    wr = (y_true - 0.5 * disl) / 0.5

    disl_one_hot = F.one_hot(disl.to(torch.int64), channel)
    disr_one_hot = F.one_hot(disr.to(torch.int64), channel)

    y_pred = y_pred.to(torch.float32)
    y_true = y_true.to(torch.float32)

    disl_one_hot = disl_one_hot.to(torch.float32)
    disr_one_hot = disr_one_hot.to(torch.float32)

    wl = wl.to(torch.float32)
    wr = wr.to(torch.float32)

    # JS divergence based on KL divergence
    y_target = disl_one_hot * wl.reshape(-1, 1) + \
        disr_one_hot * wr.reshape(-1, 1)
    y_middle = 0.5 * (y_target + y_pred)

    # beta stands the exp weights
    beta = 0.1
    KL_criterion = torch.nn.KLDivLoss(size_average=False)
    uncertainty_map = 0.5 * (KL_criterion(y_pred, y_middle) +
                             KL_criterion(y_target, y_middle))

    uncertainty_map = torch.clip(uncertainty_map, 1e-8, 1)
    mae_loss = abs(
        y_true_label.reshape(-1) - y_pred_label.reshape(-1))
    loss = torch.mean(uncertainty_map**beta * mae_loss)
    return loss


def rgb_loss(rgb_values, rgb_gt):
    rgb_values = rgb_values.reshape(-1,3)
    rgb_gt = rgb_gt.reshape(-1, 3)
    N,_ = rgb_gt.shape
    rgb_loss = (nn.L1Loss(reduction='sum')(rgb_values, rgb_gt))/ N
    return rgb_loss


if __name__ == '__main__':
    is_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda else "cpu")
    gt = torch.rand((2, 1, 32, 32)).to(device)
    pred = torch.rand((2, 1, 32, 32)).to(device)
    loss_edge = edge_aware_per_pixel_loss(gt, pred,device=device)
    print(loss_edge)
    loss_mae = torch.mean(torch.abs(gt - pred))
    print(loss_mae)
