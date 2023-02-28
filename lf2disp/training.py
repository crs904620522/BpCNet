# from lf2disp import icp
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch


class BaseTrainer(object):
    ''' Base trainer class.
    '''

    def evaluate(self, val_loader):
        ''' Performs an evaluation.
        Args:
            val_loader (dataloader): pytorch dataloader
        '''
        eval_list = defaultdict(list)
        i = 0
        print("开始验证/测试")
        for data in tqdm(val_loader):
            torch.cuda.empty_cache()
            eval_step_dict = self.eval_step(data, imgid=i)
            print('val:', eval_step_dict)
            for k, v in eval_step_dict.items():
                eval_list[k].append(v)
            i += 1
        eval_dict = {k: np.mean(v) for k, v in eval_list.items()}
        return eval_dict

    def train_step(self, *args, **kwargs):
        ''' Performs a training step.
        '''
        raise NotImplementedError

    def eval_step(self, *args, **kwargs):
        ''' Performs an evaluation step.
        '''
        raise NotImplementedError

    def visualize(self, *args, **kwargs):
        ''' Performs  visualization.
        '''
        raise NotImplementedError
