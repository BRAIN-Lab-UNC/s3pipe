#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:42:32 2022

@author: fenqiang
"""
    
import torch
import numpy as np

from s3pipe.utils.utils import get_par_fs_lookup_table

lookup_table_vec, lookup_table_scalar, lookup_table_name = get_par_fs_lookup_table()


def compute_dice(pred, gt, NUM_ROIS=36, device=torch.device('cpu')):
    assert len(pred) == len(gt), "Number of labels is not consistent!"
    if torch.is_tensor(pred):
        pred_mask = torch.zeros((pred.shape[0], NUM_ROIS), dtype=torch.float32, device=device)
        pred_mask[np.asarray(list(range(pred_mask.shape[0]))), pred] = 1.0
        gt_mask = torch.zeros((gt.shape[0], NUM_ROIS), dtype=torch.float32, device=device)
        gt_mask[np.asarray(list(range(gt_mask.shape[0]))), gt] = 1.0
    
        tmp = gt_mask.sum(0)
        tmp[tmp==0] = 1
        dice = torch.mean(torch.div(2*torch.sum(pred_mask * gt_mask, dim=0),
                                    pred_mask.sum(0)+tmp))
        return dice.item()
    else:
        if len(pred.shape) == 2 and pred.shape[1] == 3:
            pred = np.where((pred[:, np.newaxis] == lookup_table_vec).all(2))[1]
            gt = np.where((gt[:, np.newaxis] == lookup_table_vec).all(2))[1]
            
        dice = np.zeros(NUM_ROIS)
        for i in range(NUM_ROIS):
            gt_indices = np.where(gt == i)[0]
            pred_indices = np.where(pred == i)[0]
            dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
        return dice
