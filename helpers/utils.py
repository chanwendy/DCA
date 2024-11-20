#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 12:11:14 2019

@author: aneesh
"""

import os
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix
from PIL import Image

def tensor_to_image(torch_tensor, mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)):
    '''
    Converts a 3D Pytorch tensor into a numpy array for display
    
    Parameters:
        torch_tensor -- Pytorch tensor in format(channels, height, width)
    '''
    for t, m, s in zip(torch_tensor, mean, std):
        t.mul_(s).add_(m)
    
    return np.uint8(torch_tensor.mul(255.0).numpy().transpose(1, 2, 0))
    
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def boundary_loss(outputs_soft, gt_sdf):
    """
    compute boundary loss for binary segmentation
    input: outputs_soft: sigmoid results,  shape=(b,2,x,y,z)
           gt_sdf: sdf of ground truth (can be original or normalized sdf); shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    pc = outputs_soft[:,...]
    dc = gt_sdf[:,...]
    multipled = torch.einsum('bxyz, bxyz->bxyz', pc, dc)
    bd_loss = multipled.mean()

    return bd_loss
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance
def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    """

    img_gt = img_gt.astype(np.uint8)

    gt_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b]
            negmask = 1-posmask
            posdis = distance(posmask)
            negdis = distance(negmask)
            boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
            sdf = negdis - posdis
            sdf[boundary==1] = 0
            gt_sdf[b][c] = sdf

    return gt_sdf

def compute_sdf_forsdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the Signed Distance Map (SDM)
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]

    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM)
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
    s_dtm = seg_dtm[:,1,...] ** 2
    g_dtm = gt_dtm[:,1,...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss
class Metrics():
    '''
    Calculates all the metrics reported in paper: Overall Accuracy, Average Accuracy,
    mean IOU and mean DICE score
    Ref: https://github.com/rmkemker/EarthMapper/blob/master/metrics.py
    
    Parameters:
        ignore_index -- which particular index to ignore when calculating all values.
                        In AeroRIT, index '5' is the undefined class and hence, the 
                        default value for this function.
    '''
    def __init__(self, ignore_index = 5):
        self.ignore_index = ignore_index
        
    def __call__(self, truth, prediction):
        
        # ignore_locs = np.where(truth == self.ignore_index)
        # truth = np.delete(truth, ignore_locs)
        # prediction= np.delete(prediction, ignore_locs)


        self.c = confusion_matrix(truth , prediction)
        return self._oa(), self._aa(), self._mIOU(), self._dice_coefficient(), self._IOU()
            
    def _oa(self):
        return np.sum(np.diag(self.c))/np.sum(self.c)
        
    def _aa(self):
        return np.nanmean(np.diag(self.c)/(np.sum(self.c, axis=1) + 1e-10))
    
    def _IOU(self):
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        union =  ground_truth_set + predicted_set - intersection + 1e-10
        
        intersection_over_union = intersection / union.astype(np.float32)
        
        return intersection_over_union
    
    def _mIOU(self):
        intersection_over_union = self._IOU()
        return np.nanmean(intersection_over_union)
    
    def _dice_coefficient(self):
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        dice = (2 * intersection) / (ground_truth_set + predicted_set + 1e-10)
        avg_dice = np.nanmean(dice)
        return avg_dice
    
class AeroCLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset
    
    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''
    
    def __init__(self, set_loc = 'left', set_type = 'train', size = 'small', hsi_sign = 'rad', 
                 hsi_mode = 'all', transforms = None, augs = None):
        
        if size == 'small':
            size = '64'
        else:
            raise Exception('Size not present in the dataset')
        
        self.working_dir = 'Image' + size
        self.working_dir = osp.join('Aerial Data', self.working_dir, 'Data-' + set_loc)
        
        self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)
        
        self.transforms = transforms
        self.augmentations = augs
        
        self.hsi_mode = hsi_mode
        self.hsi_dict = {
                '3b':[7, 15, 25],
                '4b':[7, 15, 25, 46],
                '6b':[7, 15, 25, 33, 40, 50], 
                'visible':'all 400 - 700 nm',
                'all': 'all 51 bands'}
        
        self.n_classes = 6
        
        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()
            
    def __getitem__(self, index):
        rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        rgb = rgb[:,:,::-1]
        
        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))
                
        
        if self.hsi_mode == 'visible':
            hsi = hsi[:,:,0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:,:,i] = hsi[:,:,bands[i]]
            hsi = hsi_temp
        
        hsi = hsi.astype(np.float32)
        
        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:,:,::-1]
        
        if self.augmentations is not None:
            rgb, hsi, label = self.augmentations(rgb, hsi, label)
        
        if self.transforms is not None:
            rgb = self.transforms(rgb)
            
            if self.hsi_sign == 'rad':
                hsi = np.clip(hsi, 0, 2**14)/2**14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            elif self.hsi_sign == 'ref':
                hsi = np.clip(hsi, 0, 100)/100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            
            label = self.encode_segmap(label)
            label = torch.from_numpy(np.array(label)).long()
            
        return rgb, hsi, label
    
    def __len__(self):
        return len(self.filelist)
    
    def get_labels(self):
        return np.asarray(
                [
                        [255, 0, 0],
                        [0, 255, 0],
                        [0, 0, 255],
                        [0, 255, 255],
                        [255, 127, 80],
                        [153, 0, 0],
                        ]
                )
    
    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask
    
    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        
        return np.uint8(rgb)


class PaviaLoader(data.Dataset):
    '''
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_loc     -- 'left', 'mid' or 'right' -> indicates which set is to be used
        set_type    -- 'train' or 'test'
        size        -- default 'small' -> 64 x 64.
        hsi_sign    -- 'rad' or 'ref' -> based on which hyperspectral image type is used
        hsi_mode    -- available sampling options are:
                        '3b' -> samples 7, 15, 25 (BGR)
                        '4b' -> samples bands 7, 15, 25, 46 (BGR + IR)
                        '6b' -> samples bands 7, 15, 25, 33, 40, 50 (BGR + 3x IR)
                        'visible' -> samples all visible bands
                        'all' -> samples all 51 bands (visible + infrared)
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    '''

    def __init__(self, set_loc='left', set_type='train', size='small', hsi_sign='rad',
                 hsi_mode='all', transforms=None, augs=None):

        if size == 'small':
            size = '32'
        else:
            raise Exception('Size not present in the dataset')

        self.working_dir = 'PuImage' + size
        self.working_dir = osp.join('/home/user02/TUTMING/ming/Aero/Aerial Data', self.working_dir, 'Data-' + set_loc)

        self.rgb_dir = 'RGB'
        self.label_dir = 'Labels'
        self.hsi_sign = hsi_sign
        self.hsi_dir = 'HSI' + '-{}'.format(self.hsi_sign)

        self.transforms = transforms
        self.augmentations = augs

        self.hsi_mode = hsi_mode
        self.hsi_dict = {
            '3b': [7, 15, 25],
            '4b': [7, 15, 25, 46],
            '6b': [7, 15, 25, 33, 40, 50],
            'visible': 'all 400 - 700 nm',
            'all': 'all 51 bands'}

        self.n_classes = 10

        with open(osp.join(self.working_dir, set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()

    def __getitem__(self, index):
        rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.tif'))
        rgb = rgb[:, :, ::-1]

        hsi = np.load(osp.join(self.working_dir, self.hsi_dir, self.filelist[index] + '.npy'))

        if self.hsi_mode == 'visible':
            hsi = hsi[:, :, 0:31]
        elif self.hsi_mode == 'all':
            hsi = hsi
        else:
            bands = self.hsi_dict[self.hsi_mode]
            hsi_temp = np.zeros((hsi.shape[0], hsi.shape[1], len(bands)))
            for i in range(len(bands)):
                hsi_temp[:, :, i] = hsi[:, :, bands[i]]
            hsi = hsi_temp

        hsi = hsi.astype(np.float32)

        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = label[:, :, ::-1]

        label =  Image.open(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.tif'))
        label = np.array(label.convert('L'))

        if self.augmentations is not None:
            rgb, hsi, label = self.augmentations(rgb, hsi, label)

        if self.transforms is not None:
            rgb = self.transforms(rgb)

            if self.hsi_sign == 'rad':
                hsi = np.clip(hsi, 0, 2 ** 14) / 2 ** 14
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)
            elif self.hsi_sign == 'ref':
                hsi = np.clip(hsi, 0, 100) / 100
                hsi = np.transpose(hsi, (2, 0, 1))
                hsi = torch.from_numpy(hsi)

            # label = self.encode_segmap(label)
            label = torch.from_numpy(np.array(label)).long()

        return rgb, hsi, label

    def __len__(self):
        return len(self.filelist)

    def get_labels(self):
        return np.asarray(
            [
                [255, 0, 0],
                [0, 255, 0],
                [0, 0, 255],
                [0, 255, 255],
                [255, 127, 80],
                [153, 0, 0],
                [255, 255, 0],
                [255, 0, 255],
                [0, 0, 153],
                [0, 153, 153]
            ]
        )

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, label_mask, plot=False):
        label_colours = self.get_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b

        return np.uint8(rgb)

def parse_args(parser):
    '''
    Standard argument parser
    '''
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
#        print (data)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args