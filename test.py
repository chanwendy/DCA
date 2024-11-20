#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""

import os
import os.path as osp

import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from thop import profile

import numpy as np

from monai.networks.nets import SwinUNETR,UNETR

from helpers.utils import Metrics, AeroCLoader, parse_args,PaviaLoader
from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.unet import unet, unetm,OurUnet, unet_ES, Unet_DS, unet_ECS, unet_CS, unet_CPS

import argparse

def get_labels():
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

def decode_segmap(n_classes, label_mask, plot=False):
    label_colours = get_labels()
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.uint8(rgb)


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')    
    
    ### 0. Config file?
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 51, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)# pavia u 103
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')


    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default = 'unet_ecs', help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_false', help = 'Use mini version of network?')


    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 9, help = 'How many blocks if ResNet architecture?', type = int)


    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_false', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_false', help = 'SE layer uses ReLU or PReLU activation?')


    ### Load weights post network config
    parser.add_argument('--network_weights_path', default = "/HDD_data/MING/HS/model/pavia_unetecs_model_lr1.542_b16.pth", help = 'Path to Saved Network weights')


    ### Use GPU or not
    parser.add_argument('--use_cuda', action = 'store_false', help = 'use GPUs?')


    args = parse_args(parser)
    print(args)

    args.use_mini = False
    args.use_SE = False
    args.use_preluSE = False
    # best model
    # args.network_weights_path = '/home/user02/TUTMING/ming/Aero/result/unetecs4_focal_dice_model_lr1.542.pth'
    args.network_weights_path = '/HDD_data/MING/HS/model/result/unetecs4_focal_dice_model_lr2.542.pth'
    # args.network_weights_path = '/HDD_data/MING/HS/model/pavia_unetecs_model_lr942_b16.pth'

    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    perf = Metrics()
    
    tx = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
    
    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = '{}b'.format(args.bands), transforms = tx)
    elif args.bands == 31:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'visible', transforms = tx)
    elif args.bands == 51:
        testset = AeroCLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'all', transforms = tx)
        # testset = PaviaLoader(set_loc = 'right', set_type = 'test', size = 'small', hsi_sign = args.hsi_c, hsi_mode = 'all', transforms = tx)
    else:
        raise NotImplementedError('required parameter not found in dictionary')
    
    print('Completed loading data...')
    
    if args.network_arch == 'resnet':
        net = ResnetGenerator(args.bands, 6, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'segnet':
        if args.use_mini == True:
            net = segnetm(args.bands, 6)
        else:
            net = segnet(args.bands, 6)
    elif args.network_arch == 'unet':
        if args.use_mini == True:
            net = unetm(args.bands, 6, use_SE = args.use_SE, use_PReLU = args.use_preluSE)
        else:
            net = unet(args.bands, 6)
    elif args.network_arch == "our_unet":
        net = OurUnet(51,6, 1)
    elif args.network_arch == 'unet_ds':
        net = Unet_DS(args.bands, 6)
    elif args.network_arch == 'unet_es':
        net = unet_ES(args.bands, 6)
    elif args.network_arch == 'unet_cs':
        net = unet_CS(args.bands, 6)
    elif args.network_arch == 'unet_ecs':
        net = unet_ECS(args.bands, 6)
    elif args.network_arch == 'unet_cps':
        net = unet_CPS(args.bands, 6)
    elif args.network_arch == "swin_vit":
        net = SwinUNETR(
            img_size=(64, 64),
            in_channels=51,
            out_channels=6,
            feature_size=24,
            use_checkpoint=True,
            spatial_dims=2
        ).to(device)
    elif args.network_arch == "vit":

        net = UNETR(
            in_channels=51,
            out_channels=6,
            img_size=(64, 64),
            feature_size=16,
            hidden_size=384,
            mlp_dim=1536,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            dropout_rate=0.0,
            spatial_dims=2
        ).to(device)
    else:
        raise NotImplementedError('required parameter not found in dictionary')

    net.load_state_dict(torch.load(args.network_weights_path))
    net.eval()
    net.to(device)
    
    print('Completed loading pretrained network weights...')
    
    print('Calculating prediction accuracy...')
    
    labels_gt = []
    labels_pred = []
    
    for img_idx in range(len(testset)):
        rgb, hsi, label = testset[img_idx]
        label = label.numpy()
        
        label_pred = net(hsi.unsqueeze(0).to(device))



        label_pred = label_pred.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        # segmap = decode_segmap(10, label_mask=label_pred)
        # labelmap = decode_segmap(10, label_mask=label)


        # result = cv2.add(segmap.transpose(1,0,2), np.uint8(rgb.permute(1, 2, 0).numpy()))
        # result2 = cv2.add(labelmap, np.uint8(rgb.permute(1, 2, 0).numpy()))
        # cv2.imwrite('/HDD_data/MING/HS-ISD/res9_result2/{}.jpg'.format(testset.filelist[img_idx]), result)
        # cv2.imwrite('/HDD_data/MING/HS/vis/ecs_result/{}.jpg'.format(testset.filelist[img_idx]), segmap)

        # cv2.imwrite('/HDD_data/MING/HS/vis/label/{}.jpg'.format(testset.filelist[img_idx]), labelmap)
        label = label.flatten()
        label_pred = label_pred.flatten()

        labels_gt = np.append(labels_gt, label)
        labels_pred = np.append(labels_pred, label_pred)
    flops, params = profile(net.to(device), inputs=(hsi.unsqueeze(0).to(device),))
    scores = perf(labels_gt, labels_pred)
    print("参数量：", params)
    print("FLOPS：", flops)
    print('Statistics on Test set:\n')
    print('Overall accuracy = {:.2f}%\nAverage Accuracy = {:.2f}%\nMean IOU is {:.2f}\
          \nMean DICE score is {:.2f}'.format(scores[0]*100, scores[1]*100, scores[2]*100, scores[3]*100))
