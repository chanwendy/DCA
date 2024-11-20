#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""

import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from thop import profile

import numpy as np

from monai.networks.nets import SwinUNETR,UNETR

from helpers.utils import Metrics, AeroCLoader, parse_args
from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.unet import unet, unetm,OurUnet, unet_ES, Unet_DS, unet_ECS, unet_CS, unet_CPS
from albumentations import RandomScale,RandomCrop,VerticalFlip, HorizontalFlip, Normalize, Compose,Resize
from LoveDA import LoveDA
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
    parser.add_argument('--bands', default = 3, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')


    ### 2. Network selections
    ### a. Which network?
    parser.add_argument('--network_arch', default = 'vit', help = 'Network architecture?')
    parser.add_argument('--use_mini', action = 'store_false', help = 'Use mini version of network?')


    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 9, help = 'How many blocks if ResNet architecture?', type = int)


    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_false', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_false', help = 'SE layer uses ReLU or PReLU activation?')


    ### Load weights post network config
    parser.add_argument('--network_weights_path', default = "/home/user02/TUTMING/ming/Aero/result/vit_DA_lr10_urban256.pth", help = 'Path to Saved Network weights')


    ### Use GPU or not
    parser.add_argument('--use_cuda', action = 'store_false', help = 'use GPUs?')


    args = parse_args(parser)
    print(args)

    args.use_mini = False
    args.use_SE = False
    args.use_preluSE = False
    # args.network_weights_path = '/home/user02/TUTMING/ming/Aero/result/unetecs4_focal_dice_model_lr1.542.pth'
    args.network_weights_path = '/home/user02/TUTMING/ming/Aero/result/vit1024_DA_lr11_urban256.pth'

    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    perf = Metrics()
    
    # tx = Compose([
    #         RandomScale((0.5, 2.0)),
    #         RandomCrop(512, 512),
    #         VerticalFlip(p=0.5),
    #         HorizontalFlip(p=0.5),
    #         Normalize(
    #             mean=[0.5, 0.5, 0.5],
    #             std=[0.5, 0.5, 0.5],
    #         ),
    #     ])

    # tx = Compose([
    #         Resize(256,256),
    #         VerticalFlip(p=0.5),
    #         HorizontalFlip(p=0.5),
    #         Normalize(
    #             mean=[0.5, 0.5, 0.5],
    #             std=[0.5, 0.5, 0.5],
    #         ),
    #     ])

    tx = Compose([
            RandomScale((0.5, 2.0)),
            RandomCrop(512, 512),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    tl = Compose([
            Resize(256,256),
            Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])


    loveda_root = '/HDD_data/MING/HS/loveda'

    loveda_train = LoveDA(loveda_root, split="train", scene=['urban', 'rural'], transforms=tx)
    loveda_val = LoveDA(loveda_root, split="val", scene=['urban', 'rural'], transforms=tl)

    trainloader = torch.utils.data.DataLoader(loveda_train, batch_size=256, shuffle=True)
    valloader = torch.utils.data.DataLoader(loveda_val, batch_size=256, shuffle=False)



    print('Completed loading data...')
    
    if args.network_arch == 'resnet':
        net = ResnetGenerator(args.bands, 8, n_blocks=args.resnet_blocks)
    elif args.network_arch == 'segnet':
        if args.use_mini == True:
            net = segnetm(args.bands, 8)
        else:
            net = segnet(args.bands, 8)
    elif args.network_arch == 'unet':
        if args.use_mini == True:
            net = unetm(args.bands, 8, use_SE = args.use_SE, use_PReLU = args.use_preluSE)
        else:
            net = unet(args.bands, 8)
    elif args.network_arch == "our_unet":
        net = OurUnet(51,8, 1)
    elif args.network_arch == 'unet_ds':
        net = Unet_DS(args.bands, 8)
    elif args.network_arch == 'unet_es':
        net = unet_ES(args.bands, 8)
    elif args.network_arch == 'unet_cs':
        net = unet_CS(args.bands, 8)
    elif args.network_arch == 'unet_ecs':
        net = unet_ECS(args.bands, 8)
    elif args.network_arch == 'unet_cps':
        net = unet_CPS(args.bands, 8)
    elif args.network_arch == "swin_vit":
        net = SwinUNETR(
            img_size=(256, 256),
            in_channels=3,
            out_channels=8,
            feature_size=48,
            use_checkpoint=True,
            spatial_dims=2
        ).to(device)


    elif args.network_arch == "vit":


        net = UNETR(
            in_channels=3,
            out_channels=8,
            img_size=(256, 256),
            feature_size=16,
            hidden_size=1024,
            mlp_dim=4096,
            num_heads=8,
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
    

    total_oa = 0
    total_aa = 0
    total_iou = 0
    total_dice = 0
    with torch.no_grad():
        for idx, (data) in enumerate(valloader, 0):
            truth = []
            pred = []
            #        print(idx)
            hsi_ip = data[0]
            labels = data[1]
            N = hsi_ip.size(0)

            outputs = net(hsi_ip.to(device))

            # loss = criterion(outputs, labels.to(device))

            # valloss_fx += loss.item()

            # valloss2.update(loss.item(), N)

            truth = np.append(truth, labels.cpu().numpy())
            pred = np.append(pred, outputs.max(1)[1].cpu().numpy())
            oa, mpca, mIOU, _dice_coefficient, IOU = perf(truth, pred)
            total_oa += oa
            total_aa += mpca
            total_iou += mIOU
            total_dice += _dice_coefficient

    # flops, params = profile(net.to(device), inputs=(hsi.unsqueeze(0).to(device),))
    # scores = perf(labels_gt, labels_pred)
    # print("参数量：", params)
    # print("FLOPS：", flops)
    OA  = total_oa / (idx + 1)
    AA  = total_aa / (idx + 1)
    MIOU  = total_iou / (idx + 1)
    DICE  = total_dice / (idx + 1)

    print('Statistics on Test set:\n')
    print('Overall accuracy = {:.2f}%\nAverage Accuracy = {:.2f}%\nMean IOU is {:.2f}\
          \nMean DICE score is {:.2f}'.format(OA*100, AA*100, MIOU*100, DICE*100))


