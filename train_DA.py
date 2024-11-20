#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:34:16 2019

@author: aneesh
"""

import os 
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import ipdb
import monai
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import optim

from helpers.augmentations import RandomHorizontallyFlip, RandomVerticallyFlip, \
    RandomTranspose, Compose
from helpers.utils import AeroCLoader, AverageMeter, Metrics, parse_args, compute_sdf, boundary_loss
from helpers.lossfunctions import cross_entropy2d

from torchvision import transforms

from networks.resnet6 import ResnetGenerator
from networks.segnet import segnet, segnetm
from networks.unet import unet, unetm, OurUnet, Unet_DS, unet_ES, unet_ECS, unet_CS, unet_CPS
from networks.model_utils import init_weights, load_weights
from monai.networks.nets import SwinUNETR,UNETR
from monai.networks import one_hot
from mmdet.models.losses.focal_loss import FocalLoss
from LoveDA import LoveDA
import argparse
from albumentations import RandomScale,RandomCrop,VerticalFlip, HorizontalFlip, Normalize, Compose,Resize
# Define a manual seed to help reproduce identical results

def train(epoch = 0):
    global trainloss
    trainloss2 = AverageMeter()
    
    print('\nTrain Epoch: %d' % epoch)
    
    net.train()

    running_loss = 0.0
    
    for idx, (data) in enumerate(trainloader, 0):
        # print(idx)
        hsi_ip = data[0]
        labels = data[1]
        N = hsi_ip.size(0)
        optimizer.zero_grad()
        
        outputs = net(hsi_ip.to(device))
        # if idx == 2:
        #     ipdb.set_trace()
        loss = criterion(outputs, labels.to(device))    # labels (1, 1024, 1024)  outputs(1, 7, 1024, 1024)
        # ipdb.set_trace()
        diceloss = loss_function(outputs, labels.unsqueeze(dim=1).to(device))
        logit = softmax(outputs)

        loss_focal = focal_loss(logit, one_hot(labels.unsqueeze(dim=1).to(device).long(), num_classes=8))

        # gt_sdf_npy = compute_sdf(labels.unsqueeze(dim=1).cpu().numpy(), outputs.shape)
        # gt_sdf = torch.from_numpy(gt_sdf_npy).float().cuda(outputs.device.index)

        # loss_boundary = boundary_loss(outputs, gt_sdf)
        total_loss = diceloss + loss_focal
        # total_loss = loss + diceloss
        # total_loss = loss


        #
        total_loss.backward()
        optimizer.step()
        #
        running_loss += total_loss.item()
        trainloss2.update(total_loss.item(), N)


        if (idx + 1) %  5 == 0:
            print('[Epoch %d, Batch %5d] loss: %.3f' % (epoch + 1, idx + 1, running_loss / 5))
            running_loss = 0.0
    
    trainloss.append(trainloss2.avg)

def val(epoch = 0):
    global valloss
    valloss2 = AverageMeter()

    
    print('\nVal Epoch: %d' % epoch)
    
    net.eval()

    valloss_fx = 0.0
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
            # print("{0:.2f}".format((idx+1)/(len(loveda_val)/100)*100), end = '-', flush = True)
    OA  = total_oa / (idx + 1)
    AA  = total_aa / (idx + 1)
    MIOU  = total_iou / (idx + 1)
    DICE  = total_dice / (idx + 1)
    print('VAL: %d loss: %.3f' % (epoch + 1, valloss_fx / (idx+1)))
    valloss.append(valloss2.avg)

    return OA, AA, MIOU, DICE,IOU

if __name__ == "__main__":
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    parser = argparse.ArgumentParser(description = 'AeroRIT baseline evalutions')


    ### 0. Config file?
    parser.add_argument('--config-file', default = None, help = 'Path to configuration file')
    
    ### 1. Data Loading
    parser.add_argument('--bands', default = 3, help = 'Which bands category to load \
                        - 3: RGB, 4: RGB + 1 Infrared, 6: RGB + 3 Infrared, 31: Visible, 51: All', type = int)
    parser.add_argument('--hsi_c', default = 'rad', help = 'Load HSI Radiance or Reflectance data?')
    parser.add_argument('--use_augs', action = 'store_false', help = 'Use data augmentations?')

    ### 2. Network selections


    ### a. Which network?
    parser.add_argument('--network_arch', default = 'vit', help = 'Network architecture?')

    parser.add_argument('--use_mini', default=False, help = 'Use mini version of network?')

    ### b. ResNet config
    parser.add_argument('--resnet_blocks', default = 9, help = 'How many blocks if ResNet architecture?', type = int)


    ### c. UNet configs
    parser.add_argument('--use_SE', action = 'store_false', help = 'Network uses SE Layer?')
    parser.add_argument('--use_preluSE', action = 'store_false', help = 'SE layer uses ReLU or PReLU activation?')

    ### Save weights post network config
    parser.add_argument('--network_weights_path', default = "/home/user02/TUTMING/ming/Aero/result/vit1024_DA_lr11_urban256.pth", help = 'Path to save Network weights')


    ### Use GPU or not
    parser.add_argument('--use_cuda', default=True, help = 'use GPUs?')



    ### Hyperparameters
    parser.add_argument('--batch-size', default = 32, type = int, help = 'Number of images sampled per minibatch?')        # ours 48
    parser.add_argument('--init_weights', default = 'kaiming', help = "Choose from: 'normal', 'xavier', 'kaiming'")
    parser.add_argument('--learning-rate', default =1e-4, type = int, help = 'Initial learning rate for training the network?')
    parser.add_argument('--epochs', default = 21, type = int, help = 'Maximum number of epochs?')


    ### Pretrained representation present?
    parser.add_argument('--pretrained_weights', default = None, help = 'Path to pretrained weights for network')
    
    args = parse_args(parser)
    print(args)
    
    if args.use_cuda and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    perf = Metrics()
    
    # if args.use_augs:
    #     augs = []
    #     augs.append(RandomHorizontallyFlip(p = 0.5))
    #     augs.append(RandomVerticallyFlip(p = 0.5))
    #     augs.append(RandomTranspose(p = 1))
    #     augs_tx = Compose(augs)
    # else:
    #     augs_tx = None
        
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

    tx = Compose([
            Resize(256,256),
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

    if args.bands == 3 or args.bands == 4 or args.bands == 6:
        hsi_mode = '{}b'.format(args.bands)
    elif args.bands == 31:
        hsi_mode = 'visible'
    elif args.bands == 51:
        hsi_mode = 'all'
    else:
        raise NotImplementedError('required parameter not found in dictionary')
    loveda_root = '/HDD_data/MING/HS/loveda'

    loveda_train = LoveDA(loveda_root, split="val", scene=['urban', 'rural'], transforms=tx)
    loveda_val = LoveDA(loveda_root, split="val", scene=['urban', 'rural'], transforms=tl)


    trainloader = torch.utils.data.DataLoader(loveda_train, batch_size = args.batch_size, shuffle = True)
    valloader = torch.utils.data.DataLoader(loveda_val, batch_size = args.batch_size, shuffle = False)
    
    #Pre-computed weights using median frequency balancing    
    # weights = [1.11, 0.37, 0.56, 4.22, 6.77, 1.0]
    # weights = torch.FloatTensor(weights)
    
    criterion = cross_entropy2d(reduction = 'mean')
    loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    focal_loss = FocalLoss()
    softmax=  torch.nn.Softmax(dim=1)
    args.use_mini = False
    args.use_SE = False
    args.use_preluSE = False

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
    elif args.network_arch == 'unet_ds':
        net = Unet_DS(args.bands, 8)
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

    elif args.network_arch == "our_unet":
        net = OurUnet(51,8, 1)

    else:
        raise NotImplementedError('required parameter not found in dictionary')
    # if args.network_arch != "vit":
    #     init_weights(net, init_type=args.init_weights)

    if args.pretrained_weights is not None:
        load_weights(net, args.pretrained_weights)
        print('Completed loading pretrained network weights')

    net.to(device)
    
    optimizer = optim.Adam(net.parameters(), lr = args.learning_rate)
    # optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, weight_decay=1e-5)

    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,50])

    trainloss = []
    valloss = []
    
    bestmiou = 0
        
    for epoch in range(args.epochs):
        # scheduler.step()
        train(epoch)
        oa, mpca, mIOU, _dice_coefficient, _ = val(epoch)

        print('Overall acc  = {:.3f}, MPCA = {:.3f}, mIOU = {:.3f} dice{:.3f}'.format(oa, mpca, mIOU, _dice_coefficient))
        if mIOU > bestmiou:
            bestmiou = mIOU
            torch.save(net.state_dict(), args.network_weights_path)

    print("============================================")
    print("train finsih bestiou {}".format(bestmiou))
    print("============================================")

