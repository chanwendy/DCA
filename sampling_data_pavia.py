#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:12:08 2019

@author: aneesh
"""

import os
import os.path as osp
import numpy as np
import scipy.io as scio
from skimage import io
from PIL import Image

def normalize(channel):
    return ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)

def create_splits(loc, size_chips, rgb, hsi, labels):
    """
    Creates the chips for for train, validation and test sets
    Parameters:
        loc: path to save the created chips. This automatically creates
        subfolders of all underlying domains - RGB, HSI & Labels
        size_chips: size for each chip, preset: 64 x 64
        rgb: path to RGB flight line
        rad: path to Radiance-calibrated flight line
        ref: path to Reflectance-converted flight line
        labels: path to Labels for the flight line
    creates individual chips, with 50% overlap, in the corresponding category and outputs two
    textfiles - train.txt, test.txt. train.txt is a list of all filenames in
    the folder, test.txt contains only non-overlapping chips.
    """
    
    os.makedirs(loc, exist_ok = True)
    os.makedirs(osp.join(loc, 'RGB'), exist_ok = True)
    os.makedirs(osp.join(loc, 'HSI-rad'), exist_ok = True)
    os.makedirs(osp.join(loc, 'Labels'), exist_ok = True)

    print('Starting chip making now')
    
    trainfile = open(osp.join(loc, 'train.txt'), 'w')
    testfile = open(osp.join(loc, 'test.txt'), 'w')

    x_arr, y_arr, _ = rgb.shape
    
    for xx in range(0, x_arr - size_chips, size_chips//4):
        for yy in range(0, y_arr - size_chips, size_chips//4):

            name = 'image_{}_{}'.format(xx,yy)
            
            rgb_temp = rgb[xx:xx + size_chips, yy:yy + size_chips,:]
            rgb_temp = Image.fromarray(rgb_temp)
            
            hsi_rad_temp = hsi[xx:xx + size_chips, yy:yy + size_chips]

            labels_temp = labels[xx:xx + size_chips, yy:yy + size_chips]
            labels_temp = Image.fromarray(labels_temp)
            
            rgb_temp.save(osp.join(loc, 'RGB', name + '.tif'))
            labels_temp.save(osp.join(loc, 'Labels', name + '.tif'))
            np.save(osp.join(loc, 'HSI-rad', name), hsi_rad_temp)

            trainfile.write("%s\n" % name)
            
            if (xx%size_chips == 0 and yy%size_chips == 0):
                testfile.write("%s\n" % name)

    trainfile.close()
    testfile.close()
    
    print('Stopping chip making now')

if __name__ == "__main__":
    
    folder_dir = osp.join('/home/user02/TUTMING/ming/Aero/Aerial Data', 'Collection') #path to full files

    image_hsi = scio.loadmat(osp.join(folder_dir, 'PaviaU.mat'))['paviaU']
    blue_channel = image_hsi[:, :, 9]
    green_channel = image_hsi[:, :, 26]
    red_channel = image_hsi[:, :, 51]
    blue = normalize(blue_channel)
    green = normalize(green_channel)
    red = normalize(red_channel)
    image_rgb = np.stack((red, green, blue), axis=-1)
    rgb_pil_image = Image.fromarray(image_rgb)

    labels = scio.loadmat(osp.join(folder_dir, 'PaviaU_gt.mat'))["paviaU_gt"]


###############################################################################
    # image1_rgb = image_rgb[:1088,:704,:]
    # image1_hsi = image_hsi[:1088,:704,:]
    # image1_labels = labels[:1088,:704]

    # image1_rgb = image_rgb[:1088,:512,:]
    # image1_hsi = image_hsi[:1088,:512,:]
    # image1_labels = labels[:1088,:512]

    image1_rgb = image_rgb[:576,:256,:]
    image1_hsi = image_hsi[:576,:256,:]
    image1_labels = labels[:576,:256]


    create_splits(loc = osp.join('/home/user02/TUTMING/ming/Aero/Aerial Data', 'PuImage32', 'Data-left'),
                  size_chips = 32,
                  rgb = image1_rgb,
                  hsi=image1_hsi,
                  labels = image1_labels
                  )

###############################################################################
    
    # image1_rgb = image_rgb[:1088,512:578,:]
    # image1_hsi = image_hsi[:1088,512:704,:]
    # image1_labels = labels[:1088,512:704]

    # image1_rgb = image_rgb[:1088,512:578,:]
    # image1_hsi = image_hsi[:1088,512:578,:]
    # image1_labels = labels[:1088,512:578]

    image1_rgb = image_rgb[:576,256:320,:]
    image1_hsi = image_hsi[:576,256:320,:]
    image1_labels = labels[:576,256:320]
    
    create_splits(loc = osp.join('/home/user02/TUTMING/ming/Aero/Aerial Data', 'PuImage32', 'Data-mid'),
                  size_chips = 32,
                  rgb = image1_rgb, 
                  hsi=image1_hsi,
                  labels = image1_labels
                  )

###############################################################################
    
    # image1_rgb = image_rgb[832:1088,:704,:]
    # image1_hsi = image_hsi[832:1088,:704,:]
    # image1_labels = labels[832:1088,:704]

    # image1_rgb = image_rgb[:1088,578:704,:]
    # image1_hsi = image_hsi[:1088,578:704,:]
    # image1_labels = labels[:1088,578:704]

    image1_rgb = image_rgb[:576,256:320,:]
    image1_hsi = image_hsi[:576,256:320,:]
    image1_labels = labels[:576,256:320]
    create_splits(loc = osp.join('/home/user02/TUTMING/ming/Aero/Aerial Data', 'PuImage32', 'Data-right'),
                  size_chips = 32,
                  rgb = image1_rgb, 
                  hsi=image1_hsi,
                  labels = image1_labels
                  )

###############################################################################

