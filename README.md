# DCA-Unet: Enhancing Small Object Segmentation in Hyperspectral Images with Dual Channel Attention Unet

## Structure of the repository
This repository is organized as:
* [Aerial Data](/Aerial%20Data/) This directory contains all the data required for training and analysis.
* [configs](/configs/) This directory contains a few examples for config files used in train and test functions.
* [helpers](/helpers/) This directory the helper zoo for non-model related functions.
* [networks](/networks/) This directory the model zoo used in the paper.
* [savedmodels](/savedmodels/) This directory will contain all the saved models post training.

## Dataset

The scene images can be found [here](https://drive.google.com/drive/folders/1yCMqa9uDC_CEGtbnxeWEQCTb-odC2r4c?usp=sharing). The directory contains four files: 
1. image_rgb - The RGB rectified hyperspectral scene.
2. image_hsi_radiance - Radiance calibrated hyperspectral scene sampled at every 10th band (400nm, 410nm, 420nm, .. 900nm).
3. image_hsi_reflectance - Reflectance calibrated hyperspectral scene sampled at every 10th band.
4. image_labels - Semantic labels for the entire AeroCampus scene.

Note: The above files only contain every 10th band from 400nm to 900nm. You can access the full versions of both radiance and reflectance at [GoogleDrive](https://drive.google.com/drive/folders/1FGLXUOKTG3VtFkAzn4lwrDNWKB3b7wEO?usp=drive_link).


## Requirements

numpy 

cv2 (opencv-python)

pytorch1.11.1

Pillow

We recommend to use [Anaconda](https://www.anaconda.com/distribution/) environment for running all sets of code. We have tested our code on Ubuntu 18.04 with Python 3.9.

## Executing codes

Before running any files, execute [sampling_data.py](/sampling_data.py/) to obtain train, validation and test splits with 64 x 64 image chips. 

Some of the important arguments used in [train](/train.py/) and [test](/test.py/) files are as follows:

| Argument | Description |
| -- | -- |
| config-file | path to configuration file if present |
| bands | how many bands to sample from HSI imagery (3 -> RGB, 51 -> all) ? |
| hsi_c | use HSI radiance or reflectance for analysis ? |
| network_arch | which network architecture to use: Res-U-Net, SegNet or U-Net? |
| network_weights_path | path to save(d) network weights |
| use_cuda | use GPUs for processing or CPU? |

 Both are great repositories - have a look!

