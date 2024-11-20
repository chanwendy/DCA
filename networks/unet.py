#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:56:02 2019

@author: aneesh
"""
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from .selayer import ChannelSELayer
from torch.nn import Softmax


def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1) # B*W W H

class Position(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self, in_dim):
        super(Position,self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x) # B C H W
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height) # BW C H 32 64 32
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)   # BH C W 32 64 32
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3) # B H W H
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width) # B H W W
        concate = self.softmax(torch.cat([energy_H, energy_W], 3)) # B H W C

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height) # BW H H
        #print(concate)
        #print(att_H)
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class unetConv2(nn.Module):
    '''
    U-Net encoder block with Squeeze and Excitation layer flag and
    a default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
        is_batchnorm-- boolean flag to indicate batch-normalization usage 
        use_se      -- boolean flag to indicate if SE block is used
        act         -- flag to indicate activation between linear layers in SE 
                        (relu vs. prelu)
    ''' 
    def __init__(self, in_size, out_size, is_batchnorm, use_se = False, use_prelu = False):
        super(unetConv2, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_size, out_size, 3, 1, 1), nn.BatchNorm2d(out_size), nn.ReLU()
            )
        else:
            self.conv1 = nn.Sequential(nn.Conv2d(in_size, out_size, 3, 1, 1), nn.ReLU())
            self.conv2 = nn.Sequential(nn.Conv2d(out_size, out_size, 3, 1, 1), nn.ReLU())
        
        if use_se == True and use_prelu == True:
            self.se_layer1 = ChannelSELayer(out_size, act = 'prelu')
            self.se_layer2 = ChannelSELayer(out_size, act = 'prelu')
        elif use_se == True and use_prelu == False:
            self.se_layer1 = ChannelSELayer(out_size, act = 'relu')
            self.se_layer2 = ChannelSELayer(out_size, act = 'relu')
        else:
            self.se_layer1 = None
            self.se_layer2 = None

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        if self.se_layer1 is not None:
            outputs = self.se_layer1(outputs)
        
        outputs = self.conv2(outputs)
        
        if self.se_layer2 is not None:
            outputs = self.se_layer2(outputs)
        return outputs

class unetUp(nn.Module):
    '''
    U-Net decoder block with default kernel size of 3
    
    Parameters:
        in_size     -- number of input channels
        out_size    -- number of output channels
        is_deconv   -- boolean flag to indicate if interpolation or de-convolution
                        should be used for up-sampling
    '''
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))

class unet(nn.Module):
    '''
    U-Net architecture
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
        is_deconv       -- boolean flag to indicate if interpolation or de-convolution
                            should be used for up-sampling
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
    ''' 
    def __init__(self, in_channels=3, out_channels = 21, feature_scale=1, is_deconv=True, is_batchnorm=True):
        super(unet, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        # filters = [16,32,64,128,256]

        filters = [int(x / self.feature_scale) for x in filters]
        
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)

        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)

        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)


        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)


        up4 = self.up_concat4(conv4, center)

        up3 = self.up_concat3(conv3, up4)

        up2 = self.up_concat2(conv2, up3)

        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final





class unet_ECS(nn.Module):
    '''
    U-Net architecture

    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
        is_deconv       -- boolean flag to indicate if interpolation or de-convolution
                            should be used for up-sampling
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
    '''

    def __init__(self, in_channels=3, out_channels=21, feature_scale=1, is_deconv=True, is_batchnorm=True):
        super(unet_ECS, self).__init__()

        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        # filters = [16,32,64,128,256]

        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.sc1 = CAM_Module(filters[0])
        self.ps1 = Position(filters[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.sc2 = CAM_Module(filters[1])
        self.ps2 = Position(filters[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.sc3 = CAM_Module(filters[2])
        self.ps3 = Position(filters[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.sc4 = CAM_Module(filters[3])
        self.ps4 = Position(filters[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
        # self.sc5 = CAM_Module(filters[4])
        # self.ps5 = PAM_Module(filters[4])

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        # conv1 = self.ps1(conv1)
        # ipdb.set_trace()
        maxpool1 = self.maxpool1(conv1)
        conv1_c = self.sc1(maxpool1)
        conv1_p = self.ps1(maxpool1)
        maxpool1 = conv1_c + conv1_p

        conv2 = self.conv2(maxpool1)
        # conv2 = self.ps2(conv2)
        maxpool2 = self.maxpool2(conv2)
        conv2_c = self.sc2(maxpool2)
        conv2_p = self.ps2(maxpool2)
        maxpool2 = conv2_c + conv2_p

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv3_c = self.sc3(maxpool3)
        conv3_p = self.ps3(maxpool3)
        maxpool3 = conv3_c + conv3_p

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        conv4_c = self.sc4(maxpool4)    # torch.Size([48, 512, 1, 1])
        conv4_p = self.ps4(maxpool4)
        maxpool4 = conv4_c + conv4_p

        center = self.center(maxpool4) # torch.Size([48, 1024, 1, 1])


        up4 = self.up_concat4(conv4, center)

        up3 = self.up_concat3(conv3, up4)

        up2 = self.up_concat2(conv2, up3)

        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final



# class unet(nn.Module):
#     '''
#     U-Net architecture
#
#     Parameters:
#         in_channels     -- number of input channels
#         out_channels    -- number of output channels
#         feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
#         is_deconv       -- boolean flag to indicate if interpolation or de-convolution
#                             should be used for up-sampling
#         is_batchnorm    -- boolean flag to indicate batch-normalization usage
#     '''
#
#     def __init__(self, in_channels=3, out_channels=21, feature_scale=1, is_deconv=True, is_batchnorm=True):
#         super(unet, self).__init__()
#
#         self.is_deconv = is_deconv
#         self.in_channels = in_channels
#         self.is_batchnorm = is_batchnorm
#         self.feature_scale = feature_scale
#
#         filters = [64, 128, 256, 512, 1024]
#         # filters = [16,32,64,128,256]
#
#         filters = [int(x / self.feature_scale) for x in filters]
#
#         # downsampling
#         self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
#
#         self.maxpool1 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
#
#         self.maxpool2 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
#
#         self.maxpool3 = nn.MaxPool2d(kernel_size=2)
#
#         self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
#
#         self.maxpool4 = nn.MaxPool2d(kernel_size=2)
#
#         self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)
#
#
#         # upsampling
#         self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
#         self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
#         self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
#         self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)
#
#         # final conv (without any concat)
#         self.final = nn.Conv2d(filters[0], out_channels, 1)
#
#     def forward(self, inputs):
#         conv1 = self.conv1(inputs)
#         # conv1 = self.ps1(conv1)
#         # ipdb.set_trace()
#
#         maxpool1 = self.maxpool1(conv1)
#
#         conv2 = self.conv2(maxpool1)
#         # conv2 = self.ps2(conv2)
#
#         maxpool2 = self.maxpool2(conv2)
#
#         conv3 = self.conv3(maxpool2)
#
#         maxpool3 = self.maxpool3(conv3)
#
#         conv4 = self.conv4(maxpool3)
#
#         maxpool4 = self.maxpool4(conv4)
#
#         center = self.center(maxpool4)
#
#
#         up4 = self.up_concat4(conv4, center)
#         up3 = self.up_concat3(conv3, up4)
#         up2 = self.up_concat2(conv2, up3)
#         up1 = self.up_concat1(conv1, up2)
#
#         final = self.final(up1)
#
#         return final

def Normalization(norm_type, out_channels,num_group=1):
    if norm_type==1:
        return nn.InstanceNorm2d(out_channels)
    elif norm_type==2:
        return nn.BatchNorm2d(out_channels,momentum=0.1)

class Conv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2, num_group=1,activation=True, norm=True,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class DoubleConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,soft=False):
        super().__init__()
        activation = torch.nn.Softplus() if soft else torch.nn.ReLU(inplace=False)
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation,
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            Normalization(norm_type,out_ch),
            activation
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class Up(torch.nn.Module):
    def __init__(self, in_ch, out_ch,norm_type=2,kernal_size=(2,2),stride=(2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )


    def forward(self, x):
        x = self.conv(x)
        return x

class Down(torch.nn.Module):
    def __init__(self, in_ch, out_ch, norm_type=2,kernal_size=(2,2),stride=(2,2),soft=False):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, in_ch, kernal_size, stride=stride, padding=0),
            DoubleConv(in_ch, out_ch, norm_type,soft=False)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


    
class unetm(nn.Module):
    '''
    mini U-Net architecture with 2 downsampling & upsampling blocks and one bottleneck
    with Squeeze and Excitation layers
    
    Parameters:
        in_channels     -- number of input channels
        out_channels    -- number of output channels
        feature_scale   -- scale for scaling default filter range in U-Net (default: 2)
        is_deconv       -- boolean flag to indicate if interpolation or de-convolution
                            should be used for up-sampling
        is_batchnorm    -- boolean flag to indicate batch-normalization usage
        use_SE          -- boolean flag to indicate SE blocks usage
        use_PReLU       -- boolean flag to indicate activation between linear layers in SE 
                            (relu vs. prelu)
    '''
    def __init__(self, in_channels=3, out_channels = 21, feature_scale=1, 
                 is_deconv=True, is_batchnorm=True, use_SE = False, use_PReLU = False):
        super(unetm, self).__init__()
        
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.use_SE = use_SE
        self.use_PReLU = use_PReLU

        filters = [64, 128, 256, 512, 1024]
#        filters = [128, 256, 512, 1024, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[1], filters[2], self.is_batchnorm, use_se = self.use_SE, use_prelu = self.use_PReLU)

        # upsampling
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, inputs):
        
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        center = self.center(maxpool2)
        up2 = self.up_concat2(conv2, center)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final