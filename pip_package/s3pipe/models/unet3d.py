#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:04:59 2022

@author: Fenqiang Zhao
"""



import torch
import torch.nn as nn

 

class down_block(nn.Module):
    """
    downsampling block in unet
    mean/max pooling => (conv => BN => ReLU) * 2
    
    """
    def __init__(self, in_ch, out_ch, first=False, pooling_type='max'):
        super(down_block, self).__init__()

        if first:
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_ch, momentum=0.15, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_ch, momentum=0.15, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                )
            
        else:
            if pooling_type == 'mean':
                pool_layer = nn.AvgPool3d(kernel_size=2)
            else:
                pool_layer = nn.MaxPool3d(kernel_size=2)
            self.block = nn.Sequential(
                pool_layer,
                nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_ch, momentum=0.15, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm3d(out_ch, momentum=0.15, track_running_stats=True),
                nn.LeakyReLU(0.2, inplace=True),
        )


    def forward(self, x):
        x = self.block(x)
        return x


class up_block(nn.Module):
    """Define the upsamping block in unet
    upconv => (conv => BN => ReLU) * 2
    
    Parameters:
            in_ch (int) - - input features/channels
            out_ch (int) - - output features/channels    
            
    """    
    def __init__(self, in_ch, out_ch):
        super(up_block, self).__init__()
        
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch, momentum=0.15, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_ch, momentum=0.15, track_running_stats=True),
            nn.LeakyReLU(0.2, inplace=True),
            )

    def forward(self, x1, x2):
        
        x1 = self.up(x1)
        x = torch.cat((x1, x2), 1)
        x = self.double_conv(x)

        return x
    


class Unet3D(nn.Module):
    """Define the UNet3D structure

    """    
    def __init__(self, in_ch, out_ch, level=5, init_ch=8):
        """ Initialize the 3D UNet.

        """
        super(Unet3D, self).__init__()
        
        chs = [in_ch]
        for i in range(level):
            chs.append(2**i*init_ch)
        
        self.down = nn.ModuleList([])
        for i in range(level):
            if i == 0:
                self.down.append(down_block(chs[i], chs[i+1], first=True, pooling_type='max'))
            else:
                self.down.append(down_block(chs[i], chs[i+1], first=False, pooling_type='max'))
      
        self.up = nn.ModuleList([])
        for i in range(level-1):
            self.up.append(up_block(chs[level-i], chs[level-1-i]))
            
        self.outc = nn.Conv3d(chs[1], out_ch, kernel_size=1, stride=1, padding=0)
        self.level = level
        
    def forward(self, x):
        # x's size should be NxCxDxHxW
        xs = [x]
        # print("x.shape: ", x.shape)
        for i in range(self.level):
            xs.append(self.down[i](xs[i]))
            # print("xs[-1].shape: ", xs[-1].shape)

        x = xs[-1]
        decoder_feat = [x]
        for i in range(self.level-1):
            x = self.up[i](x, xs[self.level-1-i])
            # print(x.shape)
            decoder_feat.append(x)

        tisse_seg = self.outc(x) # Nx7xDxHxW
        # print("tisse_seg shape: ", tisse_seg.shape)
        return tisse_seg, decoder_feat
        