#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 19:51:52 2022

@author: ychenp
"""

import torch
import torch.nn as nn
import numpy as np

from sphericalunet.models.unet3d import Unet3D
from sphericalunet.models.models import down_block, simple_up_block

from sphericalunet.utils.utils import Get_neighs_order, Get_upconv_index
from sphericalunet.models.layers import onering_conv_layer


class SurfReconNet(nn.Module):
    """
    surface reconstruction network,
    
    input: original 3D brain MRI image
    output: left inner, left outer,
            rigth inner, right outer, all with 40962 vertices
    
    """
    def __init__(self, in_ch, out_ch, level=5, init_ch=8, img_size=(160, 192, 160), device=None):
        super(SurfReconNet, self).__init__()

        self.chs = [in_ch]
        for i in range(level):
            self.chs.append(2**i*init_ch)

        self.unet3d = Unet3D(in_ch=in_ch, out_ch=out_ch, level=level, init_ch=init_ch)
        # sunet = SUnet(in_ch=, out_ch=, level=7, n_res=5)
        self.upsample = nn.Upsample(size=img_size, mode='trilinear')
        
        # four modules each for inferring a deformation field at a spherical resoulution level
        # 642, 2562, 10242, 40962
        # based on 3dunet decoder features
        self.out_deform_layers = nn.ModuleList([])
        for i in range(4):
            self.out_deform_layers.append(
                nn.Sequential(
                    nn.Conv3d(self.chs[4-i] * 2, 32, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm3d(32, momentum=0.15, track_running_stats=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(32, 3, kernel_size=3, stride=1, padding=1)
                    )
                   )
                
        self.img_size = img_size
        self.device = device
        
        # construct coordinate matrix
        coord = np.concatenate((np.repeat(np.arange(self.img_size[0]/2)[:, np.newaxis], self.img_size[1]/2, axis=1)[np.newaxis,:],
                                  np.repeat(np.arange(self.img_size[1]/2)[np.newaxis, :], self.img_size[0]/2, axis=0)[np.newaxis,:]),
                                  axis=0)
        coord = np.repeat(coord[:, :, :, np.newaxis], self.img_size[2]/2, axis=-1)
        coord = np.concatenate((coord,
                                np.repeat(np.repeat(np.arange(self.img_size[2]/2)[np.newaxis, np.newaxis, :], self.img_size[0]/2, axis=0), self.img_size[1]/2, axis=1)[np.newaxis,:]),
                                axis=0)
        coord = torch.from_numpy(coord.astype(np.int64) * 2).to(self.device)
        self.coord = torch.permute(coord, (1,2,3,0))
                  
    def forward(self, x, vertices, faces):
        tisse_seg, decoder_feat = self.unet3d(x)
                
        dist_map = compute_sdf_surf_np(vertices, faces, self.img_size)
        dist_map = dist_map.unsqueeze(0).to(self.device)
        deform = self.out_deform_layers[1](torch.cat((self.upsample(decoder_feat[2]), dist_map.repeat(1,64,1,1,1)), dim=1))
        
        return tisse_seg, decoder_feat, dist_map, deform
    


class ShapeAE(nn.Module):
    """
    
    """
    def __init__(self, in_ch=3, out_ch=3, level=7, n_res=3, complex_chs=16, code_dim=256):
        super(ShapeAE, self).__init__()

        assert (level-n_res) >=1, "number of resolution levels in unet should be at least 1 smaller than input level"
        self.n_res = n_res
        chs = [in_ch]
        for i in range(n_res):
            chs.append(2**i*complex_chs)
              
        conv_layer = onering_conv_layer
        neigh_orders = Get_neighs_order()
        neigh_orders = neigh_orders[8-level:8-level+n_res]
        upconv_indices = Get_upconv_index()
        upconv_indices = upconv_indices[16-2*level:16-2*level+(n_res-1)*2]
        self.n_vertex_last = int(len(neigh_orders[-1])/7)
        
        self.down = nn.ModuleList([])
        for i in range(n_res):
            if i == 0:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], None, True))
            else:
                self.down.append(down_block(conv_layer, chs[i], chs[i+1], neigh_orders[i], neigh_orders[i-1]))
        self.enc_fc = nn.Linear(chs[n_res]*self.n_vertex_last , code_dim)
       
        self.dec_fc = nn.Linear(code_dim, chs[n_res]*self.n_vertex_last)
        self.up = nn.ModuleList([])
        for i in range(n_res-1):
            self.up.append(simple_up_block(conv_layer, chs[n_res-i], chs[n_res-1-i],
                                    neigh_orders[n_res-2-i], upconv_indices[(n_res-2-i)*2], upconv_indices[(n_res-2-i)*2+1]))
        self.outc = conv_layer(chs[1], out_ch, neigh_orders=neigh_orders[0])
            
    def encode(self, x):
        for i in range(self.n_res):
            x = self.down[i](x)
        x = self.enc_fc(torch.flatten(x).unsqueeze(0))
        return x
        
    def decode(self, x): 
        x = self.dec_fc(x)
        x = torch.reshape(x, (self.n_vertex_last, -1))
        for i in range(self.n_res-1):
            x = self.up[i](x)
        x = self.outc(x)
        return x
        
    def forward(self, x):
        raise NotImplementedError('Not implemented') 

    
    
class SDF(nn.Module):
    """

    """
    def __init__(self, code_dim=256):
        super(SDF, self).__init__()
 
        self.chs = [code_dim+3, 512, 512, 512, 512, 512, 512, 512, 1]
        
        self.sdf_dec = nn.ModuleList([])
        for i in range(len(self.chs)-1):
            if i == 5:
                self.sdf_dec.append(nn.Sequential(
                    nn.Linear(self.chs[i]+code_dim+3, self.chs[i+1]),
                    nn.BatchNorm1d(self.chs[i+1], momentum=0.15, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    ))
            else:
                self.sdf_dec.append(nn.Sequential(
                    nn.Linear(self.chs[i], self.chs[i+1]),
                    nn.BatchNorm1d(self.chs[i+1], momentum=0.15, affine=True, track_running_stats=True),
                    nn.LeakyReLU(0.2, inplace=True),
                    ))
        self.last_tanh = nn.Tanh()
                 
    def forward(self, code, xyz):
        # code should be size 1x256, xyz should have size Nx3, N=batch_size
        res = torch.cat((torch.repeat_interleave(code, xyz.shape[0], dim=0), xyz), dim=0)  # Nx259
        print(res.shape)
        x = self.sdf_dec[0](res)
        print(x.shape)
        for i in range(1, 4):
            x = self.sdf_dec[i](x)
            print(x.shape)
        x = self.sdf_dec[4](torch.cat((x, res), dim=0))
        for i in range(5, len(self.chs)):
            x = self.sdf_dec[i](x)
            print(x.shape)
            
        return self.last_tanh(x)
        
 