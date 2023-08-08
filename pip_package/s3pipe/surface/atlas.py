#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:42:18 2020

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""

import numpy as np
import torch
from itertools import combinations  

from s3pipe.utils.vtk import write_vtk, read_vtk
from s3pipe.utils.utils import S3_normalize, get_par_fs_lookup_table


class HarmonizationSphere(torch.utils.data.Dataset):
    def __init__(self, files, scanner_ids, ages, n_vertex,
                  bcp_mean, bcp_std, map_mean, map_std, john_mean, john_std, ndar_mean, ndar_std):
        self.files =files
        self.ages = ages
        self.scanner_ids = scanner_ids
        self.n_vertex = n_vertex
        self.bcp_mean = bcp_mean
        self.bcp_std = bcp_std
        self.map_mean = map_mean
        self.map_std = map_std
        self.john_mean = john_mean 
        self.john_std = john_std
        self.ndar_mean = ndar_mean
        self.ndar_std = ndar_std
       
    def __getitem__(self, index):
        data = np.load(self.files[index])
        data = data[0:self.n_vertex]
        scanner_id = self.scanner_ids[index]
        
        """ data normalization: prior mean and std """
        age_idx = np.clip(self.ages[index], 0, 720)
        if scanner_id == 0:
            data_mean = self.bcp_mean[round(age_idx/30/3)]
            data_std = self.bcp_std[round(age_idx/30/3)]
        elif scanner_id == 1:
            data_mean = self.map_mean[round(age_idx/30/3)]
            data_std = self.map_std[round(age_idx/30/3)]
        elif scanner_id == 2:
            data_mean = self.john_mean[round(age_idx/360)]
            data_std = self.john_std[round(age_idx/360)]
        elif scanner_id == 3:
            tmp_age = age_idx/30/12.0
            if tmp_age < 270/30/12.0:
                data_mean = self.ndar_mean[0]
                data_std = self.ndar_std[0]
            else:
                data_mean = self.ndar_mean[int(np.round(tmp_age))]
                data_std = self.ndar_std[int(np.round(tmp_age))]
        else:
            raise NotImplementedError('error')
        data = (data - data_mean)/data_std
         
        # """ data normalization: prior max"""
        # data = data / 5.0
        
        return data[:, np.newaxis].astype(np.float32), scanner_id.astype(np.int64), self.files[index], self.ages[index]
    
    def __len__(self):
        return len(self.files)
    

class SingleBrainSphere(torch.utils.data.Dataset):
    """ return one surface with sulc+curv features 
    using default normalization method (SD: subtract median and standardize variance)
            
    """  
    def __init__(self, files, config):
        self.n_vertex = config['n_vertexs'][-1]
        self.val = config['val']
        self.files = files
        self.n_levels = config['n_levels']
        self.features = config['features']
        
    def __getitem__(self, index):
        file = self.files[index]
        if file.split('.')[-1] == 'vtk':
            surf = read_vtk(file)
            data = []
            for i in range(self.n_levels):
                data.append(surf[self.features[i]])
            data = np.array(data)
            data = np.transpose(data)
        elif file.split('.')[-1] == 'npy':
            data = np.load(file)
            if len(data.shape) == 1:
                data = data[:, np.newaxis]
        else:
            raise NotImplementedError('NotImplementedError')
        
        assert len(data.shape) == 2, 'len(data.shape) != 2'
        assert data.shape[0] > data.shape[1], 'Data shape should be NxC (163842x2) not CxN (2x163842).'
        for i in range(data.shape[1]):
            data[:,i] = S3_normalize(data[:,i]) 
        
        # if has age, this is specifically for BCP data with day age
        # age = float(file.split('/')[-2].split('_')[-1]) / 30.0
        # if age < 2:
        #     age = '01'
        # elif age < 4.5:
        #     age = '03'
        # elif age < 7.5:
        #     age = '06'
        # elif age < 10.5:
        #     age = '09'
        # elif age < 15:
        #     age = '12'
        # elif age < 21:
        #     age = '18'
        # elif age < 30:
        #     age = '24'
        # elif age < 42:
        #     age = '36'
        # elif age < 54:
        #     age = '48'
        # elif age < 66:
        #     age = '60'
        # else:
        #     age = '72'
     
        return data.astype(np.float32), file
        # return data.astype(np.float32), age, file

    def __len__(self):
        return len(self.files)



class LongitudinalRegAndParcSpheres(torch.utils.data.Dataset):
    """ return all surfaces of the same subject for longitudinal reg and parc
            
    """  
    def __init__(self, sub_file_dic, n_vertex=163842, num_long_scans=5):
        self.sub_file_dic = sub_file_dic
        self.n_vertex = n_vertex
        self.subjects = list(sub_file_dic.keys())
        self.num_long_scans = num_long_scans
        lookup_table_vec, self.lookup_table_scalar, lookup_table_name = get_par_fs_lookup_table()
        
    def __getitem__(self, index):
        subject = self.subjects[index]
        files = self.sub_file_dic[subject]
        age = [ float(x.split('/')[-2].split('_')[1]) for x in files ]  # BCP
        # age = [ float(x.split('/')[-2].split('_d')[-1]) for x in files ]  # OASIS
        sort_idx = np.argsort(np.asarray(age))
        files = [files[x] for x in sort_idx]
        
        data = np.zeros((len(files), self.n_vertex, 2))
        parc_label = np.zeros((len(files), self.n_vertex, 1))
        for file in files:
            tmp = np.load(file)
            sulc = S3_normalize(tmp[0:self.n_vertex, 0])
            curv = S3_normalize(tmp[0:self.n_vertex, 1])
            data[files.index(file)] = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
            
            # convert freesurfer scalar label to 0-35 label
            parc_label[files.index(file), :] = np.where(tmp[0:self.n_vertex, [2]] == self.lookup_table_scalar)[1][:, np.newaxis]
            
        if data.shape[0] < self.num_long_scans:
            data = np.concatenate((data, np.repeat(data[[-1],:,:], 
                                                   self.num_long_scans-data.shape[0], 
                                                   axis=0)), axis=0)
            parc_label = np.concatenate((parc_label, np.repeat(parc_label[[-1],:,:],
                                                               self.num_long_scans-parc_label.shape[0], 
                                                               axis=0)), axis=0)
        
        if data.shape[0] > self.num_long_scans:
            data = data[0:self.num_long_scans]
            parc_label = parc_label[0:self.num_long_scans]
            files= files[0:self.num_long_scans]
        
        return data.astype(np.float32), parc_label.astype(np.int64), files

    def __len__(self):
        return len(self.subjects)



class JointRegAndParcSphere(torch.utils.data.Dataset):

    def __init__(self, files, n_vertex):
        self.files = files
        self.n_vertex = n_vertex
        lookup_table_vec, self.lookup_table_scalar, lookup_table_name = get_par_fs_lookup_table()

    def __getitem__(self, index):
        file = self.files[index]
        tmp = np.load(file)
        sulc = S3_normalize(tmp[0:self.n_vertex, 0])
        curv = S3_normalize(tmp[0:self.n_vertex, 1])
        data = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
        
        # convert freesurfer scalar label to 0-35 label
        parc_label = np.where(tmp[0:self.n_vertex, [2]] == self.lookup_table_scalar)[1][:, np.newaxis]
        
        return data.astype(np.float32), parc_label.astype(np.int64), file

    def __len__(self):
        return len(self.files)
    



class BrainSphereForAtlasConstruction(torch.utils.data.Dataset):
    """ return the surface features for atlas construction,
     with prior max and min, and age information, which are needed
    for atlas construction.
    
    """

    def __init__(self, files, min_age, max_age, prior_sulc_min, 
                 prior_sulc_max, prior_curv_min, prior_curv_max, n_vertex):
        self.files = files
        # self.subs = subs
        self.min_age = min_age
        self.max_age = max_age
        self.prior_sulc_min = prior_sulc_min
        self.prior_sulc_max = prior_sulc_max
        self.prior_curv_min = prior_curv_min
        self.prior_curv_max = prior_curv_max
        self.n_vertex = n_vertex

    def __getitem__(self, index):
        file = self.files[index]
        data = np.load(file)
        sulc = S3_normalize(data[0:self.n_vertex, 0], norm_method='PriorMinMax', 
                         mi=self.prior_sulc_min, ma=self.prior_sulc_max)
        curv = S3_normalize(data[0:self.n_vertex, 1], norm_method='PriorMinMax',
                         mi=self.prior_curv_min, ma=self.prior_curv_max)
        data = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
        
        age = float(file.split('/')[-2].split('_')[1])
        age = (age - self.min_age) / (self.max_age - self.min_age)
        
        # sub = file.split('/')[-3].split('_')[0]
        # sub_id = self.subs.index(sub)
        # tmp = np.zeros(len(self.subs))
        # tmp[sub_id] = 1.0        
        
        return data.astype(np.float32), np.asarray(age).astype(np.float32), file #, tmp.astype(np.float32)

    def __len__(self):
        return len(self.files)



class IntraSubjectSpheres(torch.utils.data.Dataset):
    """ return single subject with all time points data, maximum=8 (batch size is 8)
            
    """  
    def __init__(self, files, subs, max_age=1200.0, prior_sulc_min=-12., 
                 prior_sulc_max=14, prior_curv_min=-1.3, prior_curv_max=1.0, n_vertex=40962):
        self.subs = subs
        self.max_age = max_age
        self.prior_sulc_min = prior_sulc_min
        self.prior_sulc_max = prior_sulc_max
        self.prior_curv_min = prior_curv_min
        self.prior_curv_max = prior_curv_max
        self.n_vertex = n_vertex
        
        self.sub_file_dic = {}
        for sub in subs:
            self.sub_file_dic[sub] = [ x for x in files if sub in x ]
        
    def __getitem__(self, index):
        sub = self.subs[index]
        files = self.sub_file_dic[sub]
        
        data = np.zeros((len(files),self.n_vertex,2))
        age = np.zeros((len(files),))
        for file in files:
            tmp = np.load(file)
            sulc = S3_normalize(tmp[0:self.n_vertex, 0], norm_method='PriorMinMax', mi=self.prior_sulc_min, ma=self.prior_sulc_max)
            curv = S3_normalize(tmp[0:self.n_vertex, 1], norm_method='PriorMinMax', mi=self.prior_curv_min, ma=self.prior_curv_max)
            data[files.index(file)] = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
         
            tmp = float(file.split('/')[-3].split('_')[1])
            age[files.index(file)] = tmp/self.max_age
        
        sub_id = np.zeros(len(self.subs))
        sub_id[index] = 1.0        
        
        return data.astype(np.float32), np.asarray(age).astype(np.float32), sub_id.astype(np.float32)

    def __len__(self):
        return len(self.subs)



class PairwiseSpheres(torch.utils.data.Dataset):
    """ return any two surfaces in the files
            
    """  
    def __init__(self, files, prior_sulc_min, prior_sulc_max, prior_curv_min, 
                 prior_curv_max, n_vertex, val=False):
        self.prior_sulc_min = prior_sulc_min
        self.prior_sulc_max = prior_sulc_max
        self.prior_curv_min = prior_curv_min
        self.prior_curv_max = prior_curv_max
        self.n_vertex = n_vertex
        self.val = val
        
        self.files = files
        self.comb = list(combinations(list(range(len(files))), 2))
        
        
    def __getitem__(self, index):
        ind = self.comb[index]
        files = [self.files[ind[0]], self.files[ind[1]]]
        
        data = np.zeros((len(files),self.n_vertex,2))
        for file in files:
            tmp = np.load(file)
            sulc = S3_normalize(tmp[0:self.n_vertex, 0], norm_method='PriorMinMax', mi=self.prior_sulc_min, ma=self.prior_sulc_max)
            curv = S3_normalize(tmp[0:self.n_vertex, 1], norm_method='PriorMinMax', mi=self.prior_curv_min, ma=self.prior_curv_max)
            data[files.index(file)] = np.concatenate((sulc[:,np.newaxis], curv[:,np.newaxis]), 1)
         
        if self.val:
            return data.astype(np.float32), files
        else:
            return data.astype(np.float32)

    def __len__(self):
        return len(self.comb)



###############################################################################

class DeformPool():
    """This class implements a buffer that stores previously generated
    deformation fields.

    This buffer enables us to update generation network using a history of generated 
    deformation fileds.
    """

    def __init__(self, pool_size, running_weight, n_vertex, device):
        """Initialize the DeformPool class

        Parameters:
            n_vertex (int) -- the number of vertices of this deformation field
            device
        """
        # assert pool_size > 0, "error"
        # assert type(pool_size) is int, "type error"
        
        self.pool_size = pool_size
        self.num_phis = 0
        self.mean = torch.zeros((n_vertex, 3), dtype=torch.float32, device=device)
        self.running_weight = running_weight

    def add(self, phi_3d):
        """Add the generated deformation filed to this buffer.

        Parameters
        ----------
        phi_3d : torch.tensor, shape [n_vertex, 3]
            the deformation field to be added.

        Return None

        """
        self.mean = self.mean.detach()
        
        # decay the running weight from 1 to stable running_weight (e.g., 0.1)
        if self.num_phis < self.pool_size:
            new_weight = -(1.-self.running_weight)/self.pool_size * self.num_phis + 1.
            self.mean = self.mean * (1.-new_weight) + phi_3d * new_weight
            self.num_phis = self.num_phis + 1.
        else:
            self.mean = self.mean * (1-self.running_weight) + phi_3d * self.running_weight
            self.num_phis = self.pool_size
        
    def get_mean(self):
        """Return the mean of deformation fileds from the pool.
        
        """
        return self.mean
    
    

def GenerateAtlas(model_gen, min_age, max_age, template, device, pool_mean=None):
    prior_sulc_min = -12.
    prior_sulc_max = 14.
    prior_curv_min = -1.3
    prior_curv_max = 1.0

    # For BCP infant
    ages = np.asarray([1, 45, 90, 120, 150, 180, 210, 240, 270, 300, 
                        330, 360, 390,420, 450, 480, 510, 540, 570, 600, 
                        630, 660, 690, 720])
    normal_ages = (ages - min_age) / max_age
    # For fetal
    # ages = np.arange(min_age, max_age+1)
    
    for i in range(len(normal_ages)):
        sucu = model_gen(torch.unsqueeze(torch.tensor([(normal_ages[i])], dtype=torch.float32, device=device), 1))
        surf = {'vertices': template['vertices'],
                'faces': template['faces'],
                'sulc': (sucu.detach().cpu().numpy()[:,0] + 1.)/2.* (prior_sulc_max-prior_sulc_min) + prior_sulc_min,
                'curv': (sucu.detach().cpu().numpy()[:,1] + 1.)/2.* (prior_curv_max-prior_curv_min) + prior_curv_min}
        if pool_mean is not None:
            surf['pool_mean'] = pool_mean.detach().cpu().numpy()*100.
        write_vtk(surf, '/media/ychenp/fq/AtlasConstructionUsingDL/generated_atlases/'+ \
                  'sulc_atlas_' + str(i) +'_('+ str(int(ages[i])) + ').vtk')
