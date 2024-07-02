#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""
import torch, os
import numpy as np
import math
import yaml


from s3pipe.utils.utils import get_neighs_order, get_upsample_order, S3_normalize,\
    get_par_fs_lookup_table, get_vertex_dis, get_neighs_faces, get_sphere_template
from s3pipe.utils.vtk import read_vtk
from s3pipe.utils.interp_torch import getEn, get_bi_inter, convert2DTo3D, diffeomorp_torch, \
    resampleStdSphereSurf_torch, resampleSphereSurf_torch, bilinearResampleSphereSurf_torch, \
        getOverlapIndex
from s3pipe.surface.atlas import DeformPool
from s3pipe.utils.interp_numpy import resampleSphereSurf, bilinearResampleSphereSurfImg
from s3pipe.models.models import SUnet

abspath = os.path.abspath(os.path.dirname(__file__))


###############################################################################
""" initial rigid align   """

def get_rot_mat_zyz(z1, y2, z3):
    """
    first z3, then y2, lastly z1
    """
    return np.array([[np.cos(z1) * np.cos(y2) * np.cos(z3) - np.sin(z1) * np.sin(z3), -np.cos(z1) * np.cos(y2) * np.sin(z3) - np.sin(z1) * np.cos(z3), np.cos(z1) * np.sin(y2)],
                     [np.cos(z1) * np.sin(z3) + np.sin(z1) * np.cos(y2) * np.cos(z3), -np.sin(z1) * np.cos(y2) * np.sin(z3) + np.cos(z1) * np.cos(z3), np.sin(z1) * np.sin(y2)],
                     [-np.sin(y2) * np.cos(z3), np.sin(y2) * np.sin(z3), np.cos(y2)]])

def get_rot_mat_zyx(z1, y2, x3):
    """
    first x3, then y2, lastly z1
    """
    return np.array([[np.cos(z1) * np.cos(y2),      np.cos(z1) * np.sin(y2) * np.sin(x3) - np.sin(z1) * np.cos(x3),        np.sin(z1) * np.sin(x3) + np.cos(z1) * np.cos(x3) * np.sin(y2)],
                     [np.cos(y2) * np.sin(z1),      np.cos(z1) * np.cos(x3) + np.sin(z1) * np.sin(y2) * np.sin(x3),        np.cos(x3) * np.sin(z1) * np.sin(y2) - np.cos(z1) * np.sin(x3)],
                     [-np.sin(y2),                  np.cos(y2) * np.sin(x3),                                               np.cos(y2) * np.cos(x3)]])

    
def initialRigidAlign(moving, fixed, 
                      SearchWidth=64/180*(np.pi), 
                      numIntervals=8, 
                      minSearchWidth=16/180*(np.pi),
                      moving_xyz=None,
                      bi=True, 
                      fixed_img=None,
                      metric='corr'):
    assert len(moving) == len(moving_xyz), "moving feature's size is not correct"
    radius = np.amax(moving_xyz[:,0])
    if bi == False:
        neigh_orders = None
        fixed_xyz = None
        raise NotImplementedError('Not implemented.')
    
    Center1 = 0.
    Center2 = 0.
    Center3 = 0.
    while SearchWidth >= minSearchWidth:
        print('Searching around [{:4.3f}, {:4.3f}, {:4.3f}] within SearchWidth {:4.3f}'.format(Center1, Center2, Center3, SearchWidth))
        rot_mats = np.mgrid[Center1-SearchWidth:Center1+SearchWidth+0.0001:2*SearchWidth/numIntervals,
                            Center2-SearchWidth:Center2+SearchWidth+0.0001:2*SearchWidth/numIntervals,
                            Center3-SearchWidth:Center3+SearchWidth+0.0001:2*SearchWidth/numIntervals].reshape(3,-1).T
        all_energy = np.zeros((rot_mats.shape[0], ), dtype=np.float64)
        for i in range(rot_mats.shape[0]):
            curr_rot = get_rot_mat_zyx(rot_mats[i, 0], rot_mats[i, 1], rot_mats[i, 2])
            curr_vertices = curr_rot.dot(np.transpose(moving_xyz))
            curr_vertices = np.transpose(curr_vertices)
            if bi:
                feat_inter = bilinearResampleSphereSurfImg(curr_vertices, fixed_img, radius=radius)
            else:
                feat_inter = resampleSphereSurf(fixed_xyz, curr_vertices, fixed, neigh_orders=neigh_orders)
            feat_inter = np.squeeze(feat_inter)
            if metric == 'corr':
                all_energy[i] = 1-(((feat_inter - feat_inter.mean()) * (moving - moving.mean())).mean() / feat_inter.std() / moving.std())
            elif metric == 'mse':
                all_energy[i] = np.mean((feat_inter - moving)**2)
            else:
                raise NotImplementedError('error')
            
            if (rot_mats[i] == np.array([0,0,0])).all():
                prev_energy = all_energy[i]
                
        best_center_idx = np.argmin(all_energy)
        Center1 = rot_mats[best_center_idx, 0]
        Center2 = rot_mats[best_center_idx, 1]
        Center3 = rot_mats[best_center_idx, 2]
        min_energy = all_energy[best_center_idx]
        print("Rotate by [{:4.3f}, {:4.3f}, {:4.3f}]. Minimal energy: {:4.3f}".format(Center1, Center2, Center3, min_energy))
           
        SearchWidth = SearchWidth/2.
        
    return np.array([Center1, Center2, Center3]), prev_energy, min_energy


###############################################################################


def readRegConfig(file="./regConfig.txt"):
    """

    Return registration configuration parameters
    -------
    None.

    """
    
    config = yaml.load(open(file, 'r'), Loader=yaml.Loader)

    return config


# load fixed/atlas surface
def get_fixed_xyz(n_vertex, device, radius=100.0):
    fixed_0 = get_sphere_template(n_vertex)
    fixed_xyz_0 = fixed_0['vertices']/radius
    fixed_xyz_0 = torch.from_numpy(fixed_xyz_0.astype(np.float32)).to(device)
    return fixed_xyz_0


def load_atlas(atlas_file, n_vertex, config):
    atlas_tmp = read_vtk(atlas_file)
    faces = get_sphere_template(n_vertex)
    par_fs_vec = atlas_tmp['par_FS_vec'][0:n_vertex,:]
    lookup_table_vec, lookup_table_scalar, lookup_table_name = get_par_fs_lookup_table()
    par_fs_scalar = np.zeros((len(par_fs_vec),))
    for p in range(len(par_fs_vec)):
        par_fs_scalar[p] = lookup_table_scalar[np.where(np.all(par_fs_vec[p] == lookup_table_vec, axis=1))[0][0]]
    par_fs_scalar = torch.from_numpy(par_fs_scalar.astype(np.int64)).to(config['device'])
    
    atlas = {'vertices': atlas_tmp['vertices'][0:n_vertex, :],
             'faces': faces['faces'],
              'sulc': torch.from_numpy(S3_normalize(atlas_tmp['Convexity'])[0:n_vertex]).to(config['device']),
              'curv': torch.from_numpy(S3_normalize(atlas_tmp['curvature'])[0:n_vertex]).to(config['device']),
              'par_fs_vec': par_fs_vec,
              'par_fs_scalar': par_fs_scalar,
              'par_fs_0_35': torch.from_numpy(np.where(par_fs_scalar.cpu().numpy()[:, np.newaxis] \
                                                       == lookup_table_scalar)[1].astype(np.int64)).to(config['device'])}
    
    # data size should meet the requirement B x C x N
    # for i in range(len(config['n_vertexs'])):
    #     atlas['data'+str(i)] = torch.from_numpy(S3_normalize(atlas_tmp[config['features'][i]]).astype(np.float32)).to(config['device'])

    return atlas


def initModel(i_level, config):
    model_dir = config['model_dir']
    val = config['val']
    
    level = config['levels'][i_level]
    n_res = level-2 if level<6 else 4
    
    model_0 = SUnet(in_ch=config['in_ch'], out_ch=config['out_ch'], level=level, n_res=n_res, rotated=0, complex_chs=16)
    model_0.to(config['device'])
    
    model_1 = SUnet(in_ch=config['in_ch'], out_ch=config['out_ch'], level=level, n_res=n_res, rotated=1, complex_chs=16)
    model_1.to(config['device'])
    
    model_2 = SUnet(in_ch=config['in_ch'], out_ch=config['out_ch'], level=level, n_res=n_res, rotated=2, complex_chs=16)
    model_2.to(config['device'])
    
    optimizer_0 = torch.optim.Adam(model_0.parameters())
    optimizer_1 = torch.optim.Adam(model_1.parameters())      
    optimizer_2 = torch.optim.Adam(model_2.parameters())

    if val or model_dir is not None:
        print('Loading pretrained models...')
        model_0.load_state_dict(torch.load(os.path.join(model_dir, 'S3Reg_' + \
                                                        config['hemi'] + '_' + config['features'][i_level]+ \
                                                        "_" + str(config['n_vertexs'][i_level])+ '_smooth_' + \
                                                            str(float(config['weight_smooth'][i_level])) + "_pretrained_0.mdl")))
        model_1.load_state_dict(torch.load(os.path.join(model_dir, 'S3Reg_' + \
                                                        config['hemi'] + '_' +    config['features'][i_level]+ \
                                                        "_" + str(config['n_vertexs'][i_level])+ '_smooth_' + \
                                                            str(float(config['weight_smooth'][i_level])) +  "_pretrained_1.mdl")))
        model_2.load_state_dict(torch.load(os.path.join(model_dir, 'S3Reg_' + \
                                                        config['hemi'] + '_' +  config['features'][i_level]+ \
                                                        "_" + str(config['n_vertexs'][i_level])+ '_smooth_' + \
                                                            str(float(config['weight_smooth'][i_level])) +  "_pretrained_2.mdl")))
        return [model_0, model_1, model_2], [optimizer_0, optimizer_1, optimizer_2]
    
    else:
        return [model_0, model_1, model_2], [optimizer_0, optimizer_1, optimizer_2]
    

def createRegConfig(config):
    """ load or generate precomputed parameters for registration
    
    """
    ns_vertex = np.array([12,42,162,642,2562,10242,40962,163842])
    n_vertexs = []
    for i_level in range(len(config['levels'])):
        n_vertexs.append(ns_vertex[config['levels'][i_level]-1])
    n_levels = len(n_vertexs)
    config['n_vertexs'] = n_vertexs
    config['n_levels'] = n_levels
    
    
    # initialize model input and output channels
    config['in_ch'] = 2 * len(config['features']) # in case multiple channels
    config['out_ch'] = 2  # two components for tangent plane deformation vector 


    # initialize models and optimizers 
    if 'val' not in config:
        config['val'] = False
    if 'model_dir' not in config:
        config['model_dir'] = None
    modelss = []
    optimizerss = []
    for n_vertex in config['n_vertexs']:
        tmp = initModel(config['n_vertexs'].index(n_vertex), config)
        modelss.append(tmp[0])
        optimizerss.append(tmp[1])
    config['modelss'] = modelss
    config['optimizerss'] = optimizerss


    device = config['device']
    
    fixed_xyz_0 = get_fixed_xyz(n_vertexs[-1], device)
    config['fixed_xyz_0'] = fixed_xyz_0
    
    neigh_orders = []
    for n_vertex in n_vertexs:
        neigh_orders.append(get_neighs_order(n_vertex))
    config['neigh_orders'] = neigh_orders
    
    neigh_faces = []
    for n_vertex in n_vertexs:
        neigh_faces.append(get_neighs_faces(n_vertex))
    config['neigh_faces'] = neigh_faces
    
    Ens = []
    for n_vertex in n_vertexs:
        Ens.append(getEn(int(n_vertex), device))
    config['Ens'] = Ens
    
    merge_indexs = []
    for n_vertex in n_vertexs:
        merge_indexs.append(getOverlapIndex(n_vertex, device))
    config['merge_indexs'] = merge_indexs
    
    bi_inter_0s = []
    for n_vertex in n_vertexs:
        bi_inter_0s.append(get_bi_inter(n_vertex, device)[0])  # only need [0] for interpolation
    config['bi_inter_0s'] = bi_inter_0s
  
    upsample_neighborss = []
    for n_vertex in n_vertexs:
        upsample_neighborss.append(get_upsample_order(n_vertex))
    config['upsample_neighborss'] = upsample_neighborss
    
    
    grad_filter = torch.ones((7, 1), dtype=torch.float32, device = device)
    grad_filter[6] = -6    
    config['grad_filter'] = grad_filter
        
    # for centralize deformations in groupwise registration
    if 'pool_size' in config:
        deform_pools = []
        for n_vertex in n_vertexs:
            deform_pools.append(DeformPool(config['pool_size'], config['running_weight'], n_vertex, device))
        config['deform_pools'] = deform_pools
      
    # sulc_std = 4.7282835 
    # sulc_mean = 0.31142648
    # sulc curv prior for atlas construction, otherwise, norm_method = SD
    config['norm_method'] = 'PrioiMaxMin'  
    config['prior_sulc_min'] =  -12.
    config['prior_sulc_max'] = 14.
    config['prior_curv_min'] = -1.3
    config['prior_curv_max'] = 1.0
    
    # read default atlas
    config['atlas'] = load_atlas(config['atlas_file'], n_vertexs[-1], config)
    
    return config


###############################################################################
"""  S3Reg framework """

def LossSimCorr(fixed_inter, moving, moving_inter, fixed, sucu=False):
    """
    Compute only correlation loss for longitudinal surfaces
    """
    if sucu:
        sulc_loss_corr_1 = 1 - ((fixed_inter[:,0] - fixed_inter[:,0].mean()) * \
                              (moving[:,0] - moving[:,0].mean())).mean() / fixed_inter[:,0].std() / moving[:,0].std()
        sulc_loss_corr_2 = 1 - ((moving_inter[:,0] - moving_inter[:,0].mean()) * \
                              (fixed[:,0] - fixed[:,0].mean())).mean() / moving_inter[:,0].std() / fixed[:,0].std()
        sulc_loss_corr = (sulc_loss_corr_1 + sulc_loss_corr_2)/2
        print("sulc corr: {:.4f}".format(1-sulc_loss_corr.item()))
        
        curv_loss_corr_1 = 1 - ((fixed_inter[:,1] - fixed_inter[:,1].mean()) * \
                               (moving[:,1] - moving[:,1].mean())).mean() / fixed_inter[:,1].std() / moving[:,1].std()
        curv_loss_corr_2 = 1 - ((moving_inter[:,1] - moving_inter[:,1].mean()) * \
                               (fixed[:,1] - fixed[:,1].mean())).mean() / moving_inter[:,1].std() / fixed[:,1].std()
        curv_loss_corr = (curv_loss_corr_1 + curv_loss_corr_2)/2
        print("curv corr: {:.4f}".format(1-curv_loss_corr.item()))
        
        loss_corr = sulc_loss_corr * 0.5 + curv_loss_corr * 0.5
    else:
        assert fixed_inter.shape[1] == 1 or len(fixed_inter.shape) == 1, 'error'
        loss_corr_1 = 1 - ((fixed_inter - fixed_inter.mean()) * (moving - moving.mean())).mean() / fixed_inter.std() / moving.std()
        loss_corr_2 = 1 - ((moving_inter - moving_inter.mean()) * (fixed - fixed.mean())).mean() / moving_inter.std() / fixed.std()
        loss_corr = (loss_corr_1 + loss_corr_2)/2
        print("corr: {:.4f}".format(1-loss_corr.item()))
        
    return loss_corr


def LossSim(fixed_inter, moving, moving_inter, fixed, chs=1):
    """
    

    Parameters
    ----------
    fixed_inter : TYPE
        fixed sphere's features interpolated at the points where 
        the moving sphere's xyz is warped using deformation field.
    moving : TYPE
        moving sphere's features.
    moving_inter : TYPE
        After warping the moving sphere's xyz and feature, the features interpolated
        at the fixed sphere's xyz
    fixed : TYPE
        fixed sphere's features..
    sucu : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    loss_corr : TYPE
        DESCRIPTION.
    loss_l2 : TYPE
        DESCRIPTION.

    """
    if chs==2:
        sulc_loss_corr_1 = 1 - ((fixed_inter[:,0] - fixed_inter[:,0].mean()) * \
                              (moving[:,0] - moving[:,0].mean())).mean() / fixed_inter[:,0].std() / moving[:,0].std()
        sulc_loss_corr_2 = 1 - ((moving_inter[:,0] - moving_inter[:,0].mean()) * \
                              (fixed[:,0] - fixed[:,0].mean())).mean() / moving_inter[:,0].std() / fixed[:,0].std()
        sulc_loss_corr = (sulc_loss_corr_1 + sulc_loss_corr_2)/2
        
        sulc_loss_l2_1 = torch.mean((fixed_inter[:,0] - moving[:,0])**2)
        sulc_loss_l2_2 = torch.mean((moving_inter[:,0] - fixed[:,0])**2)
        sulc_loss_l2 = (sulc_loss_l2_1 + sulc_loss_l2_2)/2
        # print("sulc corr and l2: ", sulc_loss_corr.item(), sulc_loss_l2.item())
        
        
        curv_loss_corr_1 = 1 - ((fixed_inter[:,1] - fixed_inter[:,1].mean()) * \
                               (moving[:,1] - moving[:,1].mean())).mean() / fixed_inter[:,1].std() / moving[:,1].std()
        curv_loss_corr_2 = 1 - ((moving_inter[:,1] - moving_inter[:,1].mean()) * \
                               (fixed[:,1] - fixed[:,1].mean())).mean() / moving_inter[:,1].std() / fixed[:,1].std()
        curv_loss_corr = (curv_loss_corr_1 + curv_loss_corr_2)/2
            
        curv_loss_l2_1 = torch.mean((fixed_inter[:,1] - moving[:,1])**2)
        curv_loss_l2_2 = torch.mean((moving_inter[:,1] - fixed[:,1])**2)
        curv_loss_l2 = (curv_loss_l2_1 + curv_loss_l2_2)/2
        # print("curv corr and l2: ", curv_loss_corr.item(), curv_loss_l2.item())
        
        loss_corr = sulc_loss_corr * 0.7 + curv_loss_corr * 0.3
        loss_l2 = sulc_loss_l2 * 0.7 + curv_loss_l2 * 0.3
    else:
        loss_corr_1 = 1 - ((fixed_inter - fixed_inter.mean()) * (moving - moving.mean())).mean() / fixed_inter.std() / moving.std()
        loss_corr_2 = 1 - ((moving_inter - moving_inter.mean()) * (fixed - fixed.mean())).mean() / moving_inter.std() / fixed.std()
        loss_corr = (loss_corr_1 + loss_corr_2)/2
        
        loss_l2_1 = torch.mean((fixed_inter - moving)**2)
        loss_l2_2 = torch.mean((moving_inter - fixed)**2)
        loss_l2 = (loss_l2_1 + loss_l2_2)/2
        
    return loss_corr, loss_l2



def LossPhi(phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, phi_3d_2_orig, 
            phi_3d_0_to_2, phi_3d_orig, merge_index, neigh_orders, n_vertex, 
            grad_filter):
    loss_phi_consistency = torch.mean(torch.abs(phi_3d_0_to_1[merge_index[7]] - phi_3d_1_orig[merge_index[7]])) + \
                           torch.mean(torch.abs(phi_3d_1_to_2[merge_index[8]] - phi_3d_2_orig[merge_index[8]])) + \
                           torch.mean(torch.abs(phi_3d_0_to_2[merge_index[9]] - phi_3d_2_orig[merge_index[9]]))
    loss_smooth = torch.abs(torch.mm(phi_3d_orig[0:n_vertex,0][neigh_orders], grad_filter)) + \
                  torch.abs(torch.mm(phi_3d_orig[0:n_vertex,1][neigh_orders], grad_filter)) + \
                  torch.abs(torch.mm(phi_3d_orig[0:n_vertex,2][neigh_orders], grad_filter))
    loss_smooth = torch.mean(loss_smooth)
    return loss_phi_consistency, loss_smooth


def S3Register(moving, fixed, 
               model_0, model_1, model_2,
               merge_index, En, device, 
               bi_inter_0, fixed_xyz_0,
               diffe=True, bi=True, 
               num_composition=6, val=False, truncated=False):
    """
    Register moving surface to fixed surface. The return deformation field is 
    to warp moving to fixed. 

    Parameters
    ----------
    moving : torch tensor (N*1)
        moving surface.
    fixed :  torch tensor (N*1)
        fixed surface.
    model_0 : TYPE
        DESCRIPTION.
    model_1 : TYPE
        DESCRIPTION.
    model_2 : TYPE
        DESCRIPTION.
    merge_index : TYPE
        DESCRIPTION.
    En : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    bi_inter_0 : TYPE
        DESCRIPTION.
    fixed_xyz_0 : TYPE
        DESCRIPTION.
    diffe : TYPE, optional
        DESCRIPTION. The default is True.
    bi : TYPE, optional
        DESCRIPTION. The default is True.
    num_composition : TYPE, optional
        DESCRIPTION. The default is 6.
    val : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    rot_mat_01, rot_mat_12, rot_mat_02, rot_mat_20, z_weight_0, \
            z_weight_1, z_weight_2, index_01, index_12, index_02, index_0_0, \
              index_1_0, index_2_0, index_double_02, index_double_12, index_double_01, \
               index_triple_computed = merge_index
    En_0, En_1, En_2 = En
    
    # registraion starts
    data = torch.cat((moving, fixed), 1)
    
    # tangent vector field phi
    phi_2d_0_orig = model_0(data)/50.0
    phi_2d_1_orig = model_1(data)/50.0
    phi_2d_2_orig = model_2(data)/50.0
    
    phi_3d_0_orig = convert2DTo3D(phi_2d_0_orig.permute(0, 2, 1).squeeze(), En_0, device)
    phi_3d_1_orig = convert2DTo3D(phi_2d_1_orig.permute(0, 2, 1).squeeze(), En_1, device)
    phi_3d_2_orig = convert2DTo3D(phi_2d_2_orig.permute(0, 2, 1).squeeze(), En_2, device)
    
    """ deformation consistency  """
    phi_3d_0_to_1 = torch.mm(rot_mat_01, torch.transpose(phi_3d_0_orig, 0, 1))
    phi_3d_0_to_1 = torch.transpose(phi_3d_0_to_1, 0, 1)
    phi_3d_1_to_2 = torch.mm(rot_mat_12, torch.transpose(phi_3d_1_orig, 0, 1))
    phi_3d_1_to_2 = torch.transpose(phi_3d_1_to_2, 0, 1)
    phi_3d_0_to_2 = torch.mm(rot_mat_02, torch.transpose(phi_3d_0_orig, 0, 1))
    phi_3d_0_to_2 = torch.transpose(phi_3d_0_to_2, 0, 1)
    
    """ first merge """
    phi_3d = torch.zeros(len(En_0), 3).to(device)
    phi_3d[index_double_02] = (phi_3d_0_to_2[index_double_02] + phi_3d_2_orig[index_double_02])/2.0
    phi_3d[index_double_12] = (phi_3d_1_to_2[index_double_12] + phi_3d_2_orig[index_double_12])/2.0
    tmp = (phi_3d_0_to_1[index_double_01] + phi_3d_1_orig[index_double_01])/2.0
    phi_3d[index_double_01] = torch.transpose(torch.mm(rot_mat_12, torch.transpose(tmp,0,1)), 0, 1)
    phi_3d[index_triple_computed] = (phi_3d_1_to_2[index_triple_computed] + \
                                     phi_3d_2_orig[index_triple_computed] + \
                                     phi_3d_0_to_2[index_triple_computed])/3.0
    phi_3d_orig = torch.transpose(torch.mm(rot_mat_20, torch.transpose(phi_3d,0,1)),0,1)
    # print("torch.linalg.norm(phi_3d_orig,dim=1).max().item() ", torch.linalg.norm(phi_3d_orig,dim=1).max().item())
    phi_3d_orig_diffe =  phi_3d_orig
    
    if truncated:
        max_disp = get_vertex_dis(moving.shape[0])/100.0*3.0
        tmp = torch.linalg.norm(phi_3d_orig, dim=1) > max_disp
        phi_3d_orig_diffe = phi_3d_orig.clone()
        phi_3d_orig_diffe[tmp] = phi_3d_orig[tmp] / (torch.linalg.norm(phi_3d_orig[tmp], dim=1, keepdim=True).repeat(1,3)) * max_disp
    
    if diffe:
        """ diffeomorphism  """
        # divide to small veloctiy field
        phi_3d = phi_3d_orig_diffe/math.pow(2,num_composition)
        moving_warp_phi_3d = diffeomorp_torch(fixed_xyz_0, phi_3d, 
                                              num_composition=num_composition, 
                                              bi=bi, bi_inter=bi_inter_0, 
                                              device=device)
    else:
        """ Non diffeomorphism  """
        # print(torch.linalg.norm(phi_3d_orig,dim=1).max().item())
        moving_warp_phi_3d = fixed_xyz_0 + phi_3d_orig
        moving_warp_phi_3d = moving_warp_phi_3d/(torch.linalg.norm(moving_warp_phi_3d, dim=1, keepdim=True).repeat(1,3))
    
    if val:
        return phi_3d_orig, moving_warp_phi_3d	
    else:
        return moving_warp_phi_3d, phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, \
                phi_3d_2_orig, phi_3d_0_to_2, phi_3d_orig
    

def regOnSingleLevel(i_level, moving_0, config, total_deform=None):
    """
    single-level multi-modal registration framework. 

    Parameters
    ----------
    i_level : int, the i-th level current registration process is performed on,
    begins from 0, generally contains 3 levels, i.e., i_level = 0 or 1 or 2
    moving_0 : torch tensor N*n_levels, where 0 channel (e.g., sulc) is the 
    feature for 0-level reg, 1st channel (e.g., curv) for 1-th level reg
    config : dictionary
        registration configuration file.
    total_deform : record the total deform from the begining of the multi-level registration,
        current model is trained based on the surface after warping using current 
        total deform
    val : bool, train or test? The default is False.

    Returns
    -------
    TYPE
        current total deformation after registration on this level.

    """
    
    # load configuration parameters
    val = config['val']
    device = config['device']
    n_vertexs = config['n_vertexs']
    n_vertex = n_vertexs[i_level]
    fixed_xyz = config['fixed_xyz_0'][0:n_vertex, :]
    neigh_orders = config['neigh_orders'][i_level]
   
    # load moving and fixed features
    moving = moving_0[0, 0:n_vertex, i_level:i_level+1]
    fixed = config['atlas']['data'+str(i_level)][0:n_vertex].unsqueeze(1)
    
    # print('moving.shape:', moving.shape)    
    # print('fixed.shape:', fixed.shape)
    
    # warp previous deformation
    if i_level != 0:
        # if current vertices is one level higher than previous vertices
        if n_vertex == n_vertexs[i_level-1]* 4 - 6:
            upsample_neighbors = config['upsample_neighborss'][i_level]
            total_deform_curr = resampleStdSphereSurf_torch(n_vertexs[i_level-1],
                                                            n_vertex, 
                                                            total_deform, 
                                                            upsample_neighbors, 
                                                            device)
            total_deform_curr = total_deform_curr / \
                (torch.linalg.norm(total_deform_curr, dim=1, keepdim=True).repeat(1,3))
        # else if current vertices is the same with previous vertices
        elif n_vertex == n_vertexs[i_level-1]:
            total_deform_curr = total_deform
        else:
            raise NotImplementedError('NotImplementedError')
        moving = resampleSphereSurf_torch(total_deform_curr, 
                                          fixed_xyz, moving, 
                                          device=device)
        
        
    # start registration using S3Reg 
    model_0, model_1, model_2 = config['modelss'][i_level]
    merge_index = config['merge_indexs'][i_level]
    bi = config['bi']
    bi_inter_0 = config['bi_inter_0s'][i_level]
    num_composition = 6
    if config['diffe']:	
        num_composition = config['num_composition']
    if 'truncated' in config:
        truncated = config['truncated'][i_level]
    else:
        truncated = False
    if val:
       phi_3d_orig, moving_warp_phi_3d = S3Register(moving.unsqueeze(0).permute(0,2,1), 
                                                    fixed.unsqueeze(0).permute(0,2,1),
                                                    model_0, model_1, model_2,
                                                    merge_index,  
                                                    config['Ens'][i_level], device, 
                                                    bi_inter_0, fixed_xyz,
                                                    diffe=config['diffe'], bi=bi, 
                                                    num_composition=num_composition,
                                                    val=val,                                                               
                                                    truncated=truncated)
    else:
        moving_warp_phi_3d, phi_3d_0_to_1, phi_3d_1_orig, phi_3d_1_to_2, \
            phi_3d_2_orig, phi_3d_0_to_2, phi_3d_orig\
                                        = S3Register(moving.unsqueeze(0).permute(0,2,1), 
                                                     fixed.unsqueeze(0).permute(0,2,1),
                                                     model_0, model_1, model_2,
                                                     merge_index,  
                                                     config['Ens'][i_level], device, 
                                                     bi_inter_0, fixed_xyz,
                                                     diffe=config['diffe'], bi=bi, 
                                                     num_composition=num_composition,
                                                     val=val,                                                               
                                                     truncated=truncated)
        
    # combine with previous total deformation
    if i_level != 0:	
        total_deform = resampleSphereSurf_torch(fixed_xyz, total_deform_curr,
                                                moving_warp_phi_3d, device=device)
    else:	
        total_deform = moving_warp_phi_3d
        
        
    # if truncated:
    #     if i_level != 0:
    #         velocity = 1.0/torch.sum(fixed_xyz * total_deform, 1).unsqueeze(1) * \
    #             total_deform - fixed_xyz
    #         max_disp = get_vertex_dis(n_vertex)/100.0*4.0
    #         tmp = torch.linalg.norm(velocity, dim=1) > max_disp
    #         velocity_clone = velocity.clone()
    #         velocity_clone[tmp] = velocity[tmp] / (torch.linalg.norm(velocity[tmp], dim=1, keepdim=True).repeat(1,3)) * max_disp
    #         # TODO, now is non diffemorphic, because the veocity is derived from 
    #         # diffemorphic, so I assume even non diffemorphic implementation here 
    #         # will not result in intersection, need further check
    #         total_deform = fixed_xyz + velocity_clone
    #         total_deform = total_deform/(torch.linalg.norm(total_deform, dim=1, keepdim=True).repeat(1,3))
    
    
    # return deformation field and losses
    if val:
        return total_deform
    else:
        moving_inter = resampleSphereSurf_torch(moving_warp_phi_3d, fixed_xyz,
                                                moving, device=device)
        if bi:
            fixed_inter = bilinearResampleSphereSurf_torch(moving_warp_phi_3d, 
                                                           fixed, bi_inter=bi_inter_0)
        else:
            fixed_inter = resampleSphereSurf_torch(fixed_xyz, moving_warp_phi_3d,
                                                   fixed, device=device)
          
        # deformation centrality loss
        if config['centra']:
            # if move moving surface, need to conmpute negtive deformation first
            neg_moved = resampleSphereSurf_torch(moving_warp_phi_3d, fixed_xyz, fixed_xyz,
                                                 device=device)
            neg_phi = 1.0/torch.sum(fixed_xyz * neg_moved, 1).unsqueeze(1) * \
                    neg_moved - fixed_xyz  # project to tangent plane
            deform_pool = config['deform_pools'][i_level]
            deform_pool.add(neg_phi)
            pool_mean = deform_pool.get_mean()
            print(torch.linalg.norm(pool_mean,dim=1).max().item())
            loss_centra = torch.mean(torch.abs(pool_mean))
        else:
            loss_centra = torch.tensor([0]).to(device)
      
        loss_corr, loss_l2 = LossSim(fixed_inter, moving, moving_inter, fixed)
        loss_phi_consistency, loss_smooth = LossPhi(phi_3d_0_to_1, phi_3d_1_orig,
                                                    phi_3d_1_to_2, phi_3d_2_orig,
                                                    phi_3d_0_to_2, phi_3d_orig,
                                                    merge_index, neigh_orders,
                                                    n_vertex, config['grad_filter'])
        
        loss =  config['weight_l2'][i_level] * loss_l2 +  \
                config['weight_corr'][i_level] * loss_corr + \
                config['weight_smooth'][i_level] * loss_smooth + \
                config['weight_phi_consis'][i_level] * loss_phi_consistency + \
                config['weight_centra'][i_level] * loss_centra

        return total_deform, loss, loss_centra.item(), loss_corr.item(), \
                loss_l2.item(), loss_phi_consistency.item(), loss_smooth.item()

