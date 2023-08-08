#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 01:44:25 2021

2023.3.3 add J_d optimization method and sulc computation code

@author: Fenqiang Zhao
"""

import os
import numpy as np
from .prop import computeVertexNormal, computeVertexEdgeLength, computeFaceArea, \
    computeFaceSign
from .s3map import projectOntoSphere

abspath = os.path.abspath(os.path.dirname(__file__))


def InflateSurface(surf, params):
    """
    faces shape is Nx3
    
    using the eq. 8 in fischl et al. cortical surface-based analysiss II ,
    stop either reaching max_iter_num or projectionOntoSphere with no negative triangles,
    
    params:
    max_iter_num,
    labda, 1-lambda for J_d metric preserving term, currently set to 0 for faster inflation since it will be minimized on sphere faster later
    save_sulc=1, save sulc or not, not saving sulc will inflate the surface faster
    n_averages, iteration number for averaging gradients
    max_grad = 1.0, maximum displacement for vertices at each iteration, set to 
    1 mm as in freesurfer, https://github.com/freesurfer/freesurfer/blob/c6bb23bc63d06486bd84740c2e021239707b6c20/utils/mrisurf.c#L4288

    Time cost (num_vertices = 300000):    
    computeVertexNormal 0.15 s
    computeSpringTerm 0.10 s
    momentumTimeStep 0.01 s
    Compute sulc 0.05 s
    Project to sphere and computeFaceSign 0.08 s
    Total: 0.3s/1 iter, 30s/100 iter, 60s/200iter (generally enough), 150s/500 iter
    
    """
    print('inflation_params:')
    for k, v in params.items():
        print (k, ':', v)
        
    max_iter_num = params['max_iter_num']
    labda =  params['lambda']
    save_sulc = params['save_sulc']
    # n_averages = params['n_averages']
    max_grad = params['max_grad']
    min_proj_iter = params['min_proj_iter']
    
    surf.centralize()
    surf.updateVerticesAppend()
    
    # surf.orig_metrics = computeVertexEdgeLength(surf)  
    # use relative metrics
    # surf.orig_metrics = surf.orig_metrics / surf.orig_metrics.sum() * (surf.num_vertices*0.8*2.5) * 2 # average edge length equals to 0.8 (mm)
    # print('average edge length:', surf.orig_metrics[surf.orig_metrics!=0].mean())
    
    surf.orig_face_area = computeFaceArea(surf.vertices, surf.faces)
    print('\ntotal surface area = {:.0f} mm^2'.format(surf.orig_face_area.sum()))
    
    dist_scale = 1.0
    surf.clearMomentum()
    if save_sulc:
        surf.sulc = np.zeros((surf.num_vertices,), dtype=np.float64)
    for i in range(max_iter_num):
        if save_sulc:
            surf.vertex_normal = computeVertexNormal(surf)
        surf.clearGradient()
        surf.updateVerticesAppend()
        # computeDistanceTerm(surf, 1-labda)
        computeSpringTerm(surf, labda)
        # surf.averageGradient(n_averages)  # this may be important to achieve very inflated/fat surface
        max_grad_mag = surf.momentumTimeStep(max_grad)
        if i % 5 == 0:
            print('Iteration {}/{}, max grad mag: {:.4f}'.format(i, max_iter_num, max_grad_mag))
        surf.applyGradient()
        
        if save_sulc and i < 110:
            surf.sulc += ((surf.gradient * surf.vertex_normal).sum(1) / dist_scale)
        tmp_scale = surf.scaleBrainBasedOnArea()   # to compare metrics fairly
        dist_scale *= tmp_scale   # to reflect original sulci/gyri depth
        
        if i > min_proj_iter and i % 20 == 0:
            sphere_0_ver = projectOntoSphere(surf.vertices)
            area_sign = computeFaceSign(sphere_0_ver, surf.faces)
            num_neg_area = int((np.abs(area_sign) - area_sign).sum() / 2)
            if num_neg_area < params['min_neg_area_num']:
                print("Negative areas of current sphere: {}. Inflation stops because negative triangles found on sphere are less than threshold.".format(num_neg_area))
                break
            else:
                print("Negative areas of current sphere: {}. Continue inflation.".format(num_neg_area))
                 
                
# need to work more on the parameters' fine-tuning to reproduce FreeSurfer's results
def computeDistanceTerm(surf, labda):
    # J_d, metric preserving term
    e_kn = surf.vertices_append[surf.neigh_vertex_mat] - \
        np.repeat(surf.vertices[:,np.newaxis,:], surf.MAX_NUM_NEIGHBORS, axis=1)
    surf.curr_metrics = np.linalg.norm(e_kn, axis=2)
    e_kn_normal = e_kn / surf.curr_metrics[:, :, np.newaxis]
    
    # compute relative metrics
    # surf.curr_metrics = surf.setZeroNeighs(surf.curr_metrics)
    # surf.curr_metrics = surf.curr_metrics / surf.curr_metrics.sum() * (surf.num_vertices*0.8*2.5) * 2 # average edge length equals to 0.8 (mm)

    mag = surf.curr_metrics - surf.orig_metrics
    mag = surf.setZeroNeighs(mag)
    tmp = (mag[:,:, np.newaxis] * e_kn_normal).sum(1)
    print( "J_d: ", np.mean(np.linalg.norm(tmp, axis=1)))
    surf.gradient += tmp * labda


# since numba does not support 2d array indexing, implement this function outside numba jit class Surface
def computeSpringTerm(surf, labda):
    spring_term_grad = (surf.vertices_append[surf.neigh_vertex_mat].sum(1) - \
        surf.vertices * surf.n_neighs_per_vertex)/4.0
    # print('Spring term: ', np.mean(np.linalg.norm(spring_term_grad, axis=1)))
    surf.gradient += spring_term_grad * labda # use freesurfer's hyper-parameter from https://github.com/freesurfer/freesurfer/blob/c6bb23bc63d06486bd84740c2e021239707b6c20/utils/mrisurf.c#L7674
    

