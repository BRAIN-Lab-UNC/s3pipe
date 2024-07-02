#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 00:32:35 2022

@author: Fenqiang Zhao
"""
import os
import numpy as np
from functools import reduce
from s3pipe.utils.utils import get_neighs_order


abspath = os.path.abspath(os.path.dirname(__file__))


def smooth_surface_map(vertices, feat, num_iter, neigh_orders=None):
    """
    smooth surface maps
    
    vertices: N*3 numpy array, surface vertices
    neigh_orders: N*7 numpy array, type: np.int32, 
    feat: [N, C], numpy array, the surface map to be smoothed
    num_iter: numbers of smooth operation
    """
    assert vertices.shape[0] == feat.shape[0], "vertices number is different from feature number"
    assert vertices.shape[0] in [42,162,642,2562,10242,40962,163842], "this function only support icosahedron discretized spheres"
    
    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    
    if neigh_orders is None:
        neigh_orders = get_neighs_order(feat.shape[0])
    assert neigh_orders.shape[0] == vertices.shape[0], "neighbor_orders size is not right"      
        
    smooth_kernel = np.array([1/12, 1/12, 1/12, 1/12, 1/12, 1/12, 1/2])
    smooth_kernel = np.repeat(smooth_kernel[:, np.newaxis], feat.shape[1], axis=1)
    for i in range(num_iter):
        tmp = np.multiply(feat[neigh_orders], smooth_kernel[np.newaxis,:])
        feat = np.sum(tmp, axis=1)
    
    return feat


def computeVertexNormal(surf):
    """
    follow freesurfer, https://github.com/freesurfer/freesurfer/blob/0366fe10ad1e53d6dd40e912f7e0e9aa489b677c/utils/mrisurf_metricProperties.cpp#L7547
    although I think the average of face normal needs to be weighted by face area (TODO)
    
    """
    v1 = surf.vertices[surf.faces[:,0], :]
    v2 = surf.vertices[surf.faces[:,1], :]
    v3 = surf.vertices[surf.faces[:,2], :]
    face_normal = np.cross(v2-v1, v3-v1)
    tmp = np.sqrt(np.sum(face_normal*face_normal,axis=1))
    tmp[tmp==0] = 1e-10
    face_normal = face_normal/tmp.reshape((-1,1))
    face_normal = np.concatenate((face_normal, np.array([[0,0,0]])), axis=0)
    vertex_normal = np.sum(face_normal[surf.neigh_faces_mat], axis=1)
    tmp = np.sqrt(np.sum(vertex_normal*vertex_normal,axis=1))
    tmp[tmp==0] = 1e-10
    return vertex_normal/np.reshape(tmp, (-1,1))
    

def computeMeanCurvature(surf, dist=0.15, neigh_ring_size=1):
    # surf.updateNeighVertexWithRingSize(neigh_ring_size) # 2-ring has no significant difference compared to 1-ring

    if (surf.vertex_normal == np.zeros((1,1), dtype=np.float64)).all() == 1:
        surf.vertex_normal = computeVertexNormal(surf)
        
    edge_length = computeVertexEdgeLength(surf)
    avg_edge_length = np.mean(edge_length.sum(1)/surf.n_neighs_per_vertex[:, 0])
    print('average edge length: ', avg_edge_length)
    dist_scale = avg_edge_length/dist
    print('pre-set edge distance for sampling: ', dist)
    print('dist scale for computing curvature: ', dist_scale)
   
    # sample neighborhood points to have similar neighborhood distance
    # normalize e_kn and sample it with fixed dist
    e_kn = surf.vertices_append[surf.neigh_vertex_mat] - \
        np.repeat(surf.vertices[:,np.newaxis,:], surf.MAX_NUM_NEIGHBORS, axis=1)
    e_kn_scaled = e_kn / dist_scale
    e_kn_proj = np.sum(e_kn_scaled * np.repeat(surf.vertex_normal[:,np.newaxis,:], 
                                               surf.MAX_NUM_NEIGHBORS, axis=1), axis=-1)
    curv = e_kn_proj / (2 * (np.sum(e_kn_scaled * e_kn_scaled, axis=-1) - e_kn_proj * e_kn_proj))
    curv = surf.setZeroNeighs(curv)
    # zero_curv_num = (np.abs(curv) < 1e-5).sum(1)
    return curv.sum(1) / surf.n_neighs_per_vertex[:, 0]
    

def computeVertexEdgeLength(surf):
    e_kn = surf.vertices_append[surf.neigh_vertex_mat] - \
        np.repeat(surf.vertices[:,np.newaxis,:], surf.MAX_NUM_NEIGHBORS, axis=1)
    e_kn_edge_length = np.linalg.norm(e_kn, axis=2)
    return surf.setZeroNeighs(e_kn_edge_length)


def countNegArea(vertices, faces):
    """
    simple code for checking triangles intersections on sphere,
    only work for sphere, may work for inflated surface,
    but not work for inner surface
    """
    area_sign = computeFaceSign(vertices, faces)
    num_neg_area = int((np.abs(area_sign) - area_sign).sum() / 2)
    return num_neg_area


def computeFaceArea(vertices, faces):
    """
    compute face-wise area

    """
    v1 = vertices[faces[:,0], :]
    v2 = vertices[faces[:,1], :]
    v3 = vertices[faces[:,2], :]
    return np.linalg.norm(np.cross(v2-v1, v3-v1), axis=1)/2.0


def computeFaceSign(v=None, f=None):
    """
    compute face-wise area with sign

    """
    
    v1 = v[f[:,0], :]
    v2 = v[f[:,1], :]
    v3 = v[f[:,2], :]
    
    cross_product = np.cross(v2-v1, v3-v1)
    area = np.linalg.norm(cross_product, axis=1)
    area[area==0] = 1e-10 
    cross_product_normal = cross_product / np.expand_dims(area, axis=1)
    v1_normal = v1 / np.expand_dims(np.linalg.norm(v1, axis=1), axis=1)
    sign = np.sum(cross_product_normal * v1_normal, axis=1)
    # sign = (sign > 0).astype(np.int32) * 2.0 - 1.0
    return np.sign(sign)


def computeVertexArea(surf):
    """
    compute area property for each vertex
    
    """
    vertices = surf.vertices
    faces = surf.faces
    surf.orig_face_area = computeFaceArea(vertices, faces)
    face_wise_area = np.append(surf.orig_face_area, 0)
    area = face_wise_area[surf.neigh_faces_mat].sum(1) / 3.0
    return area


def computeEdgeDistance(vertices, edges):
    v1 = vertices[edges[:,0], :]
    v2 = vertices[edges[:,1], :]
    dis_12 = np.linalg.norm(v2 - v1, axis=1)
    return dis_12


def computeMetrics_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS=6):
    # geodesic distance, Nx6, center vetex to neighbor vertex distance
    distan = computeDistanceOnResp_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS=NUM_NEIGHBORS) 
    
    # triangle area, NxNUM_NEIGHBORS
    area = computeAreaOnResp_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS=NUM_NEIGHBORS)

    # compute angles, for miccai 2022 figures
    # angle = computeAngleOnResp_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS=NUM_NEIGHBORS)
    
    return distan/distan.sum() * 100000.0, area/area.sum() * 100000.0
    # return distan/distan.sum() * 100000.0, area/area.sum() * 100000.0, angle


def computeDistanceOnResp_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS):
    vector_CtoN = vertices[neigh_sorted_orders[:, 1:]] - vertices[neigh_sorted_orders[:, [0]]].repeat(NUM_NEIGHBORS, axis=1)
    distan = np.linalg.norm(vector_CtoN, axis=2)
    return distan
    
    
def computeAreaOnResp_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS):
    """
    

    Parameters
    ----------
    vertices : TYPE
        DESCRIPTION.
    neigh_sorted_orders : The neighbors need to be sorted, so area of neighbor triangles
    can be sequentially computed according to the neighbor orders one by one.
        DESCRIPTION.
    NUM_NEIGHBORS : TYPE
        DESCRIPTION.

    Returns
    -------
    area : TYPE
        DESCRIPTION.

    """
    vertices_append = np.concatenate((vertices, np.array([[0,0,0]])), axis=0)
    area = np.zeros((vertices.shape[0], NUM_NEIGHBORS))
    for i in range(NUM_NEIGHBORS):
        if i < NUM_NEIGHBORS-1:
            a = vertices_append[neigh_sorted_orders[:, i+1]]
            b = vertices_append[neigh_sorted_orders[:, i+2]]
        else:
            a = vertices_append[neigh_sorted_orders[:, -1]]
            b = vertices_append[neigh_sorted_orders[:, 1]]
        c = vertices
        cros_vec = np.cross(a-c, b-c)
        area[:, i] = 1/2 * np.linalg.norm(cros_vec, axis=1)

    return area


def computeAngleOnResp_np(vertices, neigh_sorted_orders, NUM_NEIGHBORS):
    # compute angles, for miccai 2022 figures
    vector_CtoN = vertices[neigh_sorted_orders[:, 1:]] - np.repeat(vertices[neigh_sorted_orders[:, [0]]], NUM_NEIGHBORS, axis=1)
    vector_CtoN[0:12, -1, 0] = 1
    angle = np.zeros((vertices.shape[0], NUM_NEIGHBORS))
    for i in range(NUM_NEIGHBORS):
        if i < NUM_NEIGHBORS-1:
            v01 = vector_CtoN[:, i, :]
            v02 = vector_CtoN[:, i+1, :]
        else:
            v01 = vector_CtoN[:, -1, :]
            v02 = vector_CtoN[:, 0, :]
        angle[:, i] = np.arccos(np.clip(np.sum(v01 * v02, axis=1) / np.linalg.norm(v01, axis=1) / np.linalg.norm(v02, axis=1), -1, 1))
    
    # normalize angles for each vertex
    angle = angle / angle.sum(1, keepdims=True) * 360
    
    return angle


def computeAngles(vertices, faces, vertex_has_angles):
    num_vers = vertices.shape[0]
    num_faces = faces.shape[0]
    angle = np.zeros((num_faces, 3))
    
    v1 = vertices[faces[:,0], :]
    v2 = vertices[faces[:,1], :]
    v3 = vertices[faces[:,2], :]
    
    v0_12 = v2 - v1
    v0_23 = v3 - v2
    v0_13 = v3 - v1
    
    tmp = reduce(np.intersect1d, (np.where(np.linalg.norm(v0_12, axis=1) > 0), 
                                  np.where(np.linalg.norm(v0_13, axis=1) > 0), 
                                  np.where(np.linalg.norm(v0_23, axis=1) > 0)))

    angle[:, 0][tmp] = np.arccos(np.clip(np.sum(v0_12 * v0_13, axis=1)[tmp] / \
                                         np.linalg.norm(v0_12, axis=1)[tmp] / \
                                         np.linalg.norm(v0_13, axis=1)[tmp], -1, 1))
    angle[:, 1][tmp] = np.arccos(np.clip(np.sum(-v0_12 * v0_23, axis=1)[tmp] / \
                                         np.linalg.norm(-v0_12, axis=1)[tmp] / \
                                         np.linalg.norm(v0_23, axis=1)[tmp], -1, 1))
    angle[:, 2][tmp] = np.arccos(np.clip(np.sum(v0_23 * v0_13, axis=1)[tmp] / \
                                         np.linalg.norm(v0_23, axis=1)[tmp] / \
                                         np.linalg.norm(v0_13, axis=1)[tmp], -1, 1))
    angle = np.reshape(angle, (num_faces*3, 1))
    angle[np.where(angle==0)] = angle.mean()
    
    # normalize angles for each vertex
    for j in range(num_vers):
        tmp = angle[vertex_has_angles[j]].sum()
        angle[vertex_has_angles[j]] = angle[vertex_has_angles[j]] / tmp
        
    return angle  
