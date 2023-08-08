#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 11:36:21 2020

@author: Fenqiang Zhao, https://github.com/BRAIN-Lab-UNC/S3pipeline

Contact: zhaofenqiang0221@gmail.com


2023.3.30 add longleaf flag for using single cpu on longleaf
...



"""
import numpy as np
from sklearn.neighbors import KDTree
from .utils import get_neighs_faces
import math, multiprocessing, os
import time

abspath = os.path.abspath(os.path.dirname(__file__))


# def diffeomorp_np(fixed_xyz, phi_3d, num_composition=6, bi=False, bi_inter=None):
#     if bi:
#         assert bi_inter is not None, "bi_inter is None!"
    
#     warped_vertices = fixed_xyz + phi_3d
#     warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
#     # compute exp
#     for i in range(num_composition):
#         if bi:
#             warped_vertices = bilinearResampleSphereSurf(warped_vertices, warped_vertices, bi_inter)
#         else:
#             warped_vertices = resampleSphereSurf(fixed_xyz, warped_vertices, warped_vertices)
        
#         warped_vertices = warped_vertices/np.linalg.norm(warped_vertices, axis=1)[:,np.newaxis]
    
#     # get deform from warped_vertices 
#     # tmp = 1.0/np.sum(np.multiply(fixed_xyz, warped_vertices), 1)[:,np.newaxis] * warped_vertices

#     return warped_vertices


def get_bi_inter(n_vertex):
    inter_indices_0 = np.load(abspath+'/neigh_indices/img_indices_'+ str(n_vertex) +'_0.npy')
    inter_weights_0 = np.load(abspath+'/neigh_indices/img_weights_'+ str(n_vertex) +'_0.npy')
    
    inter_indices_1 = np.load(abspath+'/neigh_indices/img_indices_'+ str(n_vertex) +'_1.npy')
    inter_weights_1 = np.load(abspath+'/neigh_indices/img_weights_'+ str(n_vertex) +'_1.npy')
    
    inter_indices_2 = np.load(abspath+'/neigh_indices/img_indices_'+ str(n_vertex) +'_2.npy')
    inter_weights_2 = np.load(abspath+'/neigh_indices/img_weights_'+ str(n_vertex) +'_2.npy')
    
    return (inter_indices_0, inter_weights_0), (inter_indices_1, inter_weights_1), (inter_indices_2, inter_weights_2)


def get_latlon_img(bi_inter, feat):
    inter_indices, inter_weights = bi_inter
    width = int(np.sqrt(len(inter_indices)))
    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]
        
    img = np.sum(np.multiply((feat[inter_indices.flatten()]).reshape((inter_indices.shape[0],
                                                                      inter_indices.shape[1], 
                                                                      feat.shape[1])), 
                             np.repeat(inter_weights[:,:, np.newaxis], 
                                       feat.shape[1], axis=-1)), 
                 axis=1)
    
    img = img.reshape((width, width, feat.shape[1]))
    
    return img

 
def isATriangle(neigh_orders, face):
    """
    neigh_orders: int, N x 7
    face: int, 3 x 1
    """
    neighs = neigh_orders[face[0]]
    if face[1] not in neighs or face[2] not in neighs:
        return False
    neighs = neigh_orders[face[1]]
    if face[2] not in neighs:
        return False
    return True


def projectVertex(vertex, v0, v1, v2):
    normal = np.cross(v0 - v2, v1 - v2)
    if np.linalg.norm(normal) == 0:
        normal = v0
    ratio = v0.dot(normal)/vertex.dot(normal)
    return ratio * vertex


def isOnSameSide(P, v0 , v1, v2):
    """
    Check if P and v0 is on the same side
    """
    edge_12 = v2 - v1
    tmp0 = P - v1
    tmp1 = v0 - v1
    
    edge_12 = edge_12 / np.linalg.norm(edge_12)
    tmp0 = tmp0 / np.linalg.norm(tmp0)
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    
    vec1 = np.cross(edge_12, tmp0)
    vec2 = np.cross(edge_12, tmp1)
    
    return vec1.dot(vec2) >= 0


def isInTriangle(vertex, v0, v1, v2):
    """
    Check if the vertices is in the triangle composed by v0 v1 v2
    vertex: N*3, check N vertices at the same time
    v0: (3,)
    v1: (3,)
    v2: (3,)
    """
    # Project point onto the triangle plane
    P = projectVertex(vertex, v0, v1, v2)
          
    return isOnSameSide(P, v0, v1, v2) and isOnSameSide(P, v1, v2, v0) and isOnSameSide(P, v2, v0, v1)


def inTriangle(vertex, v0, v1, v2, threshold):
    normal = np.cross(v1-v2, v0-v2)
    if np.linalg.norm(normal) < 1e-10:
        normal = vertex
    normal = normal / np.linalg.norm(normal) 
    vertex_proj = v0.dot(normal)/vertex.dot(normal) * vertex
    area_BCP = np.linalg.norm(np.cross(v2-vertex_proj, v1-vertex_proj))/2.0
    area_ACP = np.linalg.norm(np.cross(v2-vertex_proj, v0-vertex_proj))/2.0
    area_ABP = np.linalg.norm(np.cross(v1-vertex_proj, v0-vertex_proj))/2.0
    area_ABC = np.linalg.norm(np.cross(v1-v2, v0-v2))/2.0
    
    return area_BCP + area_ACP + area_ABP - area_ABC < threshold
    
     
def searchCandiFaces(nn_index,
                     neigh_faces,
                     faces,
                     nn_init,
                     num_candi_faces):
    nearest_vertex_index = nn_index[0:nn_init]
    candi_faces = [ ind for x in nearest_vertex_index for ind in neigh_faces[x] ]
    candi_faces = list(set(candi_faces))
    while len(candi_faces) < num_candi_faces and nn_init < 10000:
        nn_init += 1
        nearest_vertex_index = nn_index[0:nn_init]
        candi_faces = [ ind for x in nearest_vertex_index for ind in neigh_faces[x] ]
        candi_faces = list(set(candi_faces))
    return faces[candi_faces[0:num_candi_faces], :]



def searchNearestFace(vertices_fix,
                      vertices_inter,
                      remain,
                      nn_indices, 
                      neigh_faces,
                      faces, 
                      num_candi_faces,
                      max_candi_faces,
                      threshold):
    '''
    num_candi_faces should be >= 3

    '''
    num_remains = len(remain)
    candi_faces = np.zeros((num_remains, num_candi_faces, 3), dtype=np.int32)
    # search candidate faces, 0.99, 1.02, 1.12 (s) for nn_init=1, 2, 3, num_candi_faces=5, best nn_init=1 (5/3)
    #                         0.91, 1.02, 1.12 , for nn_init=1, 2, 3, num_candi_faces=4, best nn_init=1 (4/3)
    #                         0.42, 0.44, 0.48 , nn_init=3, 4, 5, num_candi_faces=10, best nn_init=3 (10/3)
    #                         0.98, 0.74, 0.55, 0.56, 0.6 for nn_init=4,5,6,7,8, num_candi_faces=20, best nn_init=6 (20/3)
    #                         1.11, 0.76, 0.67, 0.69, 0.74 for 8,9,10,11,12, num_candi_faces=30, best nn_init=10 (30/3)
    for i in range(num_remains):
        tmp = searchCandiFaces(nn_indices[remain[i]],
                                neigh_faces,
                                faces,
                                nn_init=int(num_candi_faces/3),
                                num_candi_faces=num_candi_faces)
        if tmp.shape == (num_candi_faces, 3):
            candi_faces[i, :, :] = tmp
        else:
            raise NotImplementedError('Error: k value used to query kd tree is too small. Cannot find correct face from the top k nearest vertices.')
    
    vertices_remain = vertices_inter[remain][:, np.newaxis, :].repeat(num_candi_faces, axis=1)
    candi_v0 = vertices_fix[candi_faces[:,:,0],:]
    candi_v1 = vertices_fix[candi_faces[:,:,1],:]
    candi_v2 = vertices_fix[candi_faces[:,:,2],:]
        
    # use formula p(x) = <p1,n>/<x,n> * x in spherical demons to calculate the intersection with each faces
    normal = np.cross(candi_v1-candi_v2, candi_v0-candi_v2, axis=-1)
    tmp0, tmp1 = (np.linalg.norm(normal, axis=-1) < 1e-10).nonzero()
    normal[tmp0, tmp1] = candi_v0[tmp0, tmp1]
    normal = normal / np.linalg.norm(normal, axis=-1)[:, :, np.newaxis]
    vertex_proj = np.expand_dims(np.sum(candi_v0*normal, axis=-1)/ \
                                 np.sum(vertices_remain*normal, axis=-1), axis=2) * vertices_remain
    
    # find the face that the inersection is in, if the intersection
    # is in, the area of 3 small triangles is equal to the whole one
    area_BCP = np.linalg.norm(np.cross(candi_v2-vertex_proj, candi_v1-vertex_proj, axis=-1), axis=-1)/2.0
    area_ACP = np.linalg.norm(np.cross(candi_v2-vertex_proj, candi_v0-vertex_proj, axis=-1), axis=-1)/2.0
    area_ABP = np.linalg.norm(np.cross(candi_v1-vertex_proj, candi_v0-vertex_proj, axis=-1), axis=-1)/2.0
    area_ABC = np.linalg.norm(np.cross(candi_v1-candi_v2, candi_v0-candi_v2, axis=-1), axis=-1)/2.0
    
    area_diff = area_BCP + area_ACP + area_ABP - area_ABC
    pass_index = np.argmin(area_diff, axis=-1)
    pass_index_TF = (area_diff[np.arange(len(area_diff)), pass_index] < threshold)
    pass_index0 = pass_index_TF.nonzero()[0]
    pass_index1 = (pass_index_TF==False).nonzero()[0]

    if num_candi_faces < max_candi_faces:
        return remain[pass_index0], candi_faces[pass_index0, pass_index[pass_index0], :], remain[pass_index1], len(pass_index1)
    else:
        print('Warning: num_candi_faces >= max_candi_faces, use candidate face with max possibility.')
        return remain, candi_faces[np.arange(num_remains), pass_index, :], [], len(pass_index1)

    
def treeQueryNearestVertex(vertexs, tree, kmax=3):
    n = len(vertexs)
    inter_indices = np.zeros((n, kmax), dtype=np.int32) - 1
    for i in range(n):
        inter_indices[i, :] = tree.query(vertexs[[i],:], k=kmax)[1].squeeze()
    return inter_indices
 

def resampleStdSphereSurf(n_curr, n_next, feat, upsample_neighbors):
    assert len(feat) == n_curr, "feat length not cosistent!"
    assert n_next == n_curr*4-6, "This function can only upsample one level higher"+ \
        "at each call. Need to call twice if upsampling with tow levels higher is needed."
    
    feat_inter = np.zeros((n_next, feat.shape[1]))
    feat_inter[0:n_curr, :] = feat
    feat_inter[n_curr:, :] = feat[upsample_neighbors].reshape(n_next-n_curr, 2, feat.shape[1]).mean(1)
    
    return feat_inter



def resampleSphereSurf(vertices_fix, 
                       vertices_inter, 
                       feat,
                       faces=None,
                       neigh_faces=None,
                       std=False,
                       upsample_neighbors=None,
                       fast=False,
                       threshold=1e-8, 
                       max_candi_faces=220,
                       init_candi_faces=5,
                       nn_step=10,
                       longleaf=False):
    """
    resample sphere surface

    Parameters
    ----------
    vertices_fix :  N*3, numpy array, 
        the original fixed vertices with features.
    vertices_inter : unknown*3, numpy array, 
        points to be interpolated.
    feat :  N*D, 
        features to be interpolated.
    neigh_faces: 
        should be a list[list]
    faces :
        N*3, numpy array
    std : bool
        standard sphere interpolation, e.g., interpolate 10242 from 2562.. The default is False.
    upsample_neighbors : TYPE, optional
        DESCRIPTION. The default is None.
    neigh_orders : TYPE, optional
        DESCRIPTION. The default is None.
    fast : TYPE, bool, if using nearest triangle for fast interpolation 
        DESCRIPTION. The default is False.
    threshold : TYPE, optional
        DESCRIPTION. The default is 1e-8.
        
    Note:
        1. num_candi_faces and nn_step can be finetuned to speedup the process based on
        the detailed output log
        
    Returns
    -------
    resampled feature.
    
    """
    print('Start resampling...')
    
    assert vertices_fix.shape[0] == feat.shape[0], "vertices.shape[0] is not equal to feat.shape[0]"
    assert vertices_fix.shape[1] == 3, "vertices size not right"
    
    kmax = int(max_candi_faces/2) + 5
    
    vertices_fix = vertices_fix.astype(np.float64)
    vertices_inter = vertices_inter.astype(np.float64)
    feat = feat.astype(np.float64)
    
    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1
    
    num_vertices = vertices_inter.shape[0]
    if len(feat.shape) == 1:
        feat = feat[:,np.newaxis]
        
    if std:
        assert upsample_neighbors is not None, " upsample_neighbors is None"
        return resampleStdSphereSurf(len(vertices_fix), len(vertices_inter), feat, upsample_neighbors)
    
    if not fast:
        assert faces.shape[1] == 3,  "faces is not given or size is not correct (shoud be num_faces x 3)"
        if neigh_faces is not None:
            print('Using precomputed neighborhood faces...')
        else:
            print('Computing neighborhood faces for resampling surface...')
            neigh_faces = get_neighs_faces(len(vertices_fix), faces)

    
    inter_indices = np.zeros((num_vertices, 3), dtype=np.int32)
    nn_indices = np.zeros((num_vertices, kmax), dtype=np.int32)
    tree = KDTree(vertices_fix, leaf_size=50)  # build kdtree
    
    # remember to change longleaf to True when running program on longleaf
    t1 = time.time()
    if longleaf:
        """ Single process on longleaf: 163842: 18s """ 
        for i in range(num_vertices):
            if i % 10000==0:
                print('Vertex number: {}/{}'.format(i, num_vertices))
            _, nn_index = tree.query(vertices_inter[[i], :], k=kmax)
            nn_indices[i,:] = nn_index[0]
    else:
        """ Single process, single thread: 163842:  9.0s, 40962:  s, 10242:  s, 2562:  s  
          multiple processes method: 163842: 1.5s (k=3), 1.7s (k=20), 1.9s (k=50), 
          40962: ?s, 10242: ?s, 2562: ?s, """
        pool = multiprocessing.Pool()
        cpus = multiprocessing.cpu_count()
        vertexs_num_per_cpu = math.ceil(num_vertices/cpus)
        results = []
        for i in range(cpus):
            results.append(pool.apply_async(treeQueryNearestVertex, 
                                            args=(vertices_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:],
                                                  tree,
                                                  kmax)))
        pool.close()
        pool.join()
        for i in range(cpus):
            nn_indices[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:] = results[i].get()
    t2 = time.time()
    print('Build kdtree and Query kdtree took {:.1f} s'.format(t2-t1))
    
    
    if fast:
        inter_indices = nn_indices[:, 0:3]
    else:    
        # optimal parameters tuned on HCP dataset with >300,000 vertices
        # Step 5/200 took 1.20 s. Remaining vertices to be interpolated: 33005
        # Step 15/200 took 0.45 s. Remaining vertices to be interpolated: 11352
        # Step 25/200 took 0.22 s. Remaining vertices to be interpolated: 4498
        # Step 35/200 took 0.11 s. Remaining vertices to be interpolated: 1807
        # Step 45/200 took 0.05 s. Remaining vertices to be interpolated: 1181
        # Step 55/200 took 0.04 s. Remaining vertices to be interpolated: 561
        # Step 65/200 took 0.03 s. Remaining vertices to be interpolated: 279
        # Step 75/200 took 0.02 s. Remaining vertices to be interpolated: 96
        # Step 85/200 took 0.01 s. Remaining vertices to be interpolated: 17
        # Step 95/200 took 0.00 s. Remaining vertices to be interpolated: 10
        # Step 105/200 took 0.00 s. Remaining vertices to be interpolated: 8
        # Step 115/200 took 0.00 s. Remaining vertices to be interpolated: 3
        # Step 125/200 took 0.00 s. Remaining vertices to be interpolated: 3
        # Step 135/200 took 0.00 s. Remaining vertices to be interpolated: 1
        # Step 145/200 took 0.00 s. Remaining vertices to be interpolated: 1
        # Step 155/200 took 0.00 s. Remaining vertices to be interpolated: 1
        # Step 165/200 took 0.00 s. Remaining vertices to be interpolated: 1
        # Step 175/200 took 0.00 s. Remaining vertices to be interpolated: 0
        # Resampling done, took 4.0 s

        remain = np.arange(num_vertices)
        num_candi_faces = init_candi_faces
        while len(remain) > 0:
            t1 = time.time()
            pass_vertex, pass_indices, \
                new_remain, length_remain = searchNearestFace(vertices_fix, 
                                                              vertices_inter, 
                                                              remain,
                                                              nn_indices,
                                                              neigh_faces,
                                                              faces, 
                                                              num_candi_faces=num_candi_faces,
                                                              max_candi_faces=max_candi_faces,
                                                              threshold=threshold)
            inter_indices[pass_vertex] = pass_indices
            remain = new_remain
            t2 = time.time()
            print('Candidate faces {:}/{:} took {:.2f} s. Remaining vertices to be interpolated: {}'.format(num_candi_faces, 
                                                                                                            max_candi_faces,
                                                                                                            t2-t1,
                                                                                                            length_remain))
            num_candi_faces += nn_step
   

    v0 = vertices_fix[inter_indices[:, 0]]
    v1 = vertices_fix[inter_indices[:, 1]]    
    v2 = vertices_fix[inter_indices[:, 2]]
    normal = np.cross(v1-v0, v2-v0, axis=1)
    vertex_proj = np.sum(v0*normal, axis=1, keepdims=True) / np.sum(vertices_inter*normal, axis=1, keepdims=True) * vertices_inter
    
    tmp_index = np.argwhere(np.isnan(vertex_proj))[:, 0]  # in case that normal is [0,0,0]
    
    area_12P = np.linalg.norm(np.cross(v2-vertex_proj, v1-vertex_proj, axis=1), axis=1, keepdims=True)/2.0
    area_02P = np.linalg.norm(np.cross(v2-vertex_proj, v0-vertex_proj, axis=1), axis=1, keepdims=True)/2.0
    area_01P = np.linalg.norm(np.cross(v1-vertex_proj, v0-vertex_proj, axis=1), axis=1, keepdims=True)/2.0
    
    inter_weights = np.concatenate(([area_12P, area_02P, area_01P]), axis=1)
    inter_weights[tmp_index] = np.array([1,0,0])    # in case that normal is [0,0,0]
    inter_weights = inter_weights / np.sum(inter_weights, axis=1, keepdims=True)

    feat_inter = np.sum(np.multiply(feat[inter_indices], np.repeat(inter_weights[:,:,np.newaxis], feat.shape[1], axis=2)), axis=1)
    
    return feat_inter



def bilinear_interpolate(im, x, y):

    x = np.clip(x, 0.0001, im.shape[1]-1.0001)
    y = np.clip(y, 0.0001, im.shape[1]-1.0001)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa[:,np.newaxis]*Ia + wb[:,np.newaxis]*Ib + wc[:,np.newaxis]*Ic + wd[:,np.newaxis]*Id

        
def bilinearResampleSphereSurf(vertices_inter, feat, bi_inter, radius=1.0):
    """
    ONLY!! assume vertices_fix are on the standard icosahedron discretized spheres!!
    
    """
    img = get_latlon_img(bi_inter, feat)
    
    return bilinearResampleSphereSurfImg(vertices_inter, img, radius=radius)


def bilinearResampleSphereSurfImg(vertices_inter_raw, img, radius=1.0):
    """
    ASSUME vertices_fix are on the standard icosahedron discretized spheres!
    
    """
    vertices_inter = np.copy(vertices_inter_raw)
    vertices_inter = vertices_inter / radius
    width = img.shape[0]
    
    vertices_inter[:,2] = np.clip(vertices_inter[:,2], -0.999999999, 0.999999999)
    beta = np.arccos(vertices_inter[:,2]/1.0)
    row = beta/(np.pi/(width-1))
    
    tmp = (vertices_inter[:,0] == 0).nonzero()[0]
    vertices_inter[:,0][tmp] = 1e-15
    
    alpha = np.arctan(vertices_inter[:,1]/vertices_inter[:,0])
    tmp = (vertices_inter[:,0] < 0).nonzero()[0]
    alpha[tmp] = np.pi + alpha[tmp]
    
    alpha = 2*np.pi + alpha
    alpha = np.remainder(alpha, 2*np.pi)
    
    col = alpha/(2*np.pi/(width-1))
    
    feat_inter = bilinear_interpolate(img, col, row)
    
    return feat_inter


def resample_label(vertices_fix, vertices_inter, label, multiprocess=True):
    """
    
    Resample label using nearest neighbor on sphere
    
    Parameters
    ----------
    vertices_fix : N*3 numpy array,
          original sphere.
    vertices_inter : M*3 numpy array
        the sphere to be interpolated.
    label : [N, ?], numpy array,
         the label on orignal sphere.

    Returns
    -------
    label_inter : TYPE
        DESCRIPTION.

    """
    assert len(vertices_fix) == len(label), "length of label should be "+\
        "consistent with the length of vertices on orginal sphere."
    if len(label.shape) == 1:
        label = label[:,np.newaxis]
    
    vertices_fix = vertices_fix / np.linalg.norm(vertices_fix, axis=1)[:,np.newaxis]  # normalize to 1
    vertices_inter = vertices_inter / np.linalg.norm(vertices_inter, axis=1)[:,np.newaxis]  # normalize to 1
    
    tree = KDTree(vertices_fix, leaf_size=50)  # build kdtree
    label_inter = np.zeros((len(vertices_inter), label.shape[1])).astype(np.int32)
    
    """ multiple processes method: 163842:  s, 40962:  s, 10242: s, 2562: s """
    if  multiprocess:
        pool = multiprocessing.Pool()
        cpus = multiprocessing.cpu_count()
        vertexs_num_per_cpu = math.ceil(vertices_inter.shape[0]/cpus)
        results = []
        for i in range(cpus):
            results.append(pool.apply_async(multiVertexLabel, 
                                            args=(vertices_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:],
                                                  vertices_fix, tree, label,)))
        pool.close()
        pool.join()
        for i in range(cpus):
            label_inter[i*vertexs_num_per_cpu:(i+1)*vertexs_num_per_cpu,:] = results[i].get()
            
    else:
        for i in range(len(vertices_inter)):
            _, nearest_vertex = tree.query(vertices_inter[i,:][np.newaxis,:], k=1)
            label_inter[i] = label[nearest_vertex]
          
    return label_inter

def multiVertexLabel(vertexs, vertices, tree, label):
    label_inter = np.zeros((vertexs.shape[0], label.shape[1]))
    for i in range(vertexs.shape[0]):
        _, nearest_vertex = tree.query(vertexs[i,:][np.newaxis,:], k=1)
        label_inter[i] = label[nearest_vertex]
    return label_inter



# def singleVertexInterpo_NN(vertex, vertices, tree, neigh_orders, nnvertex_iter=1, 
#                            nnvertex_threshold=3, threshold=1e-8, debug=False):
#     """
#     assume the triangle is in the one-ring of the nnvertex_iter-th nearest neighborhood vertex
#     """
#     if nnvertex_iter > nnvertex_threshold:
#         print("ring_iter > ring_threshold, use nearest 3 neighbors")
#         _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
#         return top3_near_vertex_index[0]

#     _, nearest_vertex_indices = tree.query(vertex[np.newaxis,:], k=3)
#     nearest_vertex_index = nearest_vertex_indices[0]
    
#     if type(neigh_orders) == list:
#         candi_vertex_0 = neigh_orders[nearest_vertex_index[0]]
#         candi_vertex_1 = neigh_orders[nearest_vertex_index[1]]
#         candi_vertex_2 = neigh_orders[nearest_vertex_index[2]]
#     else:
#         candi_vertex_0 = neigh_orders[nearest_vertex_index[0]][:-1]
#         candi_vertex_1 = neigh_orders[nearest_vertex_index[1]][:-1]
#         candi_vertex_2 = neigh_orders[nearest_vertex_index[2]][:-1]

#     candi_faces = []
#     for (nearest_vertex_index, candi_vertex) in zip(nearest_vertex_index, [candi_vertex_0, candi_vertex_1, candi_vertex_2]):
#         n_candi_vertex = len(candi_vertex)
#         for t in range(n_candi_vertex):
#             candi_faces.append([nearest_vertex_index, candi_vertex[t], candi_vertex[(t+1)%n_candi_vertex]])
#     candi_faces = np.asarray(candi_faces)
#     candi_faces = np.unique(candi_faces, axis=0)

#     orig_vertex_1 = vertices[candi_faces[:,0]]
#     orig_vertex_2 = vertices[candi_faces[:,1]]
#     orig_vertex_3 = vertices[candi_faces[:,2]]
#     edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
#     edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
#     faces_normal = np.cross(edge_12, edge_13)    # normals of all the faces
#     tmp = (np.linalg.norm(faces_normal, axis=1) < 1e-10).nonzero()[0]
#     faces_normal[tmp] = orig_vertex_1[tmp]
#     faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:,np.newaxis]

#     # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
#     tmp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
#     P = tmp[:, np.newaxis] * vertex  # intersection points

#     # find the triangle face that the inersection is in, if the intersection
#     # is in, the area of 3 small triangles is equal to the whole one
#     area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
#     area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABC = np.linalg.norm(faces_normal, axis=1)/2.0
    
#     tmp = area_BCP + area_ACP + area_ABP - area_ABC
#     index = np.argmin(tmp)
    
#     if tmp[index] > threshold:
#         return singleVertexInterpo_NN(vertex, vertices, tree, neigh_orders, 
#                                         nnvertex_iter=nnvertex_iter+1, nnvertex_threshold=nnvertex_threshold, 
#                                         threshold=threshold, debug=False)

#     return candi_faces[index]
    


# def singleVertexInterpo_ring(vertex, vertices, tree, neigh_orders, ring_iter=1, ring_threshold=3, threshold=1e-8, debug=False):
    
#     if ring_iter > ring_threshold:
#         print("ring_iter > ring_threshold, use nearest 3 neighbors")
#         _, top3_near_vertex_index = tree.query(vertex[np.newaxis,:], k=3)
#         return np.squeeze(top3_near_vertex_index)

#     _, top1_near_vertex_index = tree.query(vertex[np.newaxis,:], k=1)
#     ring = []
    
#     if type(neigh_orders) == list:
#         ring.append({np.squeeze(top1_near_vertex_index).tolist()})  # 0-ring index
#         ring.append(set(neigh_orders[np.squeeze(top1_near_vertex_index)]))        # 1-ring neighs
#         for i in range(ring_iter-1):
#             tmp = set()
#             for j in ring[i+1]:
#                 tmp = set.union(tmp, set(neigh_orders[j]))
#             ring.append(tmp-ring[i]-ring[i+1])
#         candi_vertex = set.union(ring[-1], ring[-2])
#     else:
#         ring.append(np.squeeze(top1_near_vertex_index))  # 0-ring index
#         ring.append(np.setdiff1d(np.unique(neigh_orders[ring[0]]), ring[0]))    # 1-ring neighs
#         for i in range(ring_iter-1):
#             tmp = np.setdiff1d(np.unique(neigh_orders[ring[i+1]].flatten()), ring[i+1])
#             ring.append(np.setdiff1d(tmp, ring[i]))
#         candi_vertex = np.append(ring[-1], ring[-2])

#     candi_faces = []
#     for t in itertools.combinations(candi_vertex, 3):
#         tmp = np.asarray(t)  # get the indices of the potential candidate triangles
#         if isATriangle(neigh_orders, tmp):
#               candi_faces.append(tmp)
#     candi_faces = np.asarray(candi_faces)

#     orig_vertex_1 = vertices[candi_faces[:,0]]
#     orig_vertex_2 = vertices[candi_faces[:,1]]
#     orig_vertex_3 = vertices[candi_faces[:,2]]
#     edge_12 = orig_vertex_2 - orig_vertex_1        # edge vectors from vertex 1 to 2
#     edge_13 = orig_vertex_3 - orig_vertex_1        # edge vectors from vertex 1 to 3
#     faces_normal = np.cross(edge_12, edge_13)    # normals of all the faces
#     tmp = (np.linalg.norm(faces_normal, axis=1) < 1e-10).nonzero()[0]
#     faces_normal[tmp] = orig_vertex_1[tmp]
#     faces_normal_norm = faces_normal / np.linalg.norm(faces_normal, axis=1)[:,np.newaxis]

#     # use formula p(x) = <p1,n>/<x,n> * x in spherical demons paper to calculate the intersection with each faces
#     tmp = np.sum(orig_vertex_1 * faces_normal_norm, axis=1) / np.sum(vertex * faces_normal_norm, axis=1)
#     P = tmp[:, np.newaxis] * vertex  # intersection points

#     # find the triangle face that the inersection is in, if the intersection
#     # is in, the area of 3 small triangles is equal to the whole one
#     area_BCP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_2-P), axis=1)/2.0
#     area_ACP = np.linalg.norm(np.cross(orig_vertex_3-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABP = np.linalg.norm(np.cross(orig_vertex_2-P, orig_vertex_1-P), axis=1)/2.0
#     area_ABC = np.linalg.norm(faces_normal, axis=1)/2.0
    
#     tmp = area_BCP + area_ACP + area_ABP - area_ABC
#     index = np.argmin(tmp)
    
#     if tmp[index] > threshold:
#         return singleVertexInterpo_ring(vertex, vertices, tree, neigh_orders, 
#                                         ring_iter=ring_iter+1, ring_threshold=ring_threshold, 
#                                         threshold=threshold, debug=False)

#     return candi_faces[index]
    
