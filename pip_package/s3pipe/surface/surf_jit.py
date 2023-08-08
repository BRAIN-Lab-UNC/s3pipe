#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:12:22 2023

@author: Fenqiang Zhao
"""
import numpy as np
from numba import jit






# only put the functions that cannot
# be implemnted by numpy matrix and involves complex iterations here, e.g. vertex-wisely
# set the neighborhood feature to 0 
        
@jit(nopython=True)
def setZeroNeighs(self, mag, n_neighs_per_vertex=None):
    if n_neighs_per_vertex == None:
        n_neighs_per_vertex = self.n_neighs_per_vertex
    for i in range(self.num_vertices):
        mag[i, n_neighs_per_vertex[i,0]:] = 0
    return mag
 
def npLinalgNorm(x, axis=1):
    return np.sqrt(np.sum(x * x, axis=axis))




# The following content should be commented


import numpy as np
from numba import float64, int32    # import the types
from numba.types import string
from numba.experimental import jitclass

import s3pipe.surface.prop as sprop


# This jitclass will be implemented by numba jit class, where all functions
# will be regarded as @jit(nopython=True), so some non-jit functions are stil
# implemented in python outside this class. But idealy, all related functions should be
# included in this class and implemented by numba jit to accelerate the process.
@jitclass([
        ('vertices', float64[:, :]), 
        ('vertices_append', float64[:, :]), 
        ('faces', int32[:,:]),
        ('num_vertices', int32),
        ('num_faces', int32),
        ('surf_status', string),
        ('neigh_vertex', int32[:,:]),
        ('neigh_vertex_1d', int32[:]),
        ('n_neighs_per_vertex', int32[:,:]),
        ('MAX_NUM_NEIGHBORS', int32),
        ('MAX_NUM_FACES', int32),
        ('n_faces_per_vertex', int32[:]),
        ('neigh_faces', int32[:,:]),
        ('orig_metrics', float64[:,:]),
        ('curr_metrics', float64[:,:]),
        ('vertex_normal', float64[:,:]),
        ('orig_face_area', float64[:]),
        ('sulc', float64[:]),
        ('curv', float64[:]),
        ('gradient', float64[:,:]),
        ('momentum_grad', float64[:,:]),
        ('neigh_vertex_nring', int32[:,:]),
        ('n_neighs_nring', int32[:,:]),
        ('MAX_NUM_NEIGHBORS_NRING', int32),
        ('edge_length_per_vertex', float64[:,:]),
        ])
class Surface(object):
    def __init__(self, vertices, faces):
        """
        This class mainly includes the surface function needs loop operation, so 
        use the numba jit to accelerate it. Other already vectorized funtions 
        are thus outside this class (numba does not support all numpy vectorized
                                     funtion).

        Parameters
        ----------
        vertices : TYPE
            DESCRIPTION.
        faces : size: Nx4, with all 3 in first column
            DESCRIPTION.

        Returns
        -------
        None.

        """
        assert len(faces.shape) == 2, "faces shape is not corret."
        assert (faces[:, 0] == 3).sum() == faces.shape[0], "first column of faces should be all 3." 
        
        self.vertices = vertices.astype(np.float64)
        self.vertices_append = np.concatenate((self.vertices, np.array([[0,0,0]])), axis=0)
        self.faces = faces[:, 1:].astype(np.int32) 
        self.num_vertices = 0
        self.num_faces = 0
        self.surf_status = "None"
        self.neigh_vertex = np.zeros((1,1), dtype=np.int32)
        self.neigh_vertex_1d = np.zeros((1,), dtype=np.int32)
        self.n_neighs_per_vertex = np.zeros((1,1), dtype=np.int32)
        self.MAX_NUM_NEIGHBORS = 0
        # self.property_dic = {'thickness': 0}
        
        self.updateSurfaceProperties()
        self.updateNeighVertex()

        self.neigh_faces = np.zeros((1,1), dtype=np.int32)
        self.vertex_normal = np.zeros((1,1), dtype=np.float64)
        self.n_faces_per_vertex = np.zeros((1,), dtype=np.int32)
        self.MAX_NUM_FACES = 0
         
        self.orig_metrics = np.zeros((1,1), dtype=np.float64)
        self.orig_face_area = np.zeros((1,), dtype=np.float64)
        self.edge_length_per_vertex = np.zeros((1,1), dtype=np.float64)
        self.sulc = np.zeros((1,), dtype=np.float64)
        self.curv = np.zeros((1,), dtype=np.float64)
        self.curr_metrics = np.zeros((1,1), dtype=np.float64)
        self.gradient = np.zeros((1,1), dtype=np.float64)
        self.momentum_grad = np.zeros((1,1), dtype=np.float64)
        
        # storing n_ring neighborhood vertex index for computing mean curvature
        self.neigh_vertex_nring = np.zeros((1,1), dtype=np.int32)
        self.n_neighs_nring = np.zeros((1,1), dtype=np.int32)
        self.MAX_NUM_NEIGHBORS_NRING = 0
        
        
    def updateSurfaceProperties(self):
        self.num_vertices = len(self.vertices)
        self.num_faces = len(self.faces)
        if self.num_vertices in [12,42,162,642,2562,10242,40962,163842]:
            self.surf_status = "PARAMETERIZED_SURFACE"
        else:
            self.surf_status =  "NON_PARAMETERIZED_SURFACE"
    

    def updateNeighVertex(self):
        # numba does not support list(set()), so use list(list()) method
        neigh_orders = []
        for i in range(self.num_vertices):
            neigh_orders.append([i])      # must assign a value, otherwise, numba jit will throw error
        for i in range(self.num_faces):
            face = self.faces[i]
            if face[1] not in neigh_orders[face[0]]:
                neigh_orders[face[0]].append(face[1])
            if face[2] not in neigh_orders[face[0]]:
                neigh_orders[face[0]].append(face[2])
            if face[0] not in neigh_orders[face[1]]:
                neigh_orders[face[1]].append(face[0])
            if face[2] not in neigh_orders[face[1]]:
                neigh_orders[face[1]].append(face[2])
            if face[0] not in neigh_orders[face[2]]:    
                neigh_orders[face[2]].append(face[0])
            if face[1] not in neigh_orders[face[2]]:
                neigh_orders[face[2]].append(face[1])
        
        self.n_neighs_per_vertex = np.zeros((self.num_vertices, 1), dtype=np.int32) 
        for i in range(self.num_vertices):
            self.n_neighs_per_vertex[i] = len(neigh_orders[i]) - 1
        
        self.MAX_NUM_NEIGHBORS = np.max(self.n_neighs_per_vertex)
        self.neigh_vertex = np.zeros((self.num_vertices, self.MAX_NUM_NEIGHBORS), dtype=np.int32) + self.num_vertices
        for i in range(self.num_vertices):
            self.neigh_vertex[i, 0:self.n_neighs_per_vertex[i,0]] = np.array(neigh_orders[i][1:])
        self.neigh_vertex_1d = self.neigh_vertex.flatten()  # jit only support 1d array index


    def updateNeighVertexWithRingSize(self, neigh_ring_size=1):
        # use n ring neighbor hood for computing mean curvature
        self.neigh_vertex_nring = self.neigh_vertex
        self.MAX_NUM_NEIGHBORS_NRING = self.MAX_NUM_NEIGHBORS
        self.n_neighs_nring = np.zeros_like(self.n_neighs_per_vertex)
        self.n_neighs_nring[:] = self.n_neighs_per_vertex[:]
        for i in range(neigh_ring_size-1):
            neigh_vertex_nring_1d = self.neigh_vertex_nring.flatten()
            neigh_vertex_nring_append = np.concatenate((self.neigh_vertex_nring, 
                                                        np.zeros((1, self.neigh_vertex_nring.shape[1]), dtype=np.int32)+self.num_vertices),
                                                        axis=0)
            neigh_vertex_nring = np.reshape(neigh_vertex_nring_append[neigh_vertex_nring_1d], 
                                            (self.num_vertices, self.MAX_NUM_NEIGHBORS_NRING, self.MAX_NUM_NEIGHBORS_NRING))
            
            for j in range(self.num_vertices):
                row = neigh_vertex_nring[j]
                row_unique = np.unique(row.flatten())
                ind = np.where(row_unique == j)[0]
                tmp = np.delete(row_unique, ind)  # remove center j-th vertex
                if j == 0:
                    neigh_vertex_nring_list = [tmp]
                else:
                    neigh_vertex_nring_list.append(tmp)
                self.n_neighs_nring[j] = len(tmp)
            
            self.MAX_NUM_NEIGHBORS_NRING = np.max(self.n_neighs_nring)
            self.neigh_vertex_nring = np.zeros((self.num_vertices, self.MAX_NUM_NEIGHBORS_NRING), dtype=np.int32) + self.num_vertices
            for j in range(self.num_vertices):
                self.neigh_vertex_nring[j, 0:self.n_neighs_nring[j,0]] = neigh_vertex_nring_list[j]
          
            
    def updateNeighFaces(self):
        # compute and store vertex-face connectivity
        vertex_has_faces = []
        for j in range(self.num_vertices):
            vertex_has_faces.append([j])
        for j in range(self.num_faces):
            face = self.faces[j]
            vertex_has_faces[face[0]].append(j)
            vertex_has_faces[face[1]].append(j)
            vertex_has_faces[face[2]].append(j)
            
        self.n_faces_per_vertex = np.zeros((self.num_vertices,), dtype=np.int32)
        for i in range(self.num_vertices):
            self.n_faces_per_vertex[i] = len(vertex_has_faces[i]) - 1
       
        self.MAX_NUM_FACES = np.max(self.n_faces_per_vertex)
        self.neigh_faces = np.zeros((self.num_vertices, self.MAX_NUM_FACES), dtype=np.int32) + self.num_faces
        for i in range(self.num_vertices):
            self.neigh_faces[i, 0:self.n_faces_per_vertex[i]] = np.array(vertex_has_faces[i][1:])


    def updateVerticesAppend(self):
        self.vertices_append = np.concatenate((self.vertices, np.array([[0,0,0]])), axis=0)

    def getNeighsOrder(self):
        return self.neigh_vertex
    

    def centralize(self):
        self.vertices = self.vertices - self.vertices.sum(0) / self.num_vertices


    def scaleBrainBasedOnArea(self):
        curr_area = sprop.computeFaceArea(self.vertices, self.faces)
        scale = np.sqrt(self.orig_face_area.sum() / curr_area.sum())
        # print("scale: ", scale)
        self.scaleBrain(scale)
        return scale
    
    def scaleBrain(self, scale):
        self.vertices = self.vertices * scale
        
    
    def setZeroNeighs(self, mag, n_neighs_per_vertex=None):
        if n_neighs_per_vertex == None:
            n_neighs_per_vertex = self.n_neighs_per_vertex
        for i in range(self.num_vertices):
            mag[i, n_neighs_per_vertex[i,0]:] = 0
        return mag

    def clearGradient(self):
        self.gradient = np.zeros((self.num_vertices, 3), dtype=np.float64)

    def clearMomentum(self):
        self.momentum_grad = np.zeros((self.num_vertices, 3), dtype=np.float64)


    def momentumTimeStep(self, max_grad=1.0, delta=0.85, momentum=0.9):
        gradient_tmp = self.gradient * delta + self.momentum_grad * momentum
        
        mag = sprop.npLinalgNorm(gradient_tmp, axis=1)
        if mag.max() > max_grad*4:
            raise NotImplementedError('Error: max gradient magnitude is too large. The surface may not be correctly reconstructed.')
        if mag.max() > max_grad:
            print('Warning: mag.max() =', mag.max())
            tmp_index = np.where(mag > max_grad)
            print(tmp_index[0].shape[0], "vertices has larger grad, truncated to max_grad")
            gradient_tmp[tmp_index] = gradient_tmp[tmp_index] / np.expand_dims((mag[tmp_index] / max_grad), axis=1) * 0.5
            
        self.gradient = gradient_tmp
        self.momentum_grad = self.gradient
        
        return mag.max()
      

    def applyGradient(self):
        self.vertices = self.vertices + self.gradient
        
        
    def normalizeCurvature(self, which_norm=None):
        if which_norm == None or which_norm == 'NORM_MEDIAN':
            self.curv = (self.curv -  np.median(self.curv))/np.sqrt(np.mean(np.square((self.curv - np.median(self.curv)))))
        elif which_norm == 'NORM_MEAN':
            self.curv = (self.curv -  np.mean(self.curv))/np.std(self.curv)
        else:
            raise NotImplementedError('NotImplementedError')
    

    def averageCurvature(self, n_averages=3):
        curv = self.curv
        for i in range(n_averages):
            curv_append = np.append(curv, [0])
            curv = (np.reshape(curv_append[self.neigh_vertex_1d], (self.num_vertices, self.MAX_NUM_NEIGHBORS)).sum(1) + curv) \
                / (self.n_neighs_per_vertex[:,0] + 1)
        self.curv = curv
      
      
    def averageGradient(self, n_averages=3, grad=None):
        if grad == None:
            grad = self.gradient
        for i in range(n_averages):
            grad_append = np.concatenate((grad, np.array([[0,0,0]])), axis=0)
            grad = np.reshape(grad_append[self.neigh_vertex_1d], 
                              (self.num_vertices, self.MAX_NUM_NEIGHBORS, int32(3))).sum(1) + grad
            grad = grad / (self.n_neighs_per_vertex + 1)
        if grad == None:
            self.gradient = grad
        else:
            return grad
       


# mySurf = Surface(surf['vertices'].astype(np.float64), surf['faces'].astype(np.int32))
# print("num_vertices ", mySurf.num_vertices)
# print("num_faces ", mySurf.num_faces)
# print("surf_status ", mySurf.surf_status)
# print("vertices ", mySurf.vertices)
# print("faces ", mySurf.faces)
# print("neigh_vertex ", mySurf.neigh_vertex)
# print("n_neighs_per_vertex ", mySurf.n_neighs_per_vertex)
# print("MAX_NUM_NEIGHBORS ", mySurf.MAX_NUM_NEIGHBORS)

