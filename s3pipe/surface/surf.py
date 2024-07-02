#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 16:12:22 2023

@author: Fenqiang Zhao
"""
import numpy as np

import s3pipe.surface.prop as sprop
from s3pipe.utils.utils import get_neighs_order
 

class Surface(object):
    def __init__(self, vertices, faces):
        """
        Surface class containing surface properties

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
        
        self.neigh_vertex_mat = np.zeros((1,1), dtype=np.int32)
        self.neigh_vertex_1d = np.zeros((1,), dtype=np.int32)
        # self.neigh_vertex_list = None
        self.n_neighs_per_vertex = np.zeros((1,1), dtype=np.int32)
        self.MAX_NUM_NEIGHBORS = 0
        # self.property_dic = {'thickness': 0}
      
        self.neigh_faces_mat = np.zeros((1,1), dtype=np.int32)
        self.neigh_faces_list = np.zeros((1,1), dtype=np.int32)
        self.n_faces_per_vertex = np.zeros((1,), dtype=np.int32)
        self.MAX_NUM_FACES = 0
        
        self.vertex_normal = np.zeros((1,1), dtype=np.float64)
        self.orig_metrics = np.zeros((1,1), dtype=np.float64)
        self.orig_face_area = np.zeros((1,), dtype=np.float64)
        self.orig_vertex_area = np.zeros((1,), dtype=np.float64)
        self.sulc = np.zeros((1,), dtype=np.float64)
        self.curv = np.zeros((1,), dtype=np.float64)
        self.curr_metrics = np.zeros((1,1), dtype=np.float64)
        self.gradient = np.zeros((1,1), dtype=np.float64)
        self.momentum_grad = np.zeros((1,1), dtype=np.float64)
        
        self.updateSurfaceBasicProperties()
        self.updateNeighVertex()
        self.updateNeighFaces()
 
        # storing n_ring neighborhood vertex index
        # self.neigh_vertex_nring = np.zeros((1,1), dtype=np.int32)
        # self.n_neighs_nring = np.zeros((1,1), dtype=np.int32)
        # self.MAX_NUM_NEIGHBORS_NRING = 0
        
        
    def updateSurfaceBasicProperties(self):
        self.num_vertices = len(self.vertices)
        self.num_faces = len(self.faces)
        if self.num_vertices in [12,42,162,642,2562,10242,40962,163842] and (self.num_vertices*2-4)==self.num_faces:
            self.surf_status = "ICOSAHEDRON_SURFACE"
        else:
            self.surf_status = "NON_PARAMETERIZED_SURFACE"
    

    def updateNeighVertex(self):
        if self.surf_status == "ICOSAHEDRON_SURFACE":
            self.MAX_NUM_NEIGHBORS = 6
            tmp = get_neighs_order(self.num_vertices)
            self.neigh_vertex_mat = tmp[:, :-1]
            self.neigh_sorted_orders = np.concatenate((np.arange(self.num_vertices)[:, np.newaxis], self.neigh_vertex_mat[:, 0:6]), axis=1)
            self.neigh_vertex_mat[0:12, -1] = self.num_vertices
            self.n_neighs_per_vertex = np.zeros((self.num_vertices, 1), dtype=np.int32) + 6
            self.n_neighs_per_vertex[0:12,:] = 5
        else:
            neigh_orders = get_neighs_order(self.num_vertices, self.faces)
            self.n_neighs_per_vertex = np.zeros((self.num_vertices, 1), dtype=np.int32) 
            for i in range(self.num_vertices):
                self.n_neighs_per_vertex[i] = len(neigh_orders[i])
            self.MAX_NUM_NEIGHBORS = np.max(self.n_neighs_per_vertex)
            self.neigh_vertex_mat = np.zeros((self.num_vertices, self.MAX_NUM_NEIGHBORS), dtype=np.int32) + self.num_vertices
            for i in range(self.num_vertices):
                self.neigh_vertex_mat[i, 0:self.n_neighs_per_vertex[i,0]] = np.asarray(neigh_orders[i])
            self.neigh_vertex_1d = self.neigh_vertex_mat.flatten()  # jit only support 1d array index
            # self.neigh_vertex_list = neigh_orders

            
    def updateNeighFaces(self):
        # compute and store vertex-face connectivity
        vertex_has_faces = []
        for j in range(self.num_vertices):
            vertex_has_faces.append([])
        for j in range(self.num_faces):
            face = self.faces[j]
            vertex_has_faces[face[0]].append(j)
            vertex_has_faces[face[1]].append(j)
            vertex_has_faces[face[2]].append(j)
            
        self.n_faces_per_vertex = np.zeros((self.num_vertices,), dtype=np.int32)
        for i in range(self.num_vertices):
            self.n_faces_per_vertex[i] = len(vertex_has_faces[i])
       
        self.MAX_NUM_FACES = np.max(self.n_faces_per_vertex)
        self.neigh_faces_mat = np.zeros((self.num_vertices, self.MAX_NUM_FACES), dtype=np.int32) + self.num_faces
        for i in range(self.num_vertices):
            self.neigh_faces_mat[i, 0:self.n_faces_per_vertex[i]] = np.array(vertex_has_faces[i][0:])

        self.neigh_faces_list = vertex_has_faces
        

    def updateVerticesAppend(self):
        self.vertices_append = np.concatenate((self.vertices, np.array([[0,0,0]])), axis=0)

    def getNeighsOrder(self):
        return self.neigh_vertex
    

    def centralize(self):
        self.vertices = self.vertices - self.vertices.sum(0) / self.num_vertices


    def scaleBrainBasedOnArea(self):
        curr_area = sprop.computeFaceArea(self.vertices, self.faces)
        scale = np.sqrt(self.orig_face_area.sum() / curr_area.sum())
        self.vertices = self.vertices * scale
        return scale
    
    
    def setZeroNeighs(self, mag, n_neighs_per_vertex='None'):
        if n_neighs_per_vertex == 'None':
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
        
        mag = np.linalg.norm(gradient_tmp, axis=1)
        if mag.max() > max_grad*4:
            print('mag.max()=', mag.max())
            raise NotImplementedError('Error: max gradient is too large. The surface may not be correctly reconstructed.')
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
        if which_norm is None or which_norm == 'NORM_MEDIAN':
            self.curv = (self.curv -  np.median(self.curv))/np.sqrt(np.mean(np.square((self.curv - np.median(self.curv)))))
        elif which_norm == 'NORM_MEAN':
            self.curv = (self.curv -  np.mean(self.curv))/np.std(self.curv)
        else:
            raise NotImplementedError('NotImplementedError')
    

    def averageCurvature(self, n_averages=3):
        curv = self.curv
        for i in range(n_averages):
            curv_append = np.append(curv, [0])
            curv = (curv_append[self.neigh_vertex_mat].sum(1) + curv) \
                / (self.n_neighs_per_vertex[:,0] + 1)
        self.curv = curv
      
      
    def averageGradient(self, n_averages=4, grad_out=None):
        if grad_out is None:
            grad = self.gradient
        else:
            grad = grad_out
        for i in range(n_averages):
            grad_append = np.concatenate((grad, np.array([[0,0,0]])), axis=0)
            grad = (grad_append[self.neigh_vertex_mat].sum(1) + grad) \
                / (self.n_neighs_per_vertex + 1)
        if grad_out is None:
            self.gradient = grad
        else:
            return grad


    # def updateNeighVertexWithRingSize(self, neigh_ring_size=1):
    #     # use n ring neighbor hood for computing mean curvature
    #     self.neigh_vertex_nring = self.neigh_vertex
    #     self.MAX_NUM_NEIGHBORS_NRING = self.MAX_NUM_NEIGHBORS
    #     self.n_neighs_nring = np.zeros_like(self.n_neighs_per_vertex)
    #     self.n_neighs_nring[:] = self.n_neighs_per_vertex[:]
    #     for i in range(neigh_ring_size-1):
    #         neigh_vertex_nring_1d = self.neigh_vertex_nring.flatten()
    #         neigh_vertex_nring_append = np.concatenate((self.neigh_vertex_nring, 
    #                                                     np.zeros((1, self.neigh_vertex_nring.shape[1]), dtype=np.int32)+self.num_vertices),
    #                                                     axis=0)
    #         neigh_vertex_nring = np.reshape(neigh_vertex_nring_append[neigh_vertex_nring_1d], 
    #                                         (self.num_vertices, self.MAX_NUM_NEIGHBORS_NRING, self.MAX_NUM_NEIGHBORS_NRING))
            
    #         for j in range(self.num_vertices):
    #             row = neigh_vertex_nring[j]
    #             row_unique = np.unique(row.flatten())
    #             ind = np.where(row_unique == j)[0]
    #             tmp = np.delete(row_unique, ind)  # remove center j-th vertex
    #             if j == 0:
    #                 neigh_vertex_nring_list = [tmp]
    #             else:
    #                 neigh_vertex_nring_list.append(tmp)
    #             self.n_neighs_nring[j] = len(tmp)
            
    #         self.MAX_NUM_NEIGHBORS_NRING = np.max(self.n_neighs_nring)
    #         self.neigh_vertex_nring = np.zeros((self.num_vertices, self.MAX_NUM_NEIGHBORS_NRING), dtype=np.int32) + self.num_vertices
    #         for j in range(self.num_vertices):
    #             self.neigh_vertex_nring[j, 0:self.n_neighs_nring[j,0]] = neigh_vertex_nring_list[j]