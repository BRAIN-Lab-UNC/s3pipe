#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Created on Wed Nov 20 09:17:52 2019

@author: Fenqiang Zhao, https://github.com/zhaofenqiang

Contact: zhaofenqiang0221@gmail.com

"""
import os
import copy
import numpy as np
from pyvista import read as pyread
from pyvista import PolyData as pydata

abspath = os.path.abspath(os.path.dirname(__file__))


def read_vtk(in_file):
    """
    Read .vtk POLYDATA file
    
    in_file: string,  the filename
    Out: dictionary, 'vertices', 'faces', 'curv', 'sulc', ...
    """
    polydata = pyread(in_file)
    
    n_faces = polydata.n_faces
    vertices = np.array(polydata.points)  # get vertices coordinate
    
    # only for triangles polygons data
    faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
    assert len(faces)/4 == n_faces, "faces number is wrong!"
    faces = np.reshape(faces, (n_faces,4))
    
    data = {'vertices': vertices,
            'faces': faces
            }
    
    point_data = polydata.point_data
    for key, value in point_data.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif  value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data
    

def write_vtk(in_dic, file, binary=True, deep_copy=False):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    deep_copy: sometimes, e.g., if in_dic contains data read via nib, it may cause error 
        after write_vtk, so use deep_copy to copy the data first
    """
    if deep_copy:
        in_dic2 = copy.deepcopy(in_dic)
        in_dic = in_dic2
        
    surf = pydata(in_dic['vertices'], in_dic['faces'])
    for key, value in in_dic.items():
        if key == 'vertices' or key == 'faces':
            continue
        surf.point_data[key] = value

    surf.save(file, binary=binary)
     
    
def write_vertices(in_ver, file):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    file: string, output file name
    """
    
    with open(file,'a') as f:
        f.write("# vtk DataFile Version 4.2 \n")
        f.write("vtk output \n")
        f.write("ASCII \n")
        f.write("DATASET POLYDATA \n")
        f.write("POINTS " + str(len(in_ver)) + " float \n")
        np.savetxt(f, in_ver)
