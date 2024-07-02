#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:34:23 2023

@author: Fenqiang Zhao
"""

import numpy as np
import SimpleITK as sitk
import os
import sys


#Read and Write NIfTI using SimpleITK by preserving header information
inputImage = sitk.ReadImage('/path/to/input.nii.gz')
# get result in the form of a numpy array
npa_res = my_algorithm(sitk.GetArrayFromImage(inputImage)) # my_algorithm does something fancy
# Converting back to SimpleITK (assumes we didn't move the image in space as we copy the information from the original)
result_image = sitk.GetImageFromArray(npa_res)
result_image.CopyInformation(inputImage)
# write the image
sitk.WriteImage(result_image, '/path/to/result.nii.gz')
    

# def combineFSParcLabel(lbl_img):
    