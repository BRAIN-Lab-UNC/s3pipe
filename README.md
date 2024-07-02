# [s3pipeline](https://pypi.org/project/s3pipe/)
Python-based spherical cortical surface processing tools, including spherical mapping, resampling, interpolation, registration, parcellation, etc. It provides fast and accurate cortical surface-based data analysis using deep learning techniques.

## Install

It can be installed from PyPI using:
```
pip install s3pipe
```
or using conda to directly create an environment from environment.yml

```
conda env create -f environment.yml
conda activate s3env
```
Then install pytorch from https://pytorch.org/get-started/locally/ with correct gpu/cpu and cuda choices.

## Main tools
[**I/O vtk file**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/main/s3pipe/utils/vtk.py). Python function for reading and writing .vtk surface file. Example code:
```
from s3pipe.utils.vtk import read_vtk, write_vtk

surf = read_vtk(file_name)
# Some operations to the surface 
write_vtk(surf, new_file_name)
```
For matlab users, please refer to this [issue](https://github.com/zhaofenqiang/Spherical_U-Net/issues/3#issuecomment-763334969) and this [repository](https://github.com/Zhengwang-Wu/CorticalSurfaceMetric).

[**Layers**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/main/s3pipe/models/layers.py) provide basic spherical convolution, pooling, upsampling layers for constructing spherical convolutional neural networks.

[**Models**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/main/s3pipe/models) provide some baseline spherical convolutional neural networks, e.g., [Spherical U-Net](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/85aa03ffc7a153de217e925c6f522e4614e619bd/s3pipe/models/models.py#L133), Spherical SegNet, Spherical VGG, Spherical ResNet, Spherical CycleGAN, etc.

[**Resample feature**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/85aa03ffc7a153de217e925c6f522e4614e619bd/s3pipe/utils/interp_numpy.py#L217) on spherical surface to standard icosahedron subdivision spheres. Example code:
```
from s3pipe.utils.interp_numpy import resampleSphereSurf
from s3pipe.utils.vtk import read_vtk, write_vtk
from s3pipe.utils.utils import get_sphere_template

template_163842 = get_sphere_template(163842)
data = read_vtk(file)
resampled_feat = resampleSphereSurf(data['vertices'], template_163842['vertices'], 
                                    np.concatenate((data['sulc'][:,np.newaxis], data['curv'][:,np.newaxis]), axis=1))
surf = {'vertices': template_163842['vertices'], 
         'faces': template_163842['faces'],
         'sulc': resampled_feat[:,0],
         'curv': resampled_feat[:,1]}
write_vtk(surf, file.replace('.vtk', '.resample.vtk'))
```
Note if you want to run it on GPU, change `interp_numpy` to `interp_torch`.

[**Resample label**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/85aa03ffc7a153de217e925c6f522e4614e619bd/s3pipe/utils/interp_numpy.py#L446) on spherical surface to standard icosahedron subdivision spheres. Example code:
```
from s3pipe.utils.vtk import read_vtk, write_vtk,
from s3pipe.utils.interp_numpy import resample_label
from s3pipe.utils.utils import get_sphere_template

template_163842 = get_sphere_template(163842) 
surf = read_vtk(file)
resampled_label = resample_label(surf['vertices'], template_163842['vertices'], surf['par_fs_vec'])
```
[**Cortical surface parcellation**](https://github.com/zhaofenqiang/Spherical_U-Net) with trained models based on this package.

[**Cortical surface registration**](https://github.com/BRAIN-Lab-UNC/S3Reg) with trained models based on this package.

[**Rigid surface alignment**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/85aa03ffc7a153de217e925c6f522e4614e619bd/s3pipe/surface/s3reg.py#L50). An example code can be found [here]().

[**Spherical surfae mapping**](https://github.com/BRAIN-Lab-UNC/S3Map) with trained models based on this package.

[**Surface inflation**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/main/s3pipe/surface/inflate.py).

[**Surface and feature smooth**](https://github.com/BRAIN-Lab-UNC/s3pipe/blob/85aa03ffc7a153de217e925c6f522e4614e619bd/s3pipe/surface/prop.py#L17).

[**Chcek folded triangles**]().



## Papers

If you use this code, please cite:

Fenqiang Zhao, et.al. [Spherical U-Net on Cortical Surfaces: Methods and Applications](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_67). Information Processing in Medical Imaging (IPMI), 2019.

Fenqiang Zhao, et.al. [Spherical Deformable U-Net: Application to Cortical Surface Parcellation and Development Prediction](https://ieeexplore.ieee.org/document/9316936). IEEE Transactions on Medical Imaging, 2021.

Fenqiang Zhao, et.al. [S3Reg: Superfast Spherical Surface Registration Based on Deep Learning](https://ieeexplore.ieee.org/document/9389746). IEEE Transactions on Medical Imaging, 2021.

Fenqiang Zhao, et.al. [Fast spherical mapping of cortical surface meshes using deep unsupervised learning](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_16). MICCAI 2022.