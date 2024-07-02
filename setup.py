import setuptools
import glob
neigh_indices_files = glob.glob('s3pipe/utils/neigh_indices/*')
 
setuptools.setup(
    name="s3pipe",
    version="1.3.0",
    author="Fenqiang Zhao",
    author_email="zhaofenqiang0221@gmail.com",
    description="This is a pipeline for superfast spherical surface processing",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/BRAIN-Lab-UNC/s3pipe",
    packages=setuptools.find_packages(),
    license="CC BY-NC-SA 4.0",
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy'
    ],
    include_package_data=True,
)

