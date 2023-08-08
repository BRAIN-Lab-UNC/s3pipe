import setuptools
import glob
neigh_indices_files = glob.glob('s3pipe/utils/neigh_indices/*')

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="s3pipe",
    version="1.2.0",
    author="Fenqiang Zhao",
    author_email="zhaofenqiang0221@gmail.com",
    description="This is the tools for superfast spherical surface processing",
    long_description="long_description",
    long_description_content_type="text/markdown",
    url="https://github.com/BRAIN-Lab-UNC/S3pipe",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'scipy'
    ],
    include_package_data=True,
)

