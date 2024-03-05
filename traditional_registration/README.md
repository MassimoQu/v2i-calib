# Reliable and fast Point Cloud registration in Python
This repository implements a lightweight Python wrapper around two registration
algorithms from the [Point Cloud Library](https://pointcloudlibrary.github.io/) with minimal dependencies due to
reliance on the Python standard library and the ubiquitous Numpy.

The approach is reasonable due to the need to embed fast and reliable registration
capabilities in existing Python projects which is not given at the moment,
as existing pure Python implementations are both too slow and poorly maintained.

The following registration algorithms are supported so far:
1. Iterative Closest Point (ICP)
2. Normal Distribution Transform (NDT)

## Installation
In order to use the module with the pre-compiled library, you need to have
Python 3 and Numpy installed.

As this code is only a thin wrapper around some of PCLs functionality, you will also have to install PCL itself.
On Ubuntu, use
```
sudo apt install libpcl-dev
```
Please check out the official PCL website linked above for installation
instructions on other architectures.

If you would like to make changes to the C++ code and compile it yourself you will
additionally need to install the g++/c++ compiler and CMake.

## Usage
The most straight forward way to use this module is to rely on the pre-compiled
library. Simply import the `reglib` module, load your data and run the registration
algorithm. To get started have a look at `test.py`.

## Functionality
So far, the module supports:
* Loading of CSV, PLY and PCD 3D point data.
* Uniform subsampling of the input data.
* Registration using either the ICP or NDT algorithm. Both return the 4x4
transformation matrix between the source and target point clouds in homogeneous
coordinates. 
* Visualization and interactive execution.
