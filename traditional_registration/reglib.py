import os
import ctypes
import csv

import numpy as np


def load_library(path: str = os.getcwd(), name: str = "libregistration_pcl110") -> None:
    global REGLIB
    try:
        REGLIB = np.ctypeslib.load_library(libname=name, loader_path=path)
        print(REGLIB)
    except OSError:
        print("Compiled C++ library was not found in the current directory. Please use `load_library` to load it from "
              "a custom directory, then ignore this message.")


load_library(f'/home/massimo/vehicle_infrastructure_calibration/traditional_registration')


def load_data(path: str, delimiter: str = ' ') -> np.ndarray:
    """Loads point cloud data of type `CSV`, `PLY` and `PCD`.

    The file should contain one point per line where each number is separated by the `delimiter` character.
    Any none numeric lines will be skipped.

    Args:
        path (str): The path to the file.
        delimiter (char): Separation of numbers in each line of the file.

    Returns:
        A ndarray of shape NxD where `N` are the number of points in the point cloud and `D` their dimension.
    """
    data = list()
    with open(path, newline='\n') as file:
        reader = csv.reader(file, delimiter=delimiter, quoting=csv.QUOTE_NONNUMERIC)
        lines = 0
        skips = 0
        while True:
            try:
                row = next(reader)
                row = [x for x in row if not isinstance(x, str)]
                if len(row) in [3, 6, 9]:
                    data.append(row[:3])
                else:
                    skips += 1
            except ValueError:
                skips += 1
                pass
            except StopIteration:
                print(f"Found {lines} lines. Skipped {skips}. Loaded {lines - skips} points.")
                break
            lines += 1
    return np.array(data)


def set_argtypes(algorithm, source, target):
    """Tells the underlying C++ code which data types and dimensions to expect.

    Args:
        algorithm (str): The registration algorithm to use. One of `icp` or `ndt`.
        source (ndarray): The source point cloud.
        target (ndarray): The target point cloud.
    """
    REGLIB.icp.restype = ctypes.c_double
    REGLIB.ndt.restype = ctypes.c_double
    argtypes = [np.ctypeslib.ndpointer(dtype=np.float64, ndim=source.ndim, shape=source.shape,
                                       flags='C_CONTIGUOUS'), ctypes.c_size_t,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=target.ndim, shape=target.shape,
                                       flags='C_CONTIGUOUS'), ctypes.c_size_t,
                np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, shape=(4, 4), flags='C_CONTIGUOUS'),
                ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_bool]
    if algorithm == 'icp':
        REGLIB.icp.argtypes = argtypes
    elif algorithm == 'ndt':
        argtypes.extend([ctypes.c_float, ctypes.c_double, ctypes.c_float])
        REGLIB.ndt.argtypes = argtypes


def icp(source,
        target,
        nr_iterations=25,
        distance_threshold=1.0,
        epsilon=0.01,
        inlier_threshold=0.05,
        downsample=0,
        visualize=False):
    """The `Iterative Closest Point` (ICP) algorithm.

    Args:
        source (ndarray): The point cloud that we want to align to the target.
        target (ndarray): The point cloud that the source is aligned to.
        nr_iterations (int): The maximum number of iterations the internal optimization should run for.
        distance_threshold (float): The maximum distance threshold between two correspondent points in
                                    source -> target. If the distance is larger than this threshold, the points will
                                    be ignored in the alignment process.
        epsilon (float): The transformation epsilon (maximum allowable difference between two consecutive
                 transformations) in order for an optimization to be considered as having converged to the final
                 solution.
        inlier_threshold (float): The inlier distance threshold for the internal RANSAC outlier rejection loop.
                          The method considers a point to be an inlier, if the distance between the target data
                          index and the transformed source index is smaller than the given inlier distance
                          threshold.
        downsample (float): Assembles a local 3D grid over a given PointCloud and downsamples + filters the data.
        visualize (bool): Can be used to visualize and control the progress of the algorithm.

    Returns:
        A ndarray with the final transformation matrix between source and target.
    """

    transformation = np.identity(4)
    set_argtypes('icp', source, target)
    score = REGLIB.icp(source, len(source), target, len(target), transformation,
                       nr_iterations, distance_threshold, epsilon, inlier_threshold, downsample, visualize)
    print(f"ICP converged. Fitness score: {score:.2f}") if score > 0 else print("ICP did not converge!")
    return transformation


def ndt(source,
        target,
        nr_iterations=25,
        distance_threshold=1.0,
        epsilon=0.01,
        inlier_threshold=0.05,
        downsample=0,
        visualize=False,
        resolution=1.0,
        step_size=0.1,
        voxelize=0):
    """The `Normal Distributions Transform` (NDT) algorithm.

    Args:
        source (ndarray): The point cloud that we want to align to the target.
        target (ndarray): The point cloud that the source is aligned to.
        nr_iterations (int): The maximum number of iterations the internal optimization should run for.
        distance_threshold (float): The maximum distance threshold between two correspondent points in
                                    source -> target. If the distance is larger than this threshold, the points will
                                    be ignored in the alignment process.
        epsilon (float): The transformation epsilon (maximum allowable difference between two consecutive
                 transformations) in order for an optimization to be considered as having converged to the final
                 solution.
        inlier_threshold (float): The inlier distance threshold for the internal RANSAC outlier rejection loop.
                          The method considers a point to be an inlier, if the distance between the target data
                          index and the transformed source index is smaller than the given inlier distance
                          threshold.
        downsample (float): Assembles a local 3D grid over a given PointCloud and downsamples + filters the data.
        visualize (bool): Can be used to visualize and control the progress of the algorithm.
        resolution (float): The resolution of the voxel grid. Try increasing this in case of core dumps.
        step_size (float): The Newton line search maximum step length.
        voxelize (bool): If set to `True`, the source cloud is converted into a voxel model before alignment.

    Returns:
        A ndarray with the final transformation matrix between source and target.
    """

    transformation = np.identity(4)
    set_argtypes('ndt', source, target)
    score = REGLIB.ndt(source, len(source), target, len(target), transformation,
                  nr_iterations, distance_threshold, epsilon, inlier_threshold, downsample, visualize,
                  resolution, step_size, voxelize)
    print(f"NDT converged. Fitness score: {score:.2f}") if score > 0 else print("NDT did not converge!")
    return transformation
