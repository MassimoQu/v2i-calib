import os
import time
import sys
sys.path.append('./reader')
from CooperativeReader import CooperativeReader
from extrinsic_utils import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true
import reglib


def main():
    # Only needed if you want to use manually compiled library code
    # reglib.load_library(os.path.join(os.curdir, "cmake-build-debug"))

    # Load you data
    # source_points = reglib.load_data(os.path.join(os.curdir, "files", "model_points.pcd"))
    # target_points = reglib.load_data(os.path.join(os.curdir, "files", "scene_points.pcd"))

    # source_points = reglib.load_data(os.path.join(os.curdir, "files", "000289.pcd"))
    # target_points = reglib.load_data(os.path.join(os.curdir, "files", "007489.pcd"))

    reader = CooperativeReader('007489', '000289')

    infra_pointcloud, vehicle_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()
    T_true = reader.get_cooperative_T_i2v()

    # Run the registration algorithm
    start = time.time()
    # trans = reglib.icp(source=infra_pointcloud, target=vehicle_pointcloud, nr_iterations=1, epsilon=0.01,
    #                    inlier_threshold=0.05, distance_threshold=5.0, downsample=0, visualize=True)
                       #resolution=12.0, step_size=0.5, voxelize=0)
    trans = reglib.ndt(source=infra_pointcloud, target=vehicle_pointcloud)
    print("Runtime:", time.time() - start)
    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(trans), convert_T_to_6DOF(T_true))
    print('RE:' , RE)
    print('TE:' , TE)


if __name__ == "__main__":
    main()

