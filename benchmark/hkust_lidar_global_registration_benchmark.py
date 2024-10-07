import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import teaserpp_python
import copy
import argparse
from time import perf_counter
from config.config import cfg, cfg_from_yaml_file, Logger
from v2x_calib.utils import get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader
from v2x_calib import V2XSim_Reader

def create_point_cloud(points, color=[0, 0.651, 0.929]):
    # 1.000, 0.706, 0.000
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)
    return pcd

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, radius_normal, radius_feature):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1

def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_teaser_solver(noise_bound, rotation_estimation_algorithm=teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = rotation_estimation_algorithm
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def fpfh_teaser(inf_pc: np.ndarray, veh_pc: np.ndarray, visualize=False):
    inf_pc = create_point_cloud(inf_pc, color=[0, 0.651, 0.929])
    veh_pc = create_point_cloud(veh_pc, color=[1.000, 0.706, 0.000])

    inf_pc = inf_pc.voxel_down_sample(voxel_size=cfg.infra.fpfh.voxel_size)
    veh_pc = veh_pc.voxel_down_sample(voxel_size=cfg.vehicle.fpfh.voxel_size)

    A_xyz = pcd2xyz(inf_pc)  # np array of size 3 by N
    B_xyz = pcd2xyz(veh_pc)  # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(inf_pc, cfg.infra.fpfh.radius_normal, cfg.infra.fpfh.radius_feature)
    B_feats = extract_fpfh(veh_pc, cfg.vehicle.fpfh.radius_normal, cfg.vehicle.fpfh.radius_feature)

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

    if cfg.rotation_estimation_algorithm == "GNC_TLS":
        rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    elif cfg.rotation_estimation_algorithm == "FGR":
        rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.FGR
    elif cfg.rotation_estimation_algorithm == "QUATRO":
        rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.QUATRO

    teaser_solver = get_teaser_solver(cfg.teaser.noise_bound, rotation_estimation_algorithm)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser, t_teaser)

    if visualize:
        A_pcd_T_teaser = copy.deepcopy(inf_pc).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser, veh_pc], window_name="Registration by Teaser++")

    return T_teaser


def numpy_to_point_cloud(points):
    """
    Convert a numpy array to an Open3D PointCloud object.
    
    Parameters:
        points (np.ndarray): A Nx3 numpy array of xyz positions.
    
    Returns:
        o3d.geometry.PointCloud: An Open3D point cloud object.
    """
    # 创建一个空的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()

    # 将 numpy array 中的数据转换成 Open3D 所需的格式
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/hkust_lidar_global_config.yaml')
    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    logger = Logger(f"{cfg.data.type}_demo_{cfg.rotation_estimation_algorithm}_fpfh__{cfg.data.data_info_path.split('/')[-1].split('.')[0]}")

    if cfg.data.type == "V2X-Sim":
        wrapper = V2XSim_Reader().generate_vehicle_vehicle_pointcloud()
    elif cfg.data.type == "DAIR-V2X":
        wrapper = CooperativeBatchingReader(path_data_info=cfg.data.data_info_path).generate_infra_vehicle_pointcloud()

    for inf_id, veh_id, inf_pc, veh_pc, T_true in wrapper:
        t1 = perf_counter()
        T = fpfh_teaser(inf_pc, veh_pc, False)
        t2 = perf_counter()
        
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T), convert_T_to_6DOF(T_true))
        ids_str = f"inf_id: {inf_id}, veh_id: {veh_id}" if cfg.data.type == "DAIR-V2X" else f"frame_id: {inf_id}, cav_id: {veh_id}"
        eval_str = f" RE: {RE}, TE: {TE}, time: {t2 - t1}"
        logger.info(ids_str + eval_str)
        # logger.info('-------------------')

        # pcd_inf_T_converted = numpy_to_point_cloud(inf_pc)
        # pcd_inf_trueT_converted = numpy_to_point_cloud(inf_pc)
        # pcd_veh = numpy_to_point_cloud(veh_pc)

        # pcd_inf_T_converted.paint_uniform_color([0, 0.651, 0.929])
        # pcd_inf_trueT_converted.paint_uniform_color([0, 0.651, 0.929])
        # pcd_veh.paint_uniform_color([1.000, 0.706, 0.000])

        # pcd_inf_T_converted.transform(T)
        # pcd_inf_trueT_converted.transform(T_true)
        # o3d.visualization.draw_geometries([pcd_inf_trueT_converted, pcd_veh], window_name=f"T_true registration ; inf_id: {inf_id}, veh_id: {veh_id}; RE: {RE}, TE: {TE}")
        # o3d.visualization.draw_geometries([pcd_inf_T_converted, pcd_veh], window_name="T_cal registration")

        # if input("Continue? (y/n)") == 'n':
        #     break

