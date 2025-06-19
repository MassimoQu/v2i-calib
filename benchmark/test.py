import itertools
import time
import os
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from point_cloud_registration import NDT

from v2x_calib.reader import CooperativeReader
from v2x_calib.utils import get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, convert_6DOF_to_T
from config.config import cfg, cfg_from_yaml_file, Logger
from v2x_calib.reader import CooperativeBatchingReader
from v2x_calib.reader import CooperativeReader
from v2x_calib.preprocess import Filter3dBoxes
from v2x_calib.reader import V2XSim_Reader

#！ 接口本身错误
def refine_with_ransac(source, target, voxel_size=0.5, ransac_n=4, max_iter=100):
    # 数据预处理
    # source_down = source.voxel_down_sample(voxel_size)
    # target_down = target.voxel_down_sample(voxel_size)
    source_down = source
    target_down = target
    
    # 特征提取（与RANSAC共用）
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))
    target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100))

    # RANSAC粗配准
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.5*voxel_size),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)
    ]
    
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size*2,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=ransac_n,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iter, 0.999)
    )
    
    return ransac_result.transformation

# 添加噪声到外参矩阵
def add_noise_to_extrinsic(extrinsic, rot_noise_deg=1.0, trans_noise_m=0.1):
    """
    对4x4外参矩阵添加旋转和平移噪声
    :param extrinsic: 原始外参矩阵 (4x4 numpy array)
    :param rot_noise_deg: 旋转噪声角度（度）
    :param trans_noise_m: 平移噪声距离（米）
    :return: 加噪后的外参矩阵
    """
    # 分解旋转和平移
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # 添加旋转噪声
    noise_rot = Rotation.from_euler('xyz', np.random.normal(0, rot_noise_deg, 3), degrees=True).as_matrix()
    R_noisy = R @ noise_rot  # 右乘扰动
    
    # 添加平移噪声
    t_noisy = t + np.random.normal(0, trans_noise_m, 3)
    
    # 重建加噪外参矩阵
    noisy_extrinsic = np.eye(4)
    noisy_extrinsic[:3, :3] = R_noisy
    noisy_extrinsic[:3, 3] = t_noisy
    return noisy_extrinsic

# 点云配准优化接口
def refine_extrinsic_ICP(source_pcd, target_pcd, initial_guess, method='point_to_plane', voxel_size = 0.1):
    """
    使用Open3D进行精配准
    :param source_pcd: Open3D点云对象(待配准)
    :param target_pcd: Open3D点云对象(目标)
    :param initial_guess: 初始变换矩阵 (4x4 numpy array)
    :param method: 配准方法，可选'point_to_point'/'point_to_plane'
    :return: 优化后的变换矩阵
    """
    # 下采样加速计算    
    source_down = source_pcd.voxel_down_sample(voxel_size)
    target_down = target_pcd.voxel_down_sample(voxel_size)
    
    # 计算法向量（用于point_to_plane）
    if method == 'point_to_plane':
        radius_normal = voxel_size * 2
        source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # 执行ICP配准
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
    if method == 'point_to_point':
        reg_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=0.2,
            init=initial_guess, criteria=criteria)
    else:
        reg_result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, max_correspondence_distance=0.2,
            init=initial_guess, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=criteria)
    
    return reg_result.transformation

#！ 接口本身错误
def refine_extrinsic_NDT(source, target, initial_guess, voxel_size=0.5, max_iter=50):
    ndt = NDT(max_iter=max_iter, voxel_size=voxel_size)
    ndt.set_target(target)
    return ndt.align(source=source, init_T=initial_guess)

# 评估误差
def compute_errors(estimated, ground_truth):
    """
    计算旋转和平移误差
    :return: (旋转误差[度], 平移误差[米])
    """
    # 分解旋转矩阵
    R_est = estimated[:3, :3]
    R_gt = ground_truth[:3, :3]
    
    # 计算旋转误差
    error_rot = Rotation.from_matrix(R_est.T @ R_gt).magnitude() * 180 / np.pi
    
    # 计算平移误差
    error_trans = np.linalg.norm(estimated[:3, 3] - ground_truth[:3, 3])
    return error_rot, error_trans


def get_processed_solo_result(cfg, inf_id, veh_id, rot_noise, trans_noise, method='point_to_plane', voxel_size=0.5):

    reader = CooperativeReader(infra_file_name=inf_id, vehicle_file_name=veh_id, data_folder=cfg.data.data_root_path)
    infra_pointcloud, vehicle_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()
    infra_pcd = o3d.geometry.PointCloud()
    vehicle_pcd = o3d.geometry.PointCloud()
    infra_pcd.points = o3d.utility.Vector3dVector(infra_pointcloud)
    vehicle_pcd.points = o3d.utility.Vector3dVector(vehicle_pointcloud)
    
    T_true = reader.get_cooperative_T_i2v()
    noisy_T = add_noise_to_extrinsic(T_true, rot_noise_deg=rot_noise, trans_noise_m=trans_noise)
    infra_pcd = infra_pcd.transform(noisy_T)
    
    start_time = time.time()
    if method == 'ndt':
        # 使用NDT方法进行配准
        # reg = o3d.pipelines.registration.registration_ndt(
        #     source=vehicle_pcd, target=infra_pcd,
        #     max_correspondence_distance=0.2,
        #     estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        #     criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
        # refined_T = reg.transformation
        try:
            refined_T = refine_extrinsic_NDT(infra_pointcloud, vehicle_pointcloud, noisy_T, voxel_size=voxel_size)
        except Exception as e:
            return f"Error during NDT refinement: {e}"
    elif method == 'point_to_point' or method == 'point_to_plane':
        refined_T = refine_extrinsic_ICP(
            source_pcd=infra_pcd,
            target_pcd=vehicle_pcd,
            initial_guess=noisy_T,
            method=method,
            voxel_size=voxel_size
        )
    elif method == 'ransac':
        try:
            refined_T = refine_with_ransac(
                source=infra_pcd,
                target=vehicle_pcd,
                voxel_size=voxel_size
            )
        except Exception as e:
            return f"Error during RANSAC refinement: {e}"
    else:
        return (f"inf_id: {inf_id}, veh_id: {veh_id}, Unsupported method: {method}.")
    end_time = time.time()

    refined_T_6DOF = convert_T_to_6DOF(np.array(refined_T))
    T_true_6DOF = convert_T_to_6DOF(T_true)
    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(refined_T_6DOF, T_true_6DOF)

    if cfg.data.type == 'V2X-Sim':
            id_str = f"frame_id: {inf_id}, cav_id: {veh_id}"
    elif cfg.data.type == 'DAIR-V2X':
        id_str = f"inf_id: {inf_id}, veh_id: {veh_id}"

    basic_result_str = f", RE: {RE}, TE: {TE}, time: {end_time - start_time}  "
    detailed_result_str = f"delta_trans: {refined_T_6DOF-T_true_6DOF}, delta_rot: {refined_T_6DOF - T_true_6DOF}"
    result_str = id_str + basic_result_str + detailed_result_str
    print(result_str)

    return result_str


def batching_test_extrisic_from_two_box_object_list(cfg, rot_noise, trans_noise, method='ransac', voxel_size=5):
    saved_record_name = f"{method}_voxel{voxel_size}_rot{rot_noise}_trans{trans_noise}"
    logger = Logger(saved_record_name)

    if cfg.data.type == 'V2X-Sim':
        wrapper = V2XSim_Reader().generate_vehicle_vehicle_bboxes_object_list(noise=cfg.data.noise)
    elif cfg.data.type == 'DAIR-V2X':
        wrapper = CooperativeBatchingReader(path_data_info = cfg.data.data_info_path, path_data_folder= cfg.data.data_root_path).generate_infra_vehicle_bboxes_object_list()

    for id1, id2, _, _, _ in wrapper:

        logger.info(get_processed_solo_result(cfg, id1, id2, rot_noise=rot_noise, trans_noise=trans_noise, method=method, voxel_size=voxel_size)) 

# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, default='./config/config.yaml')
#     args = parser.parse_args()
#     cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

#     rot_noise_list = [0, 0.5, 1, 1.5, 2]
#     trans_noise_list = [0, 0.5, 1, 1.5, 2]

#     for rot_noise, trans_noise in zip(rot_noise_list, trans_noise_list):
#         batching_test_extrisic_from_two_box_object_list(cfg, rot_noise, trans_noise)

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# rot_noise_list = [0, 0.5, 1, 1.5, 2]
# trans_noise_list = [0, 0.5, 1, 1.5, 2]

rot_noise_list = [10, 20]
trans_noise_list = [10, 20]

def thread_task(args):
    cfg, rot_noise, trans_noise = args
    batching_test_extrisic_from_two_box_object_list(cfg, rot_noise, trans_noise)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    # 创建参数组合列表
    params = [(cfg, r, t) for r, t in zip(rot_noise_list, trans_noise_list)]
    
    # 启动线程池（建议max_workers=参数组合数量）
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(thread_task, params)