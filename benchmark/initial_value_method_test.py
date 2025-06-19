import numpy as np
import os
import math
import time
import sys
import argparse
import open3d as o3d
# import numba as nb
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from v2x_calib.reader import CooperativeBatchingReader
from v2x_calib.reader import CooperativeReader
from v2x_calib.preprocess import Filter3dBoxes
from v2x_calib.reader import V2XSim_Reader
from v2x_calib.corresponding import CorrespondingDetector
from v2x_calib.corresponding import BoxesMatch
from v2x_calib.search import Matches2Extrinsics
from v2x_calib.utils import implement_T_3dbox_object_list, implement_T_points_n_3, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, convert_6DOF_to_T
from v2x_calib.corresponding.CBM_torch import CBM
from config.config import cfg, cfg_from_yaml_file, Logger
from visualize import BBoxVisualizer_open3d
from visualize import BBoxVisualizer_open3d_standardized
from v2x_calib.corresponding.VIPS import create_affinity_matrix, find_optimal_matching


# def vips_simple_matching(infra_boxes_list, vehicle_boxes_list, init_T):
#     converted_infra_boxes_list = implement_T_3dbox_object_list(init_T, infra_boxes_list)

#     matches_with_score_dict = CorrespondingDetector(converted_infra_boxes_list, vehicle_boxes_list, distance_threshold=cfg.v2x_calib.distance_threshold, parallel=cfg.v2x_calib.correspoding_parallel_flag).get_matches_with_score()
#     matches_with_score_list = []
#     for match, score in matches_with_score_dict.items():
#         infra_idx, vehicle_idx = match
#         matches_with_score_list.append(((infra_idx, vehicle_idx), score))
#     return matches_with_score_list


def vips_matching(infra_boxes_list, vehicle_boxes_list, init_T):
    """
    将原始VIPS匹配逻辑封装成新函数。

    输入:
        infra_boxes_list (list): 基础设施（或视角1）的检测框列表。
        vehicle_boxes_list (list): 车辆（或视角2）的检测框列表，结构同上。
        init_T (np.array): 4x4的变换矩阵。注意：当前实现中，如果'world_position'
                           已经是统一坐标系，则此矩阵可能不直接用于坐标转换。

    输出:
        matches_with_score_list (list): 匹配对列表，格式为 [((infra_idx, vehicle_idx), score), ...]
    """

    L1 = len(infra_boxes_list)
    L2 = len(vehicle_boxes_list)

    if L1 == 0 or L2 == 0:
        return []

    # 1. 构建 G1 和 G2 特征矩阵 (与您原main函数逻辑一致)
    G1 = np.zeros((L1, 11), dtype=np.float32)
    G2 = np.zeros((L2, 11), dtype=np.float32)

    for i, box in enumerate(infra_boxes_list):
        box_8_3 = box.get_bbox3d_8_3()
        cemter_point = np.mean(box_8_3, axis=0)
        x, y, z = cemter_point[0], cemter_point[1], cemter_point[2]
        l, w, h = np.abs(box_8_3[4] - box_8_3[2])
        transformed_box_8_3 = implement_T_points_n_3(init_T, box_8_3).reshape(-1, 3)
        transformed_cemter_point = np.mean(transformed_box_8_3, axis=0)
        # catagory = box.get_bbox_type()
        catagory = 0
        G1[i] = [catagory, x, y, z, l, w, h, transformed_cemter_point[0], transformed_cemter_point[1], transformed_cemter_point[2], 0]

    for i, box in enumerate(vehicle_boxes_list):
        box_8_3 = box.get_bbox3d_8_3()
        cemter_point = np.mean(box_8_3, axis=0)
        x, y, z = cemter_point[0], cemter_point[1], cemter_point[2]
        l, w, h = np.abs(box_8_3[4] - box_8_3[2])
        # catagory = box.get_bbox_type()
        catagory = 0
        G2[i] = [catagory, x, y, z, l, w, h, x, y, z, 0]

    # 2. 创建亲和度矩阵 M (调用您项目中已有的函数)
    # M_affinity 是优化目标函数中的 A 矩阵 (x^T * A * x)
    M_affinity = create_affinity_matrix(G1, G2, L1, L2)

    if not isinstance(M_affinity, np.ndarray) or M_affinity.size == 0:
        # print("[vips_matching] Warning: Affinity matrix is empty or not an array.")
        return []
    if M_affinity.ndim != 2 or M_affinity.shape[0] != M_affinity.shape[1]:
        # print(f"[vips_matching] Warning: Affinity matrix M must be square, got {M_affinity.shape}")
        return []
    if M_affinity.shape[0] == 0 : # Double check for empty square matrix
        return []


    # 3. 优化求解 (与您原main函数逻辑一致)
    # 初始猜测向量x，其长度应等于M_affinity的维度
    w_initial_guess = np.zeros(M_affinity.shape[0])

    # 定义目标函数
    # @nb.njit()
    def objective_function(x=np.array([]), A=np.array([[]])):
        return -(x.T @ A @ x)

    # 定义约束条件：||x||^2 - 1 = 0
    # @nb.njit()
    def constraint(x=np.array([])):
        return np.linalg.norm(x)**2 - 1

    # 定义约束 (确保 mock_vips_constraint 在作用域内或正确导入)
    constraint_eq = {'type': 'eq', 'fun': constraint}

    # 执行优化 (确保 mock_vips_objective_function 在作用域内或正确导入)
    try:
        optimization_result = minimize(objective_function,
                                       w_initial_guess,
                                       args=(M_affinity,),
                                       method='SLSQP',
                                       constraints=constraint_eq)
        w_a_optimized_vector = optimization_result.x
    except Exception as e:
        # print(f"[vips_matching] Optimization failed: {e}")
        return []


    # 标准化 w_a_optimized_vector (可选，取决于find_optimal_matching的需要)
    w_a_normalized = w_a_optimized_vector # 默认不标准化
    if np.ptp(w_a_optimized_vector) > 1e-9: # ptp (peak-to-peak) 检查范围是否有效
        w_a_normalized = (w_a_optimized_vector - np.min(w_a_optimized_vector)) / np.ptp(w_a_optimized_vector)
    elif len(w_a_optimized_vector) > 0 : # 如果范围是0 (所有值相同)
        w_a_normalized = np.full_like(w_a_optimized_vector, 0.5) # 或0, 或1, 取决于意义


    # 4. 寻找最优匹配 (调用您项目中已有的函数)
    # raw_matching_results 的格式应为 [[idx_infra, idx_vehicle], ...]
    raw_matching_results = find_optimal_matching(w_a_normalized, L1, L2, threshold=0.5)

    # 5. 转换输出格式为 [((infra_idx, vehicle_idx), score), ...]
    matches_with_score_list = []
    if isinstance(raw_matching_results, np.ndarray) and raw_matching_results.ndim == 2 and raw_matching_results.shape[1] == 2:
        for i in range(raw_matching_results.shape[0]):
            infra_idx = int(raw_matching_results[i, 0])
            vehicle_idx = int(raw_matching_results[i, 1])
            #  当前版本VIPS逻辑中似乎没有直接的每个匹配对的分数，除非从 w_a_normalized 中提取
            #  为了与cbm_matching接口一致，我们给一个默认分数
            score = 1.0
            matches_with_score_list.append(((infra_idx, vehicle_idx), score))

    return matches_with_score_list


def cbm_matching(infra_boxes_object_list, vehicle_boxes_object_list, init_T):
    """
    
    """

    def get_cav_box7_nparray(boxes_object_list):
        box7_nparray = np.zeros((len(boxes_object_list), 7))
        for i, box_object in enumerate(boxes_object_list):
            box_8_3 = box_object.get_bbox3d_8_3()
            cemter_point = np.mean(box_8_3, axis=0)
            x, y, z = cemter_point[0], cemter_point[1], cemter_point[2]
            l, w, h = np.abs(box_8_3[4] - box_8_3[2])
            box7 = np.array([x, y, z, h, w, l, 0])
            box7_nparray[i, :] = box7
        return box7_nparray

    infra_box7_nparray = get_cav_box7_nparray(infra_boxes_object_list)
    vehicle_box7_nparray = get_cav_box7_nparray(vehicle_boxes_object_list)

    if infra_box7_nparray.shape[0] == 0 or vehicle_box7_nparray.shape[0] == 0:
        return []

    CBM_ = CBM(cfg.args)
    matching = CBM_(infra_box7_nparray, vehicle_box7_nparray, init_T)
    if not isinstance(matching, np.ndarray):
        matching = np.asarray(matching.cpu())

    matches_with_score_list = [((int(matching[i, 0]), int(matching[i, 1])), 1) for i in range(len(matching))]

    return matches_with_score_list

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

def batching_test_extrisic_from_two_box_object_list(cfg, rot_noise=0, trans_noise=0, method='cbm'):
    
    # saved_record_name = f"{cfg.data.type}_{cfg.v2x_calib.filter_num}_{cfg.v2x_calib.similarity_strategy}_{cfg.v2x_calib.core_similarity_component_list}_{cfg.v2x_calib.matches_filter_strategy}"
    saved_record_name = f"{method}_rot{rot_noise}_trans{trans_noise}_{cfg.data.type}_{cfg.v2x_calib.filter_num}_{cfg.v2x_calib.matches_filter_strategy}"
    if cfg.v2x_calib.matches_filter_strategy == 'threshold':
        saved_record_name += f"_{cfg.v2x_calib.filter_threshold}"
    saved_record_name += f"_{cfg.v2x_calib.matches2extrinsic_strategies}_{cfg.data.data_info_path.split('/')[-1].split('.')[0]}"
    saved_record_name += f"_{cfg.v2x_calib.svd_strategy}"
    if cfg.v2x_calib.parallel_flag == 1:
        saved_record_name += f"_parallel"
    logger = Logger(saved_record_name)

    if cfg.data.type == 'V2X-Sim':
        wrapper = V2XSim_Reader().generate_vehicle_vehicle_bboxes_object_list(noise=cfg.data.noise)
    elif cfg.data.type == 'DAIR-V2X':
        wrapper = CooperativeBatchingReader(path_data_info = cfg.data.data_info_path, path_data_folder= cfg.data.data_root_path).generate_infra_vehicle_bboxes_object_list()
    elif cfg.data.type == 'DAIR-V2X_detected':
        wrapper = CooperativeBatchingReader(path_data_info = cfg.data.data_info_path, path_data_folder= cfg.data.data_root_path).generate_infra_vehicle_bboxes_object_list_predicted()
    for id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true in wrapper:

        logger.info(get_processed_solo_result(cfg, id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true, rot_noise=rot_noise, trans_noise=trans_noise, method=method)[0])



def get_processed_solo_result(cfg, id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true, time_veerbose = False, rot_noise=0, trans_noise=0, method='cbm'):

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    
    converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
    filtered_available_matches = CorrespondingDetector(
        converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold, parallel=cfg.v2x_calib.correspoding_parallel_flag).get_matches()
    
    converted_original_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
    total_available_matches_cnt = CorrespondingDetector(
        converted_original_infra_boxes_object_list, vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold, parallel=cfg.v2x_calib.correspoding_parallel_flag).get_matched_num()
    # print(sorted(CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold).get_matches_with_score().items(), key=lambda x: x[1], reverse=True))
    ##################
    start_time = time.time()

    init_T = add_noise_to_extrinsic(T_true, rot_noise, trans_noise)

    if method == 'cbm':
        matches_with_score_list = cbm_matching(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, init_T)
    elif method == 'vips':
        matches_with_score_list = vips_matching(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, init_T)
    # elif method == 'vips_simple':
    #     matches_with_score_list = vips_simple_matching(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, init_T)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # print(f"matches_with_score_list: {matches_with_score_list}")
    matches = [match[0] for match in matches_with_score_list]


    if len(matches_with_score_list) > 0:
        T_6DOF_result = Matches2Extrinsics(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, matches_score_list=matches_with_score_list, svd_strategy=cfg.v2x_calib.svd_strategy) \
                            .get_combined_extrinsic(matches2extrinsic_strategies=cfg.v2x_calib.matches2extrinsic_strategies)
        stability = -1

    else:
        T_6DOF_result, stability = [0, 0, 0, 0, 0, 0], 0

        
    if time_veerbose:
        end_time = time.time()
        print(f"Time taken for filtering: {end_time - start_time} seconds")
        start_time = time.time()

    end_time = time.time()
    ##################

    T_6DOF_true = convert_T_to_6DOF(T_true)
    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF_result, T_6DOF_true)

    result_matches = []

    for match in matches:
        if match in filtered_available_matches:
            result_matches.append(match)

    result_matched_cnt, filtered_available_matches_cnt, wrong_result_matches_cnt =  len(result_matches), len(filtered_available_matches), len(matches) - len(result_matches)

    if cfg.data.type == 'V2X-Sim':
            id_str = f"frame_id: {id1}, cav_id: {id2}"
    elif cfg.data.type == 'DAIR-V2X':
        id_str = f"inf_id: {id1}, veh_id: {id2}"
    elif cfg.data.type == 'DAIR-V2X_detected':
        id_str = f"inf_id: {id1}, veh_id: {id2}"

    basic_result_str = f", RE: {RE}, TE: {TE}, stability: {stability}, time: {end_time - start_time}  "
    detailed_result_str = f" ==details==> infra_total_box_cnt : {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}, filtered_available_matches_cnt: {filtered_available_matches_cnt}, result_matched_cnt: {result_matched_cnt}, wrong_result_matches_cnt: {wrong_result_matches_cnt}"

    return id_str + basic_result_str + detailed_result_str, T_6DOF_result


def test_solo_with_dataset(cgf, inf_id, veh_id, time_veerbose = False):
# DAIR-V2X
    reader = CooperativeReader(infra_file_name=inf_id, vehicle_file_name=veh_id, data_folder=cfg.data.data_root_path)
    infra_boxes_object_list, vehicle_boxes_object_list = reader.get_cooperative_infra_vehicle_boxes_object_list()
    T_true = reader.get_cooperative_T_i2v()
    
    result_str, T_6DOF_result = get_processed_solo_result(cfg, inf_id, veh_id, infra_boxes_object_list, vehicle_boxes_object_list, T_true, time_veerbose = time_veerbose)
    print(result_str)

    infra_pointcloud, vehicle_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    
    T_result = convert_6DOF_to_T(T_6DOF_result)
    T_result_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_result, filtered_infra_boxes_object_list)
    T_result_converted_infra_pointcloud = implement_T_points_n_3(T_result, infra_pointcloud)

    T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
    T_true_converted_infra_pointcloud = implement_T_points_n_3(T_true, infra_pointcloud)

    # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([T_result_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_result_converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)], 'result_T')
    # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_true_converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)], 'true T')
    
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_result_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_result_converted_infra_pointcloud, vehicle_pointcloud], {}, vis_id=0)
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_true_converted_infra_pointcloud, vehicle_pointcloud], {}, vis_id=0)

    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_result_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [], {}, vis_id=0)
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [], {}, vis_id=0)


    
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

rot_noise_list = [0, 0.5, 1, 1.5, 2]
trans_noise_list = [0, 0.5, 1, 1.5, 2]
# rot_noise_list = [0]
# trans_noise_list = [0]
methodd = 'vips_simple'
# rot_noise_list = [10, 20]
# trans_noise_list = [10, 20]

def thread_task(args):
    cfg, rot_noise, trans_noise = args
    batching_test_extrisic_from_two_box_object_list(cfg, rot_noise, trans_noise, method=methodd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--test_type', type=str, default='single', help='single or batch')
    parser.add_argument('--inf_id', type=str, default='001366')
    parser.add_argument('--veh_id', type=str, default='017314')
    
    parser.add_argument('--filter_num', type=int, default=-1)
    parser.add_argument('--matches_filter_strategy', type=str, default='', help='thresholdRetained or topRetained or trueRetained or allRetained')
    parser.add_argument('--matches2extrinsic_strategies', type=str, default='', help='weightedSVD or evenSVD')
    parser.add_argument('--svd_strategy', type=str, default='', help='svd_with_match or svd_without_match')
    parser.add_argument('--parallel_flag', type=int, default=-1, help='True or False')

    parser.add_argument('--sigma1', default=10 * math.pi / 180,
                            help='rad')
    parser.add_argument('--sigma2', default=3,
                        help='m')
    parser.add_argument('--sigma3', default=1,
                        help='m')
    parser.add_argument('--absolute_dis_lim', default=20,
                        help='m')

    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    cfg.args = args

    if args.filter_num != -1:
        cfg.v2x_calib.filter_num = args.filter_num
    if args.matches_filter_strategy != '':
        cfg.v2x_calib.matches_filter_strategy = args.matches_filter_strategy
    if args.matches2extrinsic_strategies != '':
        cfg.v2x_calib.matches2extrinsic_strategies = args.matches2extrinsic_strategies
    if args.svd_strategy != '':
        cfg.v2x_calib.svd_strategy = args.svd_strategy
    if args.parallel_flag != -1:
        cfg.v2x_calib.parallel_flag = args.parallel_flag

    # 创建参数组合列表
    params = [(cfg, r, t) for r, t in zip(rot_noise_list, trans_noise_list)]
    
    # 启动线程池（建议max_workers=参数组合数量）
    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(thread_task, params)

    # batching_test_extrisic_from_two_box_object_list(cfg, 0, 0, method=methodd)