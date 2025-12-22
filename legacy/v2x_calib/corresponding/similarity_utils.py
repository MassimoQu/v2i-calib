import numpy as np
from pathlib import Path
from ..utils import (
    cal_3dIoU,
    get_lwh_from_bbox3d_8_3,
    get_bbox3d_8_3_from_xyz_lwh,
    get_vector_between_bbox3d_8_3,
    get_length_between_bbox3d_8_3,
    get_time_judge,
    implement_T_3dbox_object_list,
    get_extrinsic_from_two_3dbox_object,
    get_extrinsic_from_two_3dbox_object_svd_without_match,
    convert_T_to_6DOF,
)
from .CorrespondingDetector import CorrespondingDetector
import multiprocessing as mp
from itertools import product
import functools # For functools.partial
import atexit # To ensure pool cleanup on exit
from scipy.optimize import linear_sum_assignment

_LARGE_COST = 1e6


def _stack_bbox_vertices(box_object_list):
    if not box_object_list:
        return np.zeros((0, 8, 3), dtype=np.float32)
    return np.stack(
        [np.asarray(box.get_bbox3d_8_3(), dtype=np.float32) for box in box_object_list],
        axis=0,
    )


def _stack_bbox_types(box_object_list):
    if not box_object_list:
        return np.zeros((0,), dtype=object)
    return np.array([str(box.get_bbox_type()).lower() for box in box_object_list], dtype=object)


def _normalize_thresholds(distance_threshold):
    normalized = {}
    for key, value in (distance_threshold or {}).items():
        normalized[str(key).lower()] = float(value)
    if not normalized:
        return normalized, 1.0
    fallback = normalized.get('detected', max(normalized.values()))
    return normalized, float(fallback)


def _solve_assignment(dist_matrix, threshold_row, type_match):
    if dist_matrix.size == 0:
        return 0
    allowed = type_match & (dist_matrix <= threshold_row[:, None])
    if not allowed.any():
        return 0
    cost = dist_matrix.copy()
    cost[~allowed] = _LARGE_COST
    row_ind, col_ind = linear_sum_assignment(cost)
    if row_ind.size == 0:
        return 0
    valid = allowed[row_ind, col_ind]
    return int(np.count_nonzero(valid))


def _solve_assignment_stats(dist_matrix, threshold_row, type_match):
    """
    Returns:
        (matched_count, avg_distance)
    """
    if dist_matrix.size == 0:
        return 0, float('inf')
    allowed = type_match & (dist_matrix <= threshold_row[:, None])
    if not allowed.any():
        return 0, float('inf')
    cost = dist_matrix.copy()
    cost[~allowed] = _LARGE_COST
    row_ind, col_ind = linear_sum_assignment(cost)
    if row_ind.size == 0:
        return 0, float('inf')
    valid = allowed[row_ind, col_ind]
    if not np.any(valid):
        return 0, float('inf')
    selected = dist_matrix[row_ind, col_ind][valid]
    return int(selected.size), float(np.mean(selected))


def _count_matches_center(converted_vertices, vehicle_centers, threshold_row, type_match):
    if converted_vertices.size == 0 or vehicle_centers.size == 0:
        return 0
    centers = converted_vertices.mean(axis=1)
    dist = np.linalg.norm(centers[:, None, :] - vehicle_centers[None, :, :], axis=2)
    return _solve_assignment(dist, threshold_row, type_match)


def _count_matches_vertex(converted_vertices, vehicle_vertices, threshold_row, type_match):
    if converted_vertices.size == 0 or vehicle_vertices.size == 0:
        return 0
    infra_flat = converted_vertices.reshape(converted_vertices.shape[0], -1)
    veh_flat = vehicle_vertices.reshape(vehicle_vertices.shape[0], -1)
    dist = np.linalg.norm(infra_flat[:, None, :] - veh_flat[None, :, :], axis=2) / 8.0
    return _solve_assignment(dist, threshold_row, type_match)


def cal_core_KP_distance_fast_components(
    infra_object_list,
    vehicle_object_list,
    use_centerpoint=True,
    use_vertex=False,
    *,
    category_flag=True,
    distance_threshold=None,
    svd_starategy='svd_with_match',
    infra_indices=None,
    vehicle_indices=None,
):
    """
    Vectorized variant that evaluates both centerpoint/vertex distance components in one pass.
    Returns per-component KP matrices (or None) along with the maximum matched count.
    """
    num_infra = len(infra_object_list)
    num_vehicle = len(vehicle_object_list)
    KP_center = np.zeros((num_infra, num_vehicle), dtype=np.float32) if use_centerpoint else None
    KP_vertex = np.zeros((num_infra, num_vehicle), dtype=np.float32) if use_vertex else None
    if num_infra == 0 or num_vehicle == 0:
        return KP_center, KP_vertex, -1

    infra_vertices = _stack_bbox_vertices(infra_object_list)
    vehicle_vertices = _stack_bbox_vertices(vehicle_object_list)
    vehicle_centers = vehicle_vertices.mean(axis=1) if use_centerpoint else None
    infra_types = _stack_bbox_types(infra_object_list)
    vehicle_types = _stack_bbox_types(vehicle_object_list)
    norm_thresholds, fallback_threshold = _normalize_thresholds(distance_threshold or {})
    threshold_row = np.array(
        [norm_thresholds.get(t, fallback_threshold) for t in infra_types], dtype=np.float32
    )
    if category_flag:
        type_match = infra_types[:, None] == vehicle_types[None, :]
    else:
        type_match = np.ones((num_infra, num_vehicle), dtype=bool)

    base_flat = infra_vertices.reshape(-1, 3)
    base_shape = infra_vertices.shape
    max_matches_num = -1

    def _transform_vertices(T_matrix):
        T_np = np.asarray(T_matrix, dtype=np.float32)
        R = T_np[:3, :3]
        t = T_np[:3, 3]
        transformed = base_flat @ R.T + t
        return transformed.reshape(base_shape)

    if infra_indices is None:
        infra_indices = range(num_infra)
    if vehicle_indices is None:
        vehicle_indices = range(num_vehicle)

    for i in infra_indices:
        infra_box = infra_object_list[i]
        infra_type = infra_types[i] if i < len(infra_types) else ''
        for j in vehicle_indices:
            vehicle_box = vehicle_object_list[j]
            if category_flag and infra_type != vehicle_types[j]:
                continue
            if svd_starategy == 'svd_with_match':
                T = get_extrinsic_from_two_3dbox_object(infra_box, vehicle_box)
            elif svd_starategy == 'svd_without_match':
                T = get_extrinsic_from_two_3dbox_object_svd_without_match(infra_box, vehicle_box)
            else:
                raise ValueError('svd_starategy should be svd_with_match or svd_without_match')
            converted_vertices = _transform_vertices(T)
            if use_centerpoint:
                matches = _count_matches_center(converted_vertices, vehicle_centers, threshold_row, type_match)
                KP_center[i, j] = max(matches - 1, 0)
                max_matches_num = max(max_matches_num, matches)
            if use_vertex:
                matches_vertex = _count_matches_vertex(
                    converted_vertices, vehicle_vertices, threshold_row, type_match
                )
                KP_vertex[i, j] = max(matches_vertex - 1, 0)
                max_matches_num = max(max_matches_num, matches_vertex)

    return KP_center, KP_vertex, max_matches_num


def cal_core_KP_distance_odist_fast(
    infra_object_list,
    vehicle_object_list,
    *,
    category_flag=True,
    distance_threshold=None,
    svd_starategy='svd_with_match',
    precision_threshold=0.0,
    alpha=0.5,
    beta=0.5,
    infra_indices=None,
    vehicle_indices=None,
):
    """
    Fast oDist-style affinity computation.

    For each candidate seed pair (i,j), compute a local alignment hypothesis via SVD,
    then derive:
      - tau_C: number of valid matched pairs under the hypothesis
      - tau_D: average matched distance (center + vertex)

    The affinity KP[i,j] is set to (tau_C - 1) if tau_D <= precision_threshold, else 0.
    """
    num_infra = len(infra_object_list)
    num_vehicle = len(vehicle_object_list)
    KP = np.zeros((num_infra, num_vehicle), dtype=np.float32)
    if num_infra == 0 or num_vehicle == 0:
        return KP, -1

    infra_vertices = _stack_bbox_vertices(infra_object_list)
    vehicle_vertices = _stack_bbox_vertices(vehicle_object_list)
    vehicle_centers = vehicle_vertices.mean(axis=1)
    vehicle_flat = vehicle_vertices.reshape(num_vehicle, -1)

    infra_types = _stack_bbox_types(infra_object_list)
    vehicle_types = _stack_bbox_types(vehicle_object_list)

    norm_thresholds, fallback_threshold = _normalize_thresholds(distance_threshold or {})
    threshold_row = np.array(
        [norm_thresholds.get(t, fallback_threshold) for t in infra_types], dtype=np.float32
    )
    if category_flag:
        type_match = infra_types[:, None] == vehicle_types[None, :]
    else:
        type_match = np.ones((num_infra, num_vehicle), dtype=bool)

    base_flat = infra_vertices.reshape(-1, 3)
    base_shape = infra_vertices.shape

    def _transform_vertices(T_matrix):
        T_np = np.asarray(T_matrix, dtype=np.float32)
        R = T_np[:3, :3]
        t = T_np[:3, 3]
        transformed = base_flat @ R.T + t
        return transformed.reshape(base_shape)

    if infra_indices is None:
        infra_indices = range(num_infra)
    if vehicle_indices is None:
        vehicle_indices = range(num_vehicle)

    precision_threshold = float(precision_threshold) if precision_threshold is not None else 0.0
    max_matches_num = -1
    for i in infra_indices:
        infra_box = infra_object_list[i]
        infra_type = infra_types[i] if i < len(infra_types) else ''
        for j in vehicle_indices:
            vehicle_box = vehicle_object_list[j]
            veh_type = vehicle_types[j] if j < len(vehicle_types) else ''
            if category_flag and infra_type != veh_type:
                continue
            if svd_starategy == 'svd_with_match':
                T = get_extrinsic_from_two_3dbox_object(infra_box, vehicle_box)
            elif svd_starategy == 'svd_without_match':
                T = get_extrinsic_from_two_3dbox_object_svd_without_match(infra_box, vehicle_box)
            else:
                raise ValueError('svd_starategy should be svd_with_match or svd_without_match')

            converted_vertices = _transform_vertices(T)
            centers = converted_vertices.mean(axis=1)
            dist_center = np.linalg.norm(
                centers[:, None, :] - vehicle_centers[None, :, :], axis=2
            )
            infra_flat = converted_vertices.reshape(num_infra, -1)
            dist_vertex = np.linalg.norm(
                infra_flat[:, None, :] - vehicle_flat[None, :, :], axis=2
            ) / 8.0
            dist = alpha * dist_center + beta * dist_vertex

            matched_count, avg_dist = _solve_assignment_stats(dist, threshold_row, type_match)
            max_matches_num = max(max_matches_num, matched_count)
            score = max(matched_count - 1, 0)
            if matched_count == 0 or (precision_threshold > 0 and avg_dist > precision_threshold):
                score = 0
            KP[i, j] = score

    return KP, max_matches_num

_PERSISTENT_PROCESS_POOL = None

def get_persistent_pool(num_processes=None):
    """
    Creates and returns a global persistent multiprocessing pool.
    Initializes it only once.
    """
    global _PERSISTENT_PROCESS_POOL
    if _PERSISTENT_PROCESS_POOL is None:
        actual_num_processes = num_processes or mp.cpu_count()
        print(f"Initializing persistent global pool with {actual_num_processes} processes.")
        _PERSISTENT_PROCESS_POOL = mp.Pool(processes=actual_num_processes)
    return _PERSISTENT_PROCESS_POOL

def cleanup_persistent_pool():
    """Closes and joins the global pool if it exists."""
    global _PERSISTENT_PROCESS_POOL
    if _PERSISTENT_PROCESS_POOL:
        print("Cleaning up persistent global pool.")
        _PERSISTENT_PROCESS_POOL.close()
        _PERSISTENT_PROCESS_POOL.join()
        _PERSISTENT_PROCESS_POOL = None

# Register the cleanup function to be called at program exit
atexit.register(cleanup_persistent_pool)

# --- Refactored Task Processing Function ---
# process_task no longer relies on global variables for task-specific data.
# All necessary data is passed as arguments.
def process_task_refactored(ij_pair, infra_list_arg, vehicle_list_arg, 
                            category_flag_arg, core_sim_arg, distance_thresh_arg, parallel):
    """
    Processes a single (i, j) pair.
    All data is passed via arguments, not globals from an initializer.
    """
    i, j = ij_pair
    infra_obj = infra_list_arg[i]
    vehicle_obj = vehicle_list_arg[j]
    
    if category_flag_arg and infra_obj.get_bbox_type() != vehicle_obj.get_bbox_type():
        return (i, j, 0.0, -1)
    
    # These functions would be your actual implementations
    # Ensure they are defined or imported in the scope where workers can access them.
    # For example:
    # from .utils import get_extrinsic_from_two_3dbox_object, implement_T_3dbox_object_list
    # from .detectors import CorrespondingDetector

    T = get_extrinsic_from_two_3dbox_object(infra_obj, vehicle_obj)
    converted_infra = implement_T_3dbox_object_list(T, infra_list_arg) # Pass infra_list_arg
    
    detector = CorrespondingDetector(
        converted_infra, vehicle_list_arg, # Pass vehicle_list_arg
        core_similarity_component=core_sim_arg,
        distance_threshold=distance_thresh_arg,
        parallel=parallel
    )
    score = int(detector.get_Yscore())
    matched_num = detector.get_matched_num()
    return (i, j, score, matched_num)


# --- Main Parallel Calculation Function (Refactored) ---
def cal_core_KP_distance_parallel_refactored(
    infra_object_list, vehicle_object_list,
    category_flag=True, core_similarity_component='centerpoint_distance',
    distance_threshold=3, num_processes=None, parallel=False
):
    """
    Calculates KP distance using a persistent global multiprocessing pool.
    """
    task_indices = [] # List of (i, j) index pairs
    for i, infra_obj in enumerate(infra_object_list):
        for j, vehicle_obj in enumerate(vehicle_object_list):
            if category_flag and infra_obj.get_bbox_type() != vehicle_obj.get_bbox_type():
                continue
            task_indices.append((i, j))
    
    if not task_indices: # No tasks to process after filtering
        return np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64), -1

    # Get the persistent global pool
    pool = get_persistent_pool(num_processes=num_processes)
    
    # Use functools.partial to create a new function with some arguments pre-filled.
    # These pre-filled arguments (infra_object_list, vehicle_object_list, etc.)
    # will be pickled and sent to the worker processes along with the tasks.
    task_processor_with_data = functools.partial(
        process_task_refactored,
        infra_list_arg=infra_object_list,
        vehicle_list_arg=vehicle_object_list,
        category_flag_arg=category_flag,
        core_sim_arg=core_similarity_component,
        distance_thresh_arg=distance_threshold, 
        parallel=parallel
    )
    
    # pool.map will now call task_processor_with_data(ij_pair) for each ij_pair in task_indices
    results = pool.map(task_processor_with_data, task_indices)
    
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    max_matches_num = -1
    if results: # Check if results is not empty
        for i_res, j_res, score_res, matched_num_res in results:
            KP[i_res, j_res] = score_res
            if matched_num_res > max_matches_num:
                max_matches_num = matched_num_res
                
    return KP, max_matches_num


def normalized_KP(KP):
    KP_ = KP.copy()
    if KP_.shape[0] > 0 and KP_.shape[1] > 0:
        max_val = np.max(KP_)
        min_val = np.min(KP_)
        for i in range(KP_.shape[0]):
            for j in range(KP_.shape[1]):
                if KP_[i, j] != 0 and max_val != min_val:
                    KP_[i, j] = int((KP_[i, j] - min_val) / (max_val - min_val) * 10)
                else:
                    KP_[i, j] = 0
    return KP_

def cal_core_KP_IoU(infra_object_list, vehicle_object_list, category_flag = True):
    max_matches_num = -1
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    for i, infra_bbox_object in enumerate(infra_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_object_list):
            if category_flag:
                if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                    continue
            T = get_extrinsic_from_two_3dbox_object(infra_bbox_object, vehicle_bbox_object)
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, infra_object_list)
            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_object_list, core_similarity_component='iou')
            KP[i, j] = int(corresponding_detector.get_Yscore()) * infra_bbox_object.get_confidence() * vehicle_bbox_object.get_confidence()
            if max_matches_num < corresponding_detector.get_matched_num():
                max_matches_num = corresponding_detector.get_matched_num()
    # return KP, normalized_KP(KP), max_matches_num
    return KP, max_matches_num
def _stack_confidences(box_object_list):
    if not box_object_list:
        return np.zeros((0,), dtype=np.float32)
    return np.array([float(getattr(box, 'confidence', 1.0)) for box in box_object_list], dtype=np.float32)


def _bounds_from_vertices(vertices):
    """
    Compute axis-aligned bounds (min/max per axis) for each box in ``vertices``.
    vertices: (N, 8, 3)
    returns: (mins, maxs) each (N, 3)
    """
    if vertices.size == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    mins = vertices.min(axis=1)
    maxs = vertices.max(axis=1)
    return mins, maxs


def cal_core_KP_IoU_fast(
    infra_object_list,
    vehicle_object_list,
    *,
    category_flag=True,
    svd_starategy='svd_with_match',
    infra_indices=None,
    vehicle_indices=None,
):
    num_infra = len(infra_object_list)
    num_vehicle = len(vehicle_object_list)
    KP = np.zeros((num_infra, num_vehicle), dtype=np.float32)
    if num_infra == 0 or num_vehicle == 0:
        return KP, -1

    infra_vertices = _stack_bbox_vertices(infra_object_list)
    vehicle_vertices = _stack_bbox_vertices(vehicle_object_list)
    infra_types = _stack_bbox_types(infra_object_list)
    vehicle_types = _stack_bbox_types(vehicle_object_list)
    infra_conf = _stack_confidences(infra_object_list)
    vehicle_conf = _stack_confidences(vehicle_object_list)

    base_flat = infra_vertices.reshape(-1, 3)
    base_shape = infra_vertices.shape

    def _transform_vertices(T_matrix):
        T_np = np.asarray(T_matrix, dtype=np.float32)
        R = T_np[:3, :3]
        t = T_np[:3, 3]
        transformed = base_flat @ R.T + t
        return transformed.reshape(base_shape)

    type_to_vehicle = {}
    for idx, v_type in enumerate(vehicle_types):
        key = str(v_type).lower()
        type_to_vehicle.setdefault(key, []).append(idx)

    veh_mins, veh_maxs = _bounds_from_vertices(vehicle_vertices)

    max_matches_num = -1
    if infra_indices is None:
        infra_indices = range(num_infra)
    if vehicle_indices is None:
        vehicle_indices = range(num_vehicle)

    for i in infra_indices:
        infra_box = infra_object_list[i]
        infra_type = str(infra_types[i]).lower()
        for j in vehicle_indices:
            vehicle_box = vehicle_object_list[j]
            veh_type = str(vehicle_types[j]).lower()
            if category_flag and infra_type != veh_type:
                continue
            if svd_starategy == 'svd_with_match':
                T = get_extrinsic_from_two_3dbox_object(infra_box, vehicle_box)
            elif svd_starategy == 'svd_without_match':
                T = get_extrinsic_from_two_3dbox_object_svd_without_match(infra_box, vehicle_box)
            else:
                raise ValueError('svd_starategy should be svd_with_match or svd_without_match')
            converted_vertices = _transform_vertices(T)
            infra_mins, infra_maxs = _bounds_from_vertices(converted_vertices)
            match_count = 0
            for infra_idx in range(num_infra):
                key = str(infra_types[infra_idx]).lower()
                veh_indices = type_to_vehicle.get(key)
                if not veh_indices:
                    continue
                infra_box_vertices = converted_vertices[infra_idx]
                infra_min = infra_mins[infra_idx]
                infra_max = infra_maxs[infra_idx]
                for veh_idx in veh_indices:
                    veh_min = veh_mins[veh_idx]
                    veh_max = veh_maxs[veh_idx]
                    if (
                        infra_max[0] < veh_min[0]
                        or infra_min[0] > veh_max[0]
                        or infra_max[1] < veh_min[1]
                        or infra_min[1] > veh_max[1]
                        or infra_max[2] < veh_min[2]
                        or infra_min[2] > veh_max[2]
                    ):
                        continue
                    iou = cal_3dIoU(infra_box_vertices, vehicle_vertices[veh_idx])
                    if iou > 0:
                        match_count += 1
            if match_count > 0:
                KP[i, j] = float(match_count - 1) * infra_conf[i] * vehicle_conf[j]
                max_matches_num = max(max_matches_num, match_count)
    return KP, max_matches_num


def cal_core_KP_distance(infra_object_list, vehicle_object_list, category_flag = True, core_similarity_component = 'centerpoint_distance', distance_threshold = 3, svd_starategy = 'svd_with_match', parallel = False):
    max_matches_num = -1
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    for i, infra_bbox_object in enumerate(infra_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_object_list):
            if category_flag:
                if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                    continue
            if svd_starategy == 'svd_with_match':
                T = get_extrinsic_from_two_3dbox_object(infra_bbox_object, vehicle_bbox_object)
            elif svd_starategy == 'svd_without_match':
                T = get_extrinsic_from_two_3dbox_object_svd_without_match(infra_bbox_object, vehicle_bbox_object)
            else:
                raise ValueError('svd_starategy should be svd_with_match or svd_without_match')
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, infra_object_list)
            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_object_list, core_similarity_component=core_similarity_component, distance_threshold=distance_threshold, parallel=parallel)
            KP[i, j] = int(corresponding_detector.get_Yscore() )#* infra_bbox_object.get_confidence() * vehicle_bbox_object.get_confidence() * 2
            if max_matches_num < corresponding_detector.get_matched_num():
                max_matches_num = corresponding_detector.get_matched_num()
    return KP, max_matches_num
    # return KP, normalized_KP(KP), max_matches_num


def process_task(args):
    """并行处理单个(i,j)对的函数"""
    i, j = args
    # 从全局变量获取数据
    infra_obj = global_infra_list[i]
    vehicle_obj = global_vehicle_list[j]
    
    # 类别检查（尽管任务已预过滤，保留以防万一）
    if global_category_flag and infra_obj.get_bbox_type() != vehicle_obj.get_bbox_type():
        return (i, j, 0.0, -1)
    
    # 计算变换矩阵并转换基础设施检测框
    T = get_extrinsic_from_two_3dbox_object(infra_obj, vehicle_obj)
    converted_infra = implement_T_3dbox_object_list(T, global_infra_list)
    
    # 计算匹配得分和匹配数
    detector = CorrespondingDetector(
        converted_infra, global_vehicle_list,
        core_similarity_component=global_core_sim,
        distance_threshold=global_distance_thresh,
        parallel=global_parallel_flag
    )
    score = int(detector.get_Yscore())
    matched_num = detector.get_matched_num()
    return (i, j, score, matched_num)


def cal_descriptor_similarity(infra_object_list, vehicle_object_list, weight=1.0, metric='cosine'):
    num_infra = len(infra_object_list)
    num_vehicle = len(vehicle_object_list)
    KP = np.zeros((num_infra, num_vehicle), dtype=np.float32)
    if weight <= 0:
        return KP
    infra_desc = []
    vehicle_desc = []
    for obj in infra_object_list:
        descriptor = getattr(obj, 'descriptor', None)
        if descriptor is None and hasattr(obj, 'get_descriptor'):
            descriptor = obj.get_descriptor()
        infra_desc.append(descriptor)
    for obj in vehicle_object_list:
        descriptor = getattr(obj, 'descriptor', None)
        if descriptor is None and hasattr(obj, 'get_descriptor'):
            descriptor = obj.get_descriptor()
        vehicle_desc.append(descriptor)
    eps = 1e-6
    for i, desc_i in enumerate(infra_desc):
        if desc_i is None:
            continue
        vec_i = np.asarray(desc_i, dtype=np.float32)
        if metric == 'cosine':
            norm_i = np.linalg.norm(vec_i)
            if norm_i < eps:
                continue
        for j, desc_j in enumerate(vehicle_desc):
            if desc_j is None:
                continue
            vec_j = np.asarray(desc_j, dtype=np.float32)
            if metric == 'cosine':
                norm_j = np.linalg.norm(vec_j)
                if norm_j < eps:
                    continue
                sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j + eps))
                KP[i, j] = weight * max(0.0, sim)
            elif metric == 'l2':
                dist = np.linalg.norm(vec_i - vec_j)
                KP[i, j] = weight * np.exp(-dist)
            else:
                # default cosine if unknown metric
                norm_i = np.linalg.norm(vec_i)
                if norm_i < eps:
                    continue
                norm_j = np.linalg.norm(vec_j)
                if norm_j < eps:
                    continue
                sim = float(np.dot(vec_i, vec_j) / (norm_i * norm_j + eps))
                KP[i, j] = weight * max(0.0, sim)
    return KP

def init_pool(infra_list, vehicle_list, category_flag, core_sim, distance_thresh, parallel_flag):
    """初始化子进程的全局变量"""
    global global_infra_list, global_vehicle_list, global_category_flag
    global global_core_sim, global_distance_thresh, global_parallel_flag
    global_infra_list = infra_list
    global_vehicle_list = vehicle_list
    global_category_flag = category_flag
    global_core_sim = core_sim
    global_distance_thresh = distance_thresh
    global_parallel_flag = parallel_flag

def cal_core_KP_distance_parallel(
    infra_object_list, vehicle_object_list,
    category_flag=True, core_similarity_component='centerpoint_distance',
    distance_threshold=3, num_processes=None, parallel=False
):
    # 生成所有可能的(i,j)对，并预过滤
    tasks = []
    for i, j in product(range(len(infra_object_list)), range(len(vehicle_object_list))):
        infra_obj = infra_object_list[i]
        vehicle_obj = vehicle_object_list[j]
        if category_flag and infra_obj.get_bbox_type() != vehicle_obj.get_bbox_type():
            continue
        tasks.append((i, j))
    
    # 配置进程池
    num_processes = num_processes or mp.cpu_count()
    init_args = (
        infra_object_list, vehicle_object_list,
        category_flag, core_similarity_component, distance_threshold, parallel
    )
    
    # 并行处理任务
    with mp.Pool(
        processes=num_processes,
        initializer=init_pool,
        initargs=init_args
    ) as pool:
        results = pool.map(process_task, tasks)
    
    # 合并结果
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    max_matches_num = -1
    for i, j, score, matched_num in results:
        KP[i, j] = score
        if matched_num > max_matches_num:
            max_matches_num = matched_num
    return KP, max_matches_num





def cal_other_edge_KP(infra_object_list, vehicle_object_list, category_flag = True, similarity_strategy = 'length'):
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    for i, infra_bbox_object in enumerate(infra_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_object_list):
            if category_flag:
                if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                    continue
            KP[i, j] = int(cal_similarity_knn(infra_object_list, i, vehicle_object_list, j, similarity_strategy=similarity_strategy))
    return normalized_KP(KP)

def cal_other_vertex_KP(infra_object_list, vehicle_object_list, category_flag = True, similarity_strategy = 'size'):
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    for i, infra_bbox_object in enumerate(infra_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_object_list):
            if category_flag:
                if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                    continue
            if similarity_strategy == 'size':
                similarity = int(cal_similarity_size(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3()))
            else:
                raise ValueError('similarity_strategy should be size')
            KP[i, j] = similarity
    return normalized_KP(KP)


def crop_image_with_box2d(image, box2d):
    x1, y1, x2, y2 = box2d
    # print('x1, y1, x2, y2: ', x1, y1, x2, y2)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    return image[y1:y2, x1:x2]

# def cal_appearance_KP(infra_object_list, vehicle_object_list, image_list):
#     KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
#     for i, infra_bbox_object in enumerate(infra_object_list):
#         for j, vehicle_bbox_object in enumerate(vehicle_object_list):
#             if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
#                 continue
#             KP[i, j] = int(cal_appearance_similarity(crop_image_with_box2d(image_list[0], infra_bbox_object.get_bbox2d_4()), crop_image_with_box2d(image_list[1], vehicle_bbox_object.get_bbox2d_4())))
#     return normalized_KP(KP)



def cal_similarity_size(infra_bbox_8_3, vehicle_bbox_8_3):
    '''
        使用 3D IoU 计算两个框的大小相似度
        构成 size 的 lwh 是包含了角度的信息，不能反映单纯size所要所表现出的意义
    '''
    il, iw, ih = get_lwh_from_bbox3d_8_3(infra_bbox_8_3)
    vl, vw, vh = get_lwh_from_bbox3d_8_3(vehicle_bbox_8_3)

    infra_box = get_bbox3d_8_3_from_xyz_lwh([0, 0, 0], [il, iw, ih])
    vehicle_box = get_bbox3d_8_3_from_xyz_lwh([0, 0, 0], [vl, vw, vh])

    similarity_size = cal_3dIoU(np.array(infra_box), np.array(vehicle_box))

    return similarity_size

def cal_similarity_angle(infra_bbox_2_8_3, vehicle_bbox_2_8_3):
    # 计算两个边的中心点
    infra_vector = get_vector_between_bbox3d_8_3(infra_bbox_2_8_3[0], infra_bbox_2_8_3[1])
    vehicle_vector = get_vector_between_bbox3d_8_3(vehicle_bbox_2_8_3[0], vehicle_bbox_2_8_3[1])

    # 计算向量之间的夹角
    # angle = np.arccos(np.clip(np.dot(infra_vector, vehicle_vector.T) / 
    #                           (np.linalg.norm(infra_vector) * np.linalg.norm(vehicle_vector)), -1.0, 1.0))
    
    # # 将夹角转换为度数
    # angle_degree = np.degrees(angle)

    # # 余弦相似度相似度 
    # similarity_angle = np.cos(angle) 

    # similarity_angle = np.clip(np.dot(infra_vector, vehicle_vector.T) / 
    #                           (np.linalg.norm(infra_vector) * np.linalg.norm(vehicle_vector)), -1.0, 1.0)
    
    similarity_angle = np.dot(infra_vector, vehicle_vector.T) / (np.linalg.norm(infra_vector) * np.linalg.norm(vehicle_vector))

    

    # if similarity_angle == n
    # return np.abs(similarity_angle)
    return similarity_angle

def cal_similarity_length(infra_bbox_2_8_3, vehicle_bbox_2_8_3):
    '''
        1. 用中心点计算两个框的长度相似度
        2. 用几何体最近点计算两个框的长度相似度
    
    '''
    infra_length = get_length_between_bbox3d_8_3(infra_bbox_2_8_3[0], infra_bbox_2_8_3[1])
    vehicle_length = get_length_between_bbox3d_8_3(vehicle_bbox_2_8_3[0], vehicle_bbox_2_8_3[1])

    similarity_length = 1 - np.abs(infra_length - vehicle_length) / np.max([infra_length, vehicle_length])

    if similarity_length == 0:
        if infra_length == 0:
            print('infra_length is 0')
            # print('infra_bbox_2_8_3[0]:', infra_bbox_2_8_3[0])
            # print('infra_bbox_2_8_3[1]:', infra_bbox_2_8_3[1])
        if vehicle_length == 0:
            print('vehicle_length is 0')
            # print('vehicle_bbox_2_8_3[0]:', vehicle_bbox_2_8_3[0])
            # print('vehicle_bbox_2_8_3[1]:', vehicle_bbox_2_8_3[1])

    return similarity_length


def get_KNN_points(box_object_list, index, k):
    selected_box_object = box_object_list[index]
    distances = []
    selected_box_object_list = []
    for box_object in box_object_list:
        if box_object != selected_box_object:
            distances.append(get_length_between_bbox3d_8_3(selected_box_object.get_bbox3d_8_3(), box_object.get_bbox3d_8_3()))
            selected_box_object_list.append(box_object)

    sorted_index = np.argsort(distances)

    pair_points =  [box_object for box_object in np.array(selected_box_object_list)[sorted_index][:k]]
    return pair_points

# similarity += count_knn_similarity(infra_pair_point, infra_object_list[infra_index], vehicle_pair_point, vehicle_object_list[vehicle_index])
def count_knn_similarity(edge1_point, edge1_start_point, edge2_point, edge2_start_point):
    # length_similar
    length_similar = cal_similarity_length((edge1_start_point.get_bbox3d_8_3(), edge1_point.get_bbox3d_8_3()), (edge2_start_point.get_bbox3d_8_3(), edge2_point.get_bbox3d_8_3()))
    if length_similar < 0.95:
        length_similar = 0

    # # size_similar
    # size_similar = cal_similarity_size(edge1_point.get_bbox3d_8_3(), edge2_point.get_bbox3d_8_3())

    # angle_similar
    # angle_similar = cal_similarity_angle((edge1_start_point.get_bbox3d_8_3(), edge1_point.get_bbox3d_8_3()), (edge2_start_point.get_bbox3d_8_3(), edge2_point.get_bbox3d_8_3()))
    # if angle_similar < 0.95:
    #     angle_similar = 0

    # print('length_similar:', length_similar)
    # print('size_similar:', size_similar)
    # print('angle_similar:', angle_similar)
    # print(length_similar + size_similar + angle_similar)
    # print('-----------------------------------')
    
    return length_similar #+ angle_similar #+ size_similar

def cal_similarity_knn(infra_object_list, infra_index, vehicle_object_list, vehicle_index, similarity_strategy = 'length', k = 0):
    k_infra, k_vehicle = k, k

    if k > len(infra_object_list) - 1:
        k_infra = len(infra_object_list) - 1
    if k > len(vehicle_object_list) - 1:
        k_vehicle = len(vehicle_object_list) - 1
        
    if k == 0:
        k_infra, k_vehicle = len(infra_object_list) - 1, len(vehicle_object_list) - 1
        
    infra_pair_points = get_KNN_points(infra_object_list, infra_index, k_infra)
    vehicle_pair_points = get_KNN_points(vehicle_object_list, vehicle_index, k_vehicle)
    
    similarity = 0

    for infra_pair_point in infra_pair_points:
        for vehicle_pair_point in vehicle_pair_points:
            if infra_pair_point.get_bbox_type() != vehicle_pair_point.get_bbox_type():
                continue
            infra_pair = (infra_object_list[infra_index].get_bbox3d_8_3(), infra_pair_point.get_bbox3d_8_3())
            vehicle_pair = (vehicle_object_list[vehicle_index].get_bbox3d_8_3(), vehicle_pair_point.get_bbox3d_8_3())
            if similarity_strategy == 'length':
                similar = cal_similarity_length(infra_pair, vehicle_pair)
            elif similarity_strategy == 'angle':
                similar = cal_similarity_angle(infra_pair, vehicle_pair)

            if similar < 0.95:
                similar = 0
            # similarity += count_knn_similarity(infra_pair_point, infra_object_list[infra_index], vehicle_pair_point, vehicle_object_list[vehicle_index])
            similarity += similar

    return similarity


def test_similarity_size(infra_object_list, vehicle_object_list):
    import matplotlib.pyplot as plt

    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    for i, infra_object in enumerate(infra_object_list):
        for j, vehicle_object in enumerate(vehicle_object_list):
            if infra_object.get_bbox_type() != vehicle_object.get_bbox_type():
                continue
            KP[i, j] = cal_similarity_size(infra_object.get_bbox3d_8_3(), vehicle_object.get_bbox3d_8_3()) * 10
    

    i = KP.nonzero()
    data = KP[i].flatten()
    print('before nonzero')
    print(len(KP.flatten()))
    print('after nonzero')
    print(len(data))
    print('data below 1')
    print(len(data[data<1]))
    print('data between 1 and 2')
    print(len(data[(data>=1) & (data<2)]))
    print('data between 2 and 3')
    print(len(data[(data>=2) & (data<3)]))
    print('data between 3 and 4')
    print(len(data[(data>=3) & (data<4)]))
    print('data between 4 and 5')
    print(len(data[(data>=4) & (data<5)]))
    print('data between 5 and 6')
    print(len(data[(data>=5) & (data<6)]))
    print('data between 6 and 7')
    print(len(data[(data>=6) & (data<7)]))
    print('data between 7 and 8')
    print(len(data[(data>=7) & (data<8)]))
    print('data between 8 and 9')
    print(len(data[(data>=8) & (data<9)]))
    print('data between 9 and 10')
    print(len(data[(data>=9) & (data<10)]))
    print(list(data))
    fig, ax = plt.subplots()
    ax.boxplot(data, patch_artist=True)
    plt.show()

def test_similarity_length(infra_object_list, vehicle_object_list):
    pass

def test_similarity_angle(infra_object_list, vehicle_object_list):
    pass
