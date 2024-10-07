from scipy.spatial.transform import Rotation
import numpy as np
import time
# import pygicp
from functools import wraps

def get_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken by {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper

def get_time_judge(verbose):
    def get_time(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if verbose:
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                print(f"Time taken by {func.__name__}: {end_time - start_time} seconds")
                return result
            else:
                return func(*args, **kwargs)
        return wrapper
    return get_time


def convert_T_to_6DOF(T):
    R = T[:3, :3]
    t = T[:3, 3]
    r = Rotation.from_matrix(R)
    euler = r.as_euler('xyz', degrees=True) # roll pitch yaw
    return np.concatenate((t, euler))

def convert_6DOF_to_T(x):
    t = x[:3]
    euler = x[3:]
    r = Rotation.from_euler('xyz', euler, degrees=True)
    R = r.as_matrix()
    T = np.eye(4)
    T[:3, :3] = np.array(R)
    T[:3, 3] = np.array(t)
    return T

def convert_T_to_Rt(T):
    T = np.array(T)
    R = np.array(T[:3, :3])
    t = np.array(T[:3, 3]).reshape(3, 1)
    return R, t

def convert_Rt_to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = np.array(R)
    T[:3, [3]] = np.array(t).reshape(3, 1)
    return T

def get_reverse_T(T):
    T = np.array(T)
    rev_R, rev_t = get_reverse_R_t(*convert_T_to_Rt(T))
    rev_T = convert_Rt_to_T(rev_R, rev_t)
    return rev_T

def get_reverse_R_t(R, t):
    rev_R = np.array(np.matrix(R).I)
    rev_t = -np.dot(rev_R, t)
    return rev_R, rev_t

def implement_R_t_points_n_3(R, t, points):
    R = np.array(R)
    t = np.array(t)
    converted_points = np.dot(R, points.reshape(-1, 3).T) + t.reshape(3, 1)
    converted_points = converted_points.T.reshape(-1, 3)
    return converted_points

def implement_R_t_3dbox_n_8_3(R, t, boxes):
    return implement_R_t_points_n_3(R, t, boxes).reshape(8, 3)

def implement_R_t_3dbox_object_list(R, t, box_object_list):
    converted_box_object_list = []
    for box_object in box_object_list:
        converted_box_object = box_object.copy()
        converted_box_object.bbox3d_8_3 = implement_R_t_3dbox_n_8_3(R, t, box_object.bbox3d_8_3).reshape(8, 3)
        converted_box_object_list.append(converted_box_object)
    return converted_box_object_list

def implement_T_points_n_3(T, points):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_points_n_3(R, t, points)    

def implement_T_3dbox_n_8_3(T, boxes):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_3dbox_n_8_3(R, t, boxes)

def implement_T_3dbox_object_list(T, box_object_list):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_3dbox_object_list(R, t, box_object_list)

def implement_T_to_3dbox_with_own_center(T, box):
    R, t = convert_T_to_Rt(T)
    center = np.mean(box, axis=0)
    moved_to_origin = box - center
    rotated = np.dot(R, moved_to_origin.T).T
    rotated_back = rotated + center
    return implement_R_t_3dbox_n_8_3(np.identity(3), t, rotated_back)

def multiply_extrinsics(T1, T2):
    R1, t1 = convert_T_to_Rt(T1)
    R1 = np.matrix(R1)
    t1 = np.matrix(t1)

    R2, t2 = convert_T_to_Rt(T2)
    R2 = np.matrix(R2)
    t2 = np.matrix(t2)

    R = R2 * R1
    t = R2 * t1 + t2

    return convert_Rt_to_T(np.array(R), np.array(t))

def get_extrinsic_from_two_points(points1, points2):
    # assert points1.shape == points2.shape

    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    points1 = points1 - centroid1
    points2 = points2 - centroid2

    H = np.dot(points1.T, points2)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # print('U', U)
    # print('Vt', Vt)
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = -np.dot(R, centroid1.T) + centroid2.T
    # print('centroid1', centroid1)
    # print('centroid2', centroid2)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()  

    return T

def get_extrinsic_from_two_points_weighted(points1, points2, weights):
    # 确保权重的长度与点的数量相同
    assert points1.shape == points2.shape
    if weights is not None:
        assert points1.shape[0] == weights.size

    # 计算加权中心
    total_weight = np.sum(weights)
    centroid1 = np.sum(points1 * weights[:, np.newaxis], axis=0) / total_weight
    centroid2 = np.sum(points2 * weights[:, np.newaxis], axis=0) / total_weight
    
    # 减去加权中心，进行中心化
    points1_centered = (points1 - centroid1) * np.sqrt(weights[:, np.newaxis])
    points2_centered = (points2 - centroid2) * np.sqrt(weights[:, np.newaxis])
    
    # 计算加权协方差矩阵
    H = np.dot(points1_centered.T, points2_centered)
    
    # SVD分解
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # 确保右手坐标系
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 计算平移向量
    t = -np.dot(R, centroid1.T) + centroid2.T

    # 构造4x4变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()

    return T

def get_extrinsic_from_two_3dbox_object(box_object_1, box_object_2):
    points1 = box_object_1.get_bbox3d_8_3()
    points2 = box_object_2.get_bbox3d_8_3()
    return get_extrinsic_from_two_points(points1, points2)
    
def get_extrinsic_from_two_mixed_3dbox_object_list(box_object_list_1, box_object_list_2, weights=None):
    if len(box_object_list_1) == 0 or len(box_object_list_2) == 0:
        return np.eye(4)
    points1 = np.concatenate([box_object.get_bbox3d_8_3() for box_object in box_object_list_1], axis=0)
    points2 = np.concatenate([box_object.get_bbox3d_8_3() for box_object in box_object_list_2], axis=0)
    if weights is None:
        weights = np.ones(points1.shape[0] // 8)
    # print('weights', weights)
    return get_extrinsic_from_two_points_weighted(points1, points2,  np.repeat(np.array(weights), 8))
    
# HPCR-VI (2023IV)
def get_RE_TE_by_compare_T_6DOF_result_true(T1_6DOF, T2_6DOF):
    # RE : °
    # TE : m
    R1, t1 = convert_T_to_Rt(convert_6DOF_to_T(T1_6DOF))
    R2, t2 = convert_T_to_Rt(convert_6DOF_to_T(T2_6DOF))
    val = np.trace(np.dot(R1, R2.T))
    if np.trace(np.dot(R1, R2.T)) > 3:
        val = 3
    elif val < -1:
        val = -1
    RE = np.arccos((val - 1) / 2) * 180 / np.pi
    TE = np.linalg.norm(t1 - t2)
    return RE, TE
    
# def optimize_extrinsic_from_two_points(points1, points2, initial_guess=np.eye(4)):
#     # assert points1.shape == points2.shape
#     # assert points1.shape[0] == 3
#     # assert points1.shape[1] >= 3

#     target = points1
#     source = points2

#     matrix = pygicp.align_points(target, source, initial_guess=initial_guess)

#     return matrix

# def optimize_extrinsic_from_two_3dbox_object(box_object_1, box_object_2):
#     points1 = box_object_1.get_bbox3d_8_3()
#     points2 = box_object_2.get_bbox3d_8_3()

#     return optimize_extrinsic_from_two_points(points1, points2, initial_guess=get_extrinsic_from_two_points(points1, points2))

# def optimize_extrinsic_from_two_mixed_3dbox_object_list(box_object_list_1, box_object_list_2):
#     if len(box_object_list_1) == 0 or len(box_object_list_2) == 0:
#         return np.eye(4)
#     points1 = np.concatenate([box_object.get_bbox3d_8_3() for box_object in box_object_list_1], axis=0)
#     points2 = np.concatenate([box_object.get_bbox3d_8_3() for box_object in box_object_list_2], axis=0)
#     return optimize_extrinsic_from_two_points(points1, points2, initial_guess=get_extrinsic_from_two_points(points1, points2))