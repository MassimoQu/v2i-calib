from scipy.spatial.transform import Rotation
import numpy as np
import time
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
    T[:3, [3]] = np.array(t)
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
    points = points.reshape(-1, 3).T
    converted_points = np.dot(R, points) + t.reshape(3, 1)
    converted_points = converted_points.T.reshape(-1, 3)
    return converted_points

def implement_R_t_3dbox_n_8_3(R, t, boxes):
    return implement_R_t_points_n_3(R, t, boxes).reshape(-1, 8, 3)

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
    # assert points1.shape[0] == 3
    # assert points1.shape[1] >= 3

    centroid1 = np.mean(points1, axis=0)
    centroid2 = np.mean(points2, axis=0)

    points1 = points1 - centroid1
    points2 = points2 - centroid2

    H = np.dot(points1.T, points2)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = -np.dot(R, centroid1.T) + centroid2.T

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t.flatten()  

    return T

def get_extrinsic_from_two_3dbox_object(box_object_1, box_object_2):
    points1 = box_object_1.get_bbox3d_8_3()
    points2 = box_object_2.get_bbox3d_8_3()
    return get_extrinsic_from_two_points(points1, points2)
    
def get_extrinsic_from_two_mixed_3dbox_object_list(box_object_list_1, box_object_list_2):
    points1 = np.concatenate([box_object.get_bbox3d_8_3() for box_object in box_object_list_1], axis=0)
    points2 = np.concatenate([box_object.get_bbox3d_8_3() for box_object in box_object_list_2], axis=0)
    return get_extrinsic_from_two_points(points1, points2)
    
# HPCR-VI (2023IV)
def get_RE_TE_by_compare_T_6DOF_result_true(T1_6DOF, T2_6DOF):
    # RE : rad
    # TE : m
    R1, t1 = convert_T_to_Rt(convert_6DOF_to_T(T1_6DOF))
    R2, t2 = convert_T_to_Rt(convert_6DOF_to_T(T2_6DOF))
    RE = np.arccos((np.trace(np.dot(R1, R2.T)) - 1) / 2) * 180 / np.pi
    TE = np.linalg.norm(t1 - t2)
    return RE, TE
    