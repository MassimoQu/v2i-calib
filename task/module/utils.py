from scipy.spatial.transform import Rotation
import numpy as np
import time


def get_time(f):
    def inner(*arg,**kwarg):
        print('开始计时')
        s_time = time.time()
        res = f(*arg,**kwarg)
        e_time = time.time()
        print('耗时：{}秒'.format(e_time - s_time))
        return res
    return inner

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

def implement_T_points_n_3(T, points):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_points_n_3(R, t, points)    

def implement_R_t_3dbox_n_8_3(R, t, boxes):
    return implement_R_t_points_n_3(R, t, boxes).reshape(-1, 8, 3)

def implement_R_t_3dbox_dict_n_8_3(R, t, boxes_dict):
    for box_type, boxes in boxes_dict.items():
        boxes_dict[box_type] = implement_R_t_3dbox_n_8_3(R, t, boxes)
    return boxes_dict

def implement_T_3dbox_n_8_3(T, boxes):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_3dbox_n_8_3(R, t, boxes)

def implement_T_3dbox_dict_n_8_3(T, boxes_dict):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_3dbox_dict_n_8_3(R, t, boxes_dict)

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

#todo: 常见的外惨转换关系，可以和上面的整合一下打包成一个类
