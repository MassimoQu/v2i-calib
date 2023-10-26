from scipy.spatial.transform import Rotation
import numpy as np

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
    T[:3, :3] = R
    T[:3, 3] = t
    return T

