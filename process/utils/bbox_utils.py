import numpy as np


def get_xyz_from_bbox3d_8_3(bbox3d_8_3):
    return np.mean(bbox3d_8_3, axis=0)

def get_lwh_from_bbox3d_8_3(bbox3d_8_3):
    size = np.abs(bbox3d_8_3[4] - bbox3d_8_3[2])
    l, w, h = size[0], size[1], size[2]
    return l, w, h

def get_bbox3d_8_3_from_xyz_lwh_yaw(xyz, lwh, yaw):
    l, w, h = lwh
    r = np.matrix(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    center = [xyz[0], xyz[1], xyz[2] - h / 2]
    corners_3d = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )
    corners_3d = r * corners_3d + np.matrix(center).T
    return np.array(corners_3d.T)

def get_bbox3d_8_3_from_xyz_lwh(xyz, lwh):
    return get_bbox3d_8_3_from_xyz_lwh_yaw(xyz, lwh, 0)

def get_bbox3d_n_8_3_from_bbox_object_list(bbox_object_list):
    return np.array([bbox_object.get_bbox3d_8_3() for bbox_object in bbox_object_list])
    

def get_vector_between_bbox3d_8_3(bbox3d_8_3_1, bbox3d_8_3_2):
    xyz1 = get_xyz_from_bbox3d_8_3(bbox3d_8_3_1)
    xyz2 = get_xyz_from_bbox3d_8_3(bbox3d_8_3_2)
    vector = xyz2 - xyz1
    return vector

def get_length_between_bbox3d_8_3(bbox3d_8_3_one, bbox3d_8_3_two):
    vector = get_vector_between_bbox3d_8_3(bbox3d_8_3_one, bbox3d_8_3_two)
    length = np.linalg.norm(vector)
    return length

