import numpy as np
from scipy.optimize import linear_sum_assignment

def convert_T_to_Rt(T):
    T = np.array(T)
    R = np.array(T[:3, :3])
    t = np.array(T[:3, 3]).reshape(3, 1)
    return R, t

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

def implement_T_3dbox_object_list(T, box_object_list):
    R, t = convert_T_to_Rt(T)
    return implement_R_t_3dbox_object_list(R, t, box_object_list)

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


def get_Yscore(infra_bboxes_object_list, vehicle_bboxes_object_list):

    corresponding_IoU_dict = {}
    Y = -1

    for i, infra_bbox_object in enumerate(infra_bboxes_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_bboxes_object_list):
            if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                box3d_IoU_score = cal_3dIoU(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                if box3d_IoU_score > 0:
                    corresponding_IoU_dict[(i, j)] = box3d_IoU_score

    if len(corresponding_IoU_dict) != 0:
        Y = np.sum(list(corresponding_IoU_dict.values()))

    total_num = min(len(infra_bboxes_object_list), len(vehicle_bboxes_object_list))

    return Y / total_num


def get_matches_with_score(infra_boxes_object_list, vehicle_boxes_object_list):

    # cal KP
    infra_node_num, vehicle_node_num = len(infra_boxes_object_list), len(vehicle_boxes_object_list)
    KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)
    for i, infra_bbox_object in enumerate(infra_boxes_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_boxes_object_list):
            if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                KP[i, j] = 0
                continue
            T = get_extrinsic_from_two_3dbox_object(infra_bbox_object, vehicle_bbox_object)
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, infra_boxes_object_list)
            KP[i, j] = int(get_Yscore(converted_infra_boxes_object_list, vehicle_boxes_object_list) * 100) 

    # normalize KP
    max_val = np.max(KP)
    min_val = np.min(KP)
    for i in range(len(infra_boxes_object_list)):
        for j in range(len(vehicle_boxes_object_list)):
            if KP[i, j] != 0:
                KP[i, j] = int((KP[i, j] - min_val) / (max_val - min_val) * 10)

    # cal matches
    non_zero_rows = np.any(KP, axis=1)
    non_zero_columns = np.any(KP, axis=0)
    reduced_KP = KP[non_zero_rows][:, non_zero_columns]
    row_ind, col_ind = linear_sum_assignment(reduced_KP, maximize=True)
    original_row_ind = np.where(non_zero_rows)[0][row_ind]
    original_col_ind = np.where(non_zero_columns)[0][col_ind]
    matches = list(zip(original_row_ind, original_col_ind))
    
    # cal sorted_matches_score_dict
    matches_score_dict = {}
    for match in matches:
        if KP[match[0], match[1]] != 0:
            matches_score_dict[match] = KP[match[0], match[1]]
    sorted_matches_score_dict = sorted(matches_score_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_matches_score_dict



if __name__ == '__main__':
    pass