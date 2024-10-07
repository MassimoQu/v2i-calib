import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import cal_3dIoU, get_lwh_from_bbox3d_8_3, get_bbox3d_8_3_from_xyz_lwh, get_vector_between_bbox3d_8_3, get_length_between_bbox3d_8_3, get_time_judge, implement_T_3dbox_object_list, get_extrinsic_from_two_3dbox_object, convert_T_to_6DOF
from CorrespondingDetector import CorrespondingDetector


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


def cal_core_KP_distance(infra_object_list, vehicle_object_list, category_flag = True, core_similarity_component = 'centerpoint_distance', distance_threshold = 3):
    max_matches_num = -1
    KP = np.zeros((len(infra_object_list), len(vehicle_object_list)), dtype=np.float64)
    for i, infra_bbox_object in enumerate(infra_object_list):
        for j, vehicle_bbox_object in enumerate(vehicle_object_list):
            if category_flag:
                if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                    continue
            T = get_extrinsic_from_two_3dbox_object(infra_bbox_object, vehicle_bbox_object)
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, infra_object_list)
            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_object_list, core_similarity_component=core_similarity_component, distance_threshold=distance_threshold)
            KP[i, j] = int(corresponding_detector.get_Yscore() )#* infra_bbox_object.get_confidence() * vehicle_bbox_object.get_confidence() * 2
            if max_matches_num < corresponding_detector.get_matched_num():
                max_matches_num = corresponding_detector.get_matched_num()
    return KP, max_matches_num
    # return KP, normalized_KP(KP), max_matches_num

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
