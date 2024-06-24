import sys
sys.path.append('./reader')
sys.path.append('./process/utils')

import numpy as np
from IoU_utils import cal_3dIoU
from bbox_utils import get_volume_from_bbox3d_8_3, get_xyz_from_bbox3d_8_3

class CorrespondingDetector():
    '''
    CorrespondingDetector is a class to obtain the corresponding pairs between two sets of bounding boxes.
    Here we hypothesize that the two sets of bounding boxes are already transformered by the extrinsic matrix.
    param:
        corresponding_strategy: 'IoU' or 'centerpoint_distance' or 'vertex_distance'
    '''
    def __init__(self, infra_bboxes_object_list, vehicle_bboxes_object_list, corresponding_strategy = 'centerpoint_distance'):
        self.infra_bboxes_object_list = infra_bboxes_object_list
        self.vehicle_bboxes_object_list = vehicle_bboxes_object_list

        self.corresponding_score_dict = {}
        self.Y = 0

        if corresponding_strategy == 'iou':
            self.cal_IoU_corresponding()
        elif corresponding_strategy == 'centerpoint_distance':
            self.cal_distance_corresponding(distance_strategy = 'centerpoint')
        elif corresponding_strategy == 'vertex_distance':
            self.cal_distance_corresponding(distance_strategy = 'vertexpoint')
        else:
            raise ValueError('corresponding_strategy should be one of the following: IoU, centerpoint_distance, vertex_distance')


    def cal_IoU_corresponding(self):
        for i, infra_bbox_object in enumerate(self.infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(self.vehicle_bboxes_object_list):
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    box3d_IoU_score = cal_3dIoU(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    if box3d_IoU_score > 0:
                        infra_volume = get_volume_from_bbox3d_8_3(infra_bbox_object.get_bbox3d_8_3())
                        vehicle_volume = get_volume_from_bbox3d_8_3(vehicle_bbox_object.get_bbox3d_8_3())
                        volume = (infra_volume + vehicle_volume) / 2
                        self.corresponding_score_dict[(i, j)] = box3d_IoU_score
                        # if volume >= 1:
                        #     self.corresponding_score_dict[(i, j)] = box3d_IoU_score / volume * 10
        if len(self.corresponding_score_dict) != 0:
            self.Y = np.sum(list(self.corresponding_score_dict.values()))

    def cal_distance_corresponding(self, distance_strategy = 'centerpoint', center_distance_threshold = 3):
        '''
        为了让整体 distance 作为 score 的时候 score 越大表示效果越好，对 distance 取负数，一对一匹配对约束 和 阈值筛选 获取匹配对
        '''
        occupation_dict = {}
        center_distance_threshold = -center_distance_threshold
        for i, infra_bbox_object in enumerate(self.infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(self.vehicle_bboxes_object_list):
                update_flag = False
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    if distance_strategy == 'centerpoint':
                        infra_bbox_centerpoint = get_xyz_from_bbox3d_8_3(infra_bbox_object.get_bbox3d_8_3())
                        vehicle_bbox_centerpoint = get_xyz_from_bbox3d_8_3(vehicle_bbox_object.get_bbox3d_8_3())
                        distance = -np.linalg.norm(infra_bbox_centerpoint - vehicle_bbox_centerpoint)
                    elif distance_strategy == 'vertexpoint':
                        infra_bbox_vertex = infra_bbox_object.get_bbox3d_8_3()
                        vehicle_bbox_vertex = vehicle_bbox_object.get_bbox3d_8_3()
                        distance = -np.linalg.norm(infra_bbox_vertex - vehicle_bbox_vertex)

                    if distance <= center_distance_threshold :
                        continue
                    elif i in occupation_dict.keys():
                        if distance <= self.corresponding_score_dict[(i, occupation_dict[i])]:
                            continue
                        else:
                            update_flag = True
                            del self.corresponding_score_dict[(i, occupation_dict[i])]
                            # del occupation_dict[i]
                    elif j in occupation_dict.values():
                        deleting = []
                        for k, v in occupation_dict.items():
                            if v == j:
                                if distance <= self.corresponding_score_dict[(k, j)]:
                                    break
                                else:
                                    update_flag = True
                                    del self.corresponding_score_dict[(k, j)]
                                    deleting.append(k)
                        for k in deleting:
                            del occupation_dict[k]
                    else:
                        update_flag = True

                    if update_flag:
                        # if i in occupation_dict.keys():
                        #     print(f'primary: {i} - {occupation_dict[i]} distance: {self.corresponding_score_dict[(i, occupation_dict[i])]}')
                        # else:
                        #     print(f'{i} has no primary key')
                        # print(f'update: {i} - {j} distance: {distance}')
                        self.corresponding_score_dict[(i, j)] = distance                      
                        occupation_dict[i] = j

        if len(self.corresponding_score_dict) != 0:
            self.Y = np.sum(list(self.corresponding_score_dict.values()))


    def get_matched_num(self):
        return len(self.corresponding_score_dict)
    
    def get_total_num(self):
        return min(len(self.infra_bboxes_object_list), len(self.vehicle_bboxes_object_list))
    
    def get_Yscore(self):
        total_num = self.get_total_num()
        if total_num == 0:
            return 0
        # return self.Y / self.get_total_num()
        return self.get_matched_num() - 1
    
    def get_matches(self):
        return self.corresponding_score_dict.keys()
    
    def get_matches_with_score(self):
        return self.corresponding_score_dict
    
    def get_distance_corresponding_precision(self):
        if self.get_matched_num() == 0:
            return -3
        return self.Y / self.get_matched_num()
    
    def get_matched_num_and_distance_corresponding_precision_under_threshold(self, distance_threshold = 3):
        distance_threshold = -distance_threshold
        corresponding_score_dict_under_threshold = [v for _, v in self.corresponding_score_dict.items() if v > distance_threshold]
        new_matched_num = len(corresponding_score_dict_under_threshold)
        if new_matched_num == 0:
            return 0, -3
        return new_matched_num,  sum(corresponding_score_dict_under_threshold) / new_matched_num
    
    def get_distance_corresponding_precision_under_thresholds(self, distance_thresholds = [1, 2, 3]):
        return {threshold: (self.get_matched_num_and_distance_corresponding_precision_under_threshold(threshold)) for threshold in distance_thresholds}
    
