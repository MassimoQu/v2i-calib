import time
import json
import numpy as np
import cv2
import sys 
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/reader')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/utils')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/corresponding')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/visualize')

from CooperativeReader import CooperativeReader
from scipy.optimize import linear_sum_assignment
from extrinsic_utils import get_extrinsic_from_two_3dbox_object,convert_T_to_6DOF
import similarity_utils


class BoxesMatch():

    def __init__(self,infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle = None, verbose=False, image_list = None, similarity_strategy = ['core', 'category'], corresponding_strategy = ['centerpoint_distance','vertex_distance']):
        '''
        param:
                similarity_strategy: ['core', 'length', 'angle', 'size', 'appearance']
                corresponding_strategy:  'iou' 'centerpoint_distance' 'vertex_distance'
        '''
        self.infra_boxes_object_list, self.vehicle_boxes_object_list = infra_boxes_object_list, vehicle_boxes_object_list
        
        self.T_infra2vehicle = T_infra2vehicle
        infra_node_num, vehicle_node_num = len(self.infra_boxes_object_list), len(self.vehicle_boxes_object_list)
        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)

        self.result_matches = []
        self.total_matches = []

        self.verbose = verbose

        # @get_time_judge(verbose)
        def cal_KP():

            if 'core' in similarity_strategy:
                if corresponding_strategy == 'iou' or 'iou' in corresponding_strategy:
                    KP, max_matches_num = similarity_utils.cal_core_KP_IoU(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in similarity_strategy))
                    self.KP += KP
                else:
                    centerpoint_max_matches_num, vertexpoint_max_matches_num = 0, 0
                    if 'centerpoint_distance' in corresponding_strategy:
                        KP_centerpoint, centerpoint_max_matches_num = similarity_utils.cal_core_KP_distance(self.infra_boxes_object_list, self.vehicle_boxes_object_list, corresponding_strategy='centerpoint_distance', category_flag=('category' in similarity_strategy))
                        self.KP += KP_centerpoint
                    if 'vertex_distance' in corresponding_strategy:
                        KP_vertexpoint, vertexpoint_max_matches_num = similarity_utils.cal_core_KP_distance(self.infra_boxes_object_list, self.vehicle_boxes_object_list, corresponding_strategy='vertex_distance', category_flag=('category' in similarity_strategy))
                        self.KP += KP_vertexpoint
                        self.KP = np.round(self.KP / 2)
                    
                    max_matches_num = max(centerpoint_max_matches_num, vertexpoint_max_matches_num)
            else:
                max_matches_num = -1

            # print(self.KP)

            if 'length' in similarity_strategy:
                self.KP += similarity_utils.cal_other_edge_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in similarity_strategy), similarity_strategy='length')

            if 'angle' in similarity_strategy:
                self.KP += similarity_utils.cal_other_edge_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in similarity_strategy), similarity_strategy='angle')

            # 尝试转到合适的位置求 3dIoU 
            if 'size' in similarity_strategy:
                self.KP += similarity_utils.cal_other_vertex_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in similarity_strategy), similarity_strategy='size')

            if 'appearance' in similarity_strategy:
                if 0 < max_matches_num < 2:
                    self.KP += similarity_utils.cal_appearance_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, image_list=image_list)

        cal_KP()

        self.matches = self.get_matched_boxes_Hungarian_matching()

    def get_KP(self):
        return self.KP

    def get_matches(self):
        return self.matches

    def get_matches_with_score(self):
        matches_score_dict = {}
        for match in self.matches:
            if self.KP[match[0], match[1]] != 0:
                matches_score_dict[match] = self.KP[match[0], match[1]]
        sorted_matches_score_dict = sorted(matches_score_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches_score_dict

    def output_intermediate_KP(self):
        output_dir = './intermediate_output'
        np.savetxt(f"{output_dir}/KP_combined.csv", self.KP, delimiter=",", fmt='%d')


    def get_matched_boxes_Hungarian_matching(self):
        non_zero_rows = np.any(self.KP, axis=1)
        non_zero_columns = np.any(self.KP, axis=0)
        reduced_KP = self.KP[non_zero_rows][:, non_zero_columns]

        row_ind, col_ind = linear_sum_assignment(reduced_KP, maximize=True)
        original_row_ind = np.where(non_zero_rows)[0][row_ind]
        original_col_ind = np.where(non_zero_columns)[0][col_ind]
        matches = list(zip(original_row_ind, original_col_ind))
        return matches


    
def visualize_score_of_candidate_extrinsics(infra_id = '007038', vehicle_id = '000546'):
    cooperative_reader = CooperativeReader(infra_id, vehicle_id)
    infra_bboxes_object_list, vehicle_bboxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
    T_true_6DOF = convert_T_to_6DOF(cooperative_reader.get_cooperative_T_i2v())
    print('T_true: ', T_true_6DOF)
    # test_get_matches_with_score(infra_bboxes_object_list, vehicle_bboxes_object_list)
    boxes_matcher = BoxesMatch(infra_bboxes_object_list, vehicle_bboxes_object_list)
    matches_with_score = boxes_matcher.get_matches_with_score()

    for match, score in matches_with_score:
        infra_bbox_object = infra_bboxes_object_list[match[0]]
        vehicle_bbox_object = vehicle_bboxes_object_list[match[1]]
        T_6DOF = convert_T_to_6DOF(get_extrinsic_from_two_3dbox_object(infra_bbox_object, vehicle_bbox_object))
        print('T_6DOF: ', T_6DOF)
        print('score: ', score)
        print('------------------------------------')


if __name__ == "__main__":
    pass
