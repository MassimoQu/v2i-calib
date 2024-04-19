import time
import json
import numpy as np
import cv2
import sys 
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/reader')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/utils')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/process/corresponding')
sys.path.append('/home/massimo/vehicle_infrastructure_calibration/visualize')

from CooperativeBatchingReader import CooperativeBatchingReader
from CooperativeReader import CooperativeReader
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
from scipy.optimize import linear_sum_assignment
from extrinsic_utils import get_time_judge, implement_T_3dbox_object_list, get_extrinsic_from_two_3dbox_object,convert_T_to_6DOF
import similarity_utils
from BBoxVisualizer_open3d import BBoxVisualizer_open3d


class BoxesMatch():

    def __init__(self,infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle = None, verbose=False, image_list = None, similarity_strategy = ['core', 'category']):

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
                KP, max_matches_num = similarity_utils.cal_core_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in similarity_strategy))
                self.KP += KP
            else:
                max_matches_num = -1

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
       

    def cal_matches_accuracy(self, matches=None):
        if matches is None:
            matches = self.matches

        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        for match in matches:
            matched_infra_bboxes_object_list.append(self.infra_boxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(self.vehicle_boxes_object_list[match[1]])

        converted_infra_boxes_object_list = implement_T_3dbox_object_list(self.T_infra2vehicle, self.infra_boxes_object_list)
        
        self.total_matches = CorrespondingDetector(converted_infra_boxes_object_list, self.vehicle_boxes_object_list).corresponding_IoU_dict.keys()
        
        for match in matches:
            if match in self.total_matches:
                self.result_matches.append(match)
                
        cnt = len(self.result_matches)

        cnt, total_matches_cnt, given_matches_num =  cnt, len(self.total_matches), len(matches)

        if self.verbose:
            print('true result matches / true matches / total: {} / {} / {}'.format(cnt, total_matches_cnt, given_matches_num))
            print('Accuracy(true result matches / true matches): ', cnt / total_matches_cnt) if total_matches_cnt > 0 else print('Accuracy(true result matches / true matches): ', 0)

            infra_result_matches_types_list, vehicle_result_matches_types_list = self.get_matches_types(self.result_matches)
            infra_result_matches_types_list = sorted(infra_result_matches_types_list)
            vehicle_result_matches_types_list = sorted(vehicle_result_matches_types_list)
            print('infra_result_matches_types_list: ', infra_result_matches_types_list)
            print('vehicle_result_matches_types_list: ', vehicle_result_matches_types_list)

            infra_total_matches_types_list, vehicle_total_matches_types_list = self.get_matches_types(self.total_matches)
            infra_total_matches_types_list = sorted(infra_total_matches_types_list)
            vehicle_total_matches_types_list = sorted(vehicle_total_matches_types_list)
            infra_missing_matches_types_list = list(set(infra_total_matches_types_list) - set(infra_result_matches_types_list))
            vehicle_missing_matches_types_list = list(set(vehicle_total_matches_types_list) - set(vehicle_result_matches_types_list))
            print('infra_missing_matches_types_list: ', infra_missing_matches_types_list)
            print('vehicle_missing_matches_types_list: ', vehicle_missing_matches_types_list)

        return cnt, total_matches_cnt, given_matches_num
    
    def get_matches_types(self, matches):
        infra_result_matches_types = []
        vehicle_result_matches_types = []
        for match in matches:
            infra_result_matches_types.append(self.infra_boxes_object_list[match[0]].get_bbox_type())
            vehicle_result_matches_types.append(self.vehicle_boxes_object_list[match[1]].get_bbox_type())
        return infra_result_matches_types, vehicle_result_matches_types

def specific_test_boxes_match(infra_num = '003920', vehicle_num = '020092', k = 10):
    infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader(infra_num, vehicle_num).get_cooperative_infra_vehicle_boxes_object_list()
    T_infra2vehicle = CooperativeReader(infra_num, vehicle_num).get_cooperative_T_i2v()
    
    infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
    vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)
    
    task = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle, verbose=True)
    task.cal_matches_accuracy()
    task.output_intermediate_KP()


def batching_test_boxes_match(verbose = False, k = 10):
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    valid_matches_list = []
    non_matches_list = []
    error_matches_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list():
        # if cnt > 10:
        #     break

        if verbose:
            print('infra: {}   vehicle: {}'.format(infra_file_name, vehicle_file_name))

        filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
        filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)

        try:
            start_time = time.time()  # 开始计时

            boxes_matcher = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, T_infra2vehicle, verbose)
            matches_cnt, available_matches_cnt, filtered_cnt = boxes_matcher.cal_matches_accuracy()
            
            end_time = time.time()  # 结束计时

            if matches_cnt > 0:
                valid_matches = {}
                valid_matches['infra_file_name'] = infra_file_name
                valid_matches['vehicle_file_name'] = vehicle_file_name
                valid_matches['matches_cnt'] = matches_cnt
                valid_matches['available_matches_cnt'] = available_matches_cnt
                valid_matches['filtered_cnt'] = filtered_cnt
                valid_matches['infra_total_box_cnt'] = len(infra_boxes_object_list)
                valid_matches['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                valid_matches['match_module_cost_time'] = end_time - start_time
                valid_matches_list.append(valid_matches)

            else:
                non_matches = {}
                non_matches['infra_file_name'] = infra_file_name
                non_matches['vehicle_file_name'] = vehicle_file_name
                non_matches['matches_cnt'] = matches_cnt
                non_matches['available_matches_cnt'] = available_matches_cnt
                non_matches['filtered_cnt'] = filtered_cnt
                non_matches['infra_total_box_cnt'] = len(infra_boxes_object_list)
                non_matches['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                non_matches['match_module_cost_time'] = end_time - start_time
                non_matches_list.append(non_matches)


        except Exception as e:
            if verbose:
                print('Error: ', infra_file_name, vehicle_file_name)
                print(e)
            error_matches = {}
            error_matches['infra_file_name'] = infra_file_name
            error_matches['vehicle_file_name'] = vehicle_file_name
            error_matches["error_message"] = str(e)
            error_matches_list.append(error_matches)
            
        cnt += 1
        print(cnt)
        if verbose:
            print('---------------------------------')

        if cnt % 100 == 0:
            if len(valid_matches_list):
                with open(f'intermediate_output/successful_matches_k{k}_cnt{cnt}.json', 'w') as f:
                    json.dump(valid_matches_list, f)

            if len(non_matches_list):
                with open(f'intermediate_output/non_matches_k{k}_cnt{cnt}.json', 'w') as f:
                    json.dump(non_matches_list, f)

            if len(error_matches_list):
                with open(f'intermediate_output/error_matches_k{k}_cnt{cnt}.json', 'w') as f:
                    json.dump(error_matches_list, f)

            valid_matches_list = []
            non_matches_list = []
            error_matches_list = []

            print('----------------write to file---------------------')

    if len(valid_matches_list):
        with open(f'intermediate_output/successful_matches_k{k}_cnt{cnt}.json', 'w') as f:
            json.dump(valid_matches_list, f)

    if len(non_matches_list):
        with open(f'intermediate_output/non_matches_k{k}_cnt{cnt}.json', 'w') as f:
            json.dump(non_matches_list, f)

    if len(error_matches_list):
        with open(f'intermediate_output/error_matches_k{k}_cnt{cnt}.json', 'w') as f:
            json.dump(error_matches_list, f)



def test_get_matches_with_score(infra_bboxes_object_list, vehicle_bboxes_object_list):
    boxes_matcher = BoxesMatch(infra_bboxes_object_list, vehicle_bboxes_object_list)
    matches_with_score = boxes_matcher.get_matches_with_score()
    print('matches_with_score: ', matches_with_score)
    for match in matches_with_score:
        print('match: ', match[0])
        print('score: ', match[1])
    
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
    1 == 1
