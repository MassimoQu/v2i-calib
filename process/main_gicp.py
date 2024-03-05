import numpy as np
import os
import time
import json
import sys
sys.path.append('./reader')
sys.path.append('./process')
sys.path.append('./process/graph')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./process/search')
sys.path.append('./visualize')
sys.path.append('./traditional_registration')
from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from BoxesMatch import BoxesMatch
import BoxesMatch_cpp
# from PSO_Executor import PSO_Executor
from Matches2Extrinsics import Matches2Extrinsics
from extrinsic_utils import implement_T_3dbox_object_list, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, get_reverse_T, convert_Rt_to_T
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
import pygicp

# pso
# def cal_extrinsic_from_two_box_object_list(box_object_list1, box_object_list2, filter_num = 15, verbose = False):
#     boxes_match = BoxesMatch(box_object_list1, box_object_list2)
#     matches = boxes_match.get_matches()
#     if matches == []:
#         return None
#     pso = PSO_Executor(box_object_list1, box_object_list2, matches=matches, filter_num=filter_num, verbose=verbose, turn_off_pso=False)
#     return pso.get_best_T_6DOF(), matches

def convert_py_bbox_to_cpp(bbox):
    bbox_8_3 = np.array(bbox.get_bbox3d_8_3(), dtype=np.float32)
    return BoxesMatch_cpp.BoxObject(bbox_8_3, bbox.get_bbox_type())

def get_matches_with_score_py(infra_boxes, vehicle_boxes):
    infra_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in infra_boxes]
    vehicle_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in vehicle_boxes]
    return BoxesMatch_cpp.get_matches_with_score(infra_boxes_cpp, vehicle_boxes_cpp)

# combination & svd
def cal_extrinsic_from_two_box_object_list(box_object_list1, box_object_list2, filter_num = 15, verbose = False, using_cpp_version = False, true_matches = None, optimize_combination = True):
    if using_cpp_version:
        matches_with_score_list = get_matches_with_score_py(box_object_list1, box_object_list2)
    else:
        boxes_match = BoxesMatch(box_object_list1, box_object_list2)
        matches_with_score_list = boxes_match.get_matches_with_score()
    
    matches = [match[0] for match in matches_with_score_list]
    if matches == []:
        return None
    matches2extrinsics = Matches2Extrinsics(box_object_list1, box_object_list2, matches_score_list=matches_with_score_list, verbose=verbose, true_matches=true_matches)
    if optimize_combination:
        return matches2extrinsics.get_combined_extrinsic(), matches
    else:
        return matches2extrinsics.get_single_extrinsic(), matches

def batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, using_cpp_version = False, using_predict_score = False, output_dict = 'intermediate_output'):
    reader = CooperativeBatchingReader(path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/hard_data_info.json')
    cnt = 0 

    valid_test_list = []
    valid_bad_test_list = []

    error_list = []

    for infra_file_name, vehicle_file_name, infra_pointcloud, vehicle_pointcloud, T_true in reader.generate_infra_vehicle_pointcloud():
        
        # try:
        if True:

            if verbose:
                print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
                
            ##################
            start_time = time.time()

            T_6DOF_result = convert_T_to_6DOF(pygicp.align_points(infra_pointcloud, vehicle_pointcloud, method='GICP'))
            
            end_time = time.time()
            ##################

            T_6DOF_true = convert_T_to_6DOF(T_true)
            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF_result, T_6DOF_true)

            valid_test = {}
            valid_test['infra_file_name'] = infra_file_name
            valid_test['vehicle_file_name'] = vehicle_file_name
            valid_test['delta_T_6DOF'] = (T_6DOF_true - T_6DOF_result).tolist()
            valid_test['RE'] = RE.tolist()
            valid_test['TE'] = TE.tolist()
            valid_test['cost_time'] = end_time - start_time

            if RE < 2 and TE < 2:
                valid_test_list.append(valid_test)
            else:
                valid_bad_test_list.append(valid_test)


        # except Exception as e:
        #     if verbose:
        #         print('Error: ', infra_file_name, vehicle_file_name)
        #         print(e)
        #     error_ = {}
        #     error_['infra_file_name'] = infra_file_name
        #     error_['vehicle_file_name'] = vehicle_file_name
        #     error_["error_message"] = str(e)
        #     error_list.append(error_)
        #     assert

        print(cnt)
        cnt += 1
        
        if verbose:
            print(f'delta_T_6DOF: {(T_6DOF_true - T_6DOF_result).tolist()}')
            print(f'RE: {RE.tolist()}')
            print(f'TE: {TE.tolist()}')
            print(f'cost time: {end_time - start_time}')
            print('---------------------------------')

        if cnt % 50 == 0:

            # if len(valid_test_list):
            with open(os.path.join(output_dict, f'valid_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(valid_test_list, f)

            # if len(valid_bad_test_list):
            with open(os.path.join(output_dict, f'valid_bad_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(valid_bad_test_list, f)

            # if len(error_list):
            with open(os.path.join(output_dict, f'error_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(error_list, f)

            valid_test_list = []
            valid_bad_test_list = []
            error_list = []

            print('----------------write to file---------------------')
    

if __name__ == '__main__':
    batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, using_predict_score=False, using_cpp_version=False,
                                                    output_dict='intermediate_output/gicp/hard_dataset')

    # cooperative_reader = CooperativeReader('014218', '012589')
    # infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()

    # k = 15
    # filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
    # filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)

    # T_6DOF_result, matches = cal_extrinsic_from_two_box_object_list(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, filter_num=k, verbose=False)

    # print('T_6DOF_result: ', T_6DOF_result)
    # print('matches: ', matches)

    # print(cal_extrinsic_from_two_box_object_list(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, filter_num=k, verbose=False))
    
    # batching_test_extrisic_from_two_box_object_list_appearance(verbose = False, filter_num = 15)
