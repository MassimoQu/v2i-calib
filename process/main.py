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
from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from BoxesMatch import BoxesMatch
import BoxesMatch_cpp
from Matches2Extrinsics import Matches2Extrinsics
from extrinsic_utils import implement_T_3dbox_object_list, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, get_reverse_T, convert_Rt_to_T
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes


def convert_py_bbox_to_cpp(bbox):
    bbox_8_3 = np.array(bbox.get_bbox3d_8_3(), dtype=np.float32)
    return BoxesMatch_cpp.BoxObject(bbox_8_3, bbox.get_bbox_type())

def get_matches_with_score_cpp(infra_boxes, vehicle_boxes):
    infra_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in infra_boxes]
    vehicle_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in vehicle_boxes]
    return BoxesMatch_cpp.get_matches_with_score(infra_boxes_cpp, vehicle_boxes_cpp)

# combination & svd
def cal_extrinsic_from_two_box_object_list(box_object_list1, box_object_list2, verbose = False, test_cpp_speed = False, true_matches = None, image_list = None,
                                           similarity_strategy = ['core', 'category'], optimization_strategy = 'svd', matches_filter_strategy = 'trueT'):

    '''
        optimization 还可以进一步拆成 rough_strategy 和 finetuning_strategy
        
    '''

    if test_cpp_speed:
        matches_with_score_list = get_matches_with_score_cpp(box_object_list1, box_object_list2)
    else:
        boxes_match = BoxesMatch(box_object_list1, box_object_list2, image_list=image_list, similarity_strategy = similarity_strategy)
        matches_with_score_list = boxes_match.get_matches_with_score()
    
    matches = [match[0] for match in matches_with_score_list]
    if matches == []:
        return None
    
    matches2extrinsics = Matches2Extrinsics(box_object_list1, box_object_list2, matches_score_list=matches_with_score_list, true_matches=true_matches)
    return matches2extrinsics.get_combined_extrinsic(matches_filter_strategy=matches_filter_strategy, optimization_strategy=optimization_strategy), matches


def batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, test_cpp_speed = False, using_predict_score = False, output_dict = 'intermediate_output',
                                                    similarity_strategy = ['core', 'category'], optimization_strategy = 'svd', matches_filter_strategy = 'trueT', data_difficulty = 'hard'):
    if data_difficulty not in ['easy', 'hard']:
        raise ValueError('data_difficulty should be easy or hard')
    reader = CooperativeBatchingReader(path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/' + data_difficulty + '_data_info.json')

    cnt = 0

    no_common_box_list = []
    valid_test_list = []
    valid_bad_test_list = []
    invalid_test_list = []

    error_list = []

    if using_predict_score:
        wrapper = reader.generate_infra_vehicle_bboxes_object_list_predicted()
    else:
        wrapper = reader.generate_infra_vehicle_bboxes_object_list()

    
    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in wrapper:
        
        # try:
        if True:

            if verbose:
                print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
                print(f'infra_total_box_cnt: {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}')

            filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(filter_num)
            if using_predict_score:
                filtered_infra_boxes_object_list = implement_T_3dbox_object_list(get_reverse_T(T_true), filtered_infra_boxes_object_list)

            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
            filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
            filtered_available_matches = CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list).corresponding_IoU_dict.keys()
            
            converted_original_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
            total_available_matches_cnt = CorrespondingDetector(converted_original_infra_boxes_object_list, vehicle_boxes_object_list).get_matched_num()

            if 'appearance' in similarity_strategy:
                infra_image, vehicle_image = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_image()
                image_list = (infra_image, vehicle_image)
            else:
                image_list = None

            ##################
            start_time = time.time()

            result = cal_extrinsic_from_two_box_object_list(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, verbose=verbose, image_list=image_list,
                                                            test_cpp_speed=test_cpp_speed, true_matches=filtered_available_matches, similarity_strategy=similarity_strategy, 
                                                            optimization_strategy = optimization_strategy, matches_filter_strategy=matches_filter_strategy)

            T_6DOF_result, matches = [0, 0, 0, 0, 0, 0], []
            if result is not None:
                T_6DOF_result, matches = result[0], result[1]
            
            end_time = time.time()
            ##################

            T_6DOF_true = convert_T_to_6DOF(T_true)
            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF_result, T_6DOF_true)
            
            result_matches = []

            for match in matches:
                if match in filtered_available_matches:
                    result_matches.append(match)

            result_matched_cnt, filtered_available_matches_cnt, total_result_matches_cnt =  len(result_matches), len(filtered_available_matches), len(matches)

            if filtered_available_matches_cnt == 0:
                no_common_box_test = {}
                no_common_box_test['infra_file_name'] = infra_file_name
                no_common_box_test['vehicle_file_name'] = vehicle_file_name
                no_common_box_test['infra_total_box_cnt'] = len(infra_boxes_object_list)
                no_common_box_test['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                no_common_box_test['cost_time'] = end_time - start_time
                no_common_box_list.append(no_common_box_test)
            elif result_matched_cnt > 0:
                valid_test = {}
                valid_test['infra_file_name'] = infra_file_name
                valid_test['vehicle_file_name'] = vehicle_file_name
                valid_test['infra_total_box_cnt'] = len(infra_boxes_object_list)
                valid_test['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)

                valid_test['result_matched_cnt'] = result_matched_cnt
                valid_test['filtered_available_matches_cnt'] = filtered_available_matches_cnt
                valid_test['total_available_matches_cnt'] = total_available_matches_cnt
                valid_test['total_result_matches_cnt'] = total_result_matches_cnt

                valid_test['delta_T_6DOF'] = (T_6DOF_true - T_6DOF_result).tolist()
                valid_test['RE'] = RE.tolist()
                valid_test['TE'] = TE.tolist()
                valid_test['cost_time'] = end_time - start_time

                if RE < 2 and TE < 2:
                    valid_test_list.append(valid_test)
                else:
                    valid_bad_test_list.append(valid_test)

            else:
                invalid_test = {}
                invalid_test['infra_file_name'] = infra_file_name
                invalid_test['vehicle_file_name'] = vehicle_file_name
                invalid_test['infra_total_box_cnt'] = len(infra_boxes_object_list)
                invalid_test['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)

                invalid_test['result_matched_cnt'] = result_matched_cnt
                invalid_test['filtered_available_matches_cnt'] = filtered_available_matches_cnt
                invalid_test['total_available_matches_cnt'] = total_available_matches_cnt
                invalid_test['total_result_matches_cnt'] = total_result_matches_cnt

                invalid_test['delta_T_6DOF'] = (T_6DOF_true - T_6DOF_result).tolist()
                invalid_test['RE'] = RE.tolist()
                invalid_test['TE'] = TE.tolist()
                invalid_test['cost_time'] = end_time - start_time

                invalid_test_list.append(invalid_test)

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
            print(f'result_matched_cnt: {result_matched_cnt}, filtered_available_matches_cnt: {filtered_available_matches_cnt}, total_result_matches_cnt: {total_result_matches_cnt}')
            print(f'delta_T_6DOF: {(T_6DOF_true - T_6DOF_result).tolist()}')
            print(f'RE: {RE.tolist()}')
            print(f'TE: {TE.tolist()}')
            print(f'cost time: {end_time - start_time}')
            print('---------------------------------')

        if cnt % 50 == 0:
            # if len(no_common_box_list):
            with open(os.path.join(output_dict, f'no_common_view_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(no_common_box_list, f)

            # if len(valid_test_list):
            with open(os.path.join(output_dict, f'valid_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(valid_test_list, f)

            # if len(valid_bad_test_list):
            with open(os.path.join(output_dict, f'valid_bad_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(valid_bad_test_list, f)

            # if len(invalid_test_list):
            with open(os.path.join(output_dict, f'invalid_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(invalid_test_list, f)

            # if len(error_list):
            with open(os.path.join(output_dict, f'error_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(error_list, f)

            no_common_box_list = []
            valid_test_list = []
            valid_bad_test_list = []
            invalid_test_list = []
            error_list = []

            print('----------------write to file---------------------')



if __name__ == '__main__':

    # method strategy
    optimization_strategy = 'svd'
    matches_filter_strategy = 'trueT'
    # core category length angle appearance
    similarity_strategy = ['category', 'core', 'appearance']

    # data
    data_difficulty = 'easy'
    result_folder = f'new_clean_result/extrinsic_core_appearance_category_svd_trueT'
    output_dict = f'{result_folder}/{data_difficulty}_dataset'

    if not os.path.exists(output_dict):
        os.makedirs(output_dict)

    batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, using_predict_score=False, test_cpp_speed=False,
                                                    output_dict=output_dict, similarity_strategy=similarity_strategy, optimization_strategy=optimization_strategy,
                                                    matches_filter_strategy=matches_filter_strategy, data_difficulty=data_difficulty)

