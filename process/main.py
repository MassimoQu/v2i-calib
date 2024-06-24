import numpy as np
import os
import time
import json
import sys
sys.path.append('./reader')
sys.path.append('./process')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./process/search')
sys.path.append('./visualize')
from InfraReader import InfraReader
from VehicleReader import VehicleReader
from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from BoxesMatch import BoxesMatch
import BoxesMatch_cpp
from Matches2Extrinsics import Matches2Extrinsics
from extrinsic_utils import implement_T_3dbox_object_list, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, get_reverse_T, convert_Rt_to_T
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
from test_V2XSim import V2XSim_Reader


def convert_py_bbox_to_cpp(bbox):
    bbox_8_3 = np.array(bbox.get_bbox3d_8_3(), dtype=np.float32)
    return BoxesMatch_cpp.BoxObject(bbox_8_3, bbox.get_bbox_type())

def get_matches_with_score_cpp(infra_boxes, vehicle_boxes):
    infra_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in infra_boxes]
    vehicle_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in vehicle_boxes]
    return BoxesMatch_cpp.get_matches_with_score(infra_boxes_cpp, vehicle_boxes_cpp)

# combination & svd
def cal_extrinsic_from_two_box_object_list(box_object_list1, box_object_list2, verbose = False, test_cpp_speed = False, true_matches = None, image_list = None,
                                           similarity_strategy = ['core', 'category'], corresponding_strategy = 'distance', optimization_strategy = 'svd8point', matches_filter_strategy = 'trueT'):

    '''
        optimization 还可以进一步拆成 rough_strategy 和 finetuning_strategy
        
    '''

    if test_cpp_speed:
        matches_with_score_list = get_matches_with_score_cpp(box_object_list1, box_object_list2)
    else:
        boxes_match = BoxesMatch(box_object_list1, box_object_list2, image_list=image_list, similarity_strategy = similarity_strategy, corresponding_strategy=corresponding_strategy)
        matches_with_score_list = boxes_match.get_matches_with_score()
    
    matches = [match[0] for match in matches_with_score_list]
    if matches == []:
        return None
    
    matches2extrinsics = Matches2Extrinsics(box_object_list1, box_object_list2, matches_score_list=matches_with_score_list, true_matches=true_matches)
    return matches2extrinsics.get_combined_extrinsic(matches_filter_strategy=matches_filter_strategy, optimization_strategy=optimization_strategy), matches, matches_with_score_list[0][1] if len(matches_with_score_list) > 0 else 0

def cal_extrinsic_from_true_matches(box_object_list1, box_object_list2, true_matches, matches_filter_strategy = 'trueT'):
    matches_with_score_list = [(match, 1) for match in true_matches]
    matches2extrinsics = Matches2Extrinsics(box_object_list1, box_object_list2, matches_score_list=matches_with_score_list, true_matches=true_matches)
    return matches2extrinsics.get_combined_extrinsic(matches_filter_strategy=matches_filter_strategy, optimization_strategy='svd8point'), true_matches, matches_with_score_list[0][1] if len(matches_with_score_list) > 0 else 0


def batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, test_cpp_speed = False, using_predict_score = False, output_dict = 'intermediate_output',
                                                    similarity_strategy = ['core', 'category'], corresponding_strategy = 'distance',optimization_strategy = 'svd', matches_filter_strategy = 'trueT', 
                                                    data_difficulty = 'hard', true_matches_flag = False, using_V2XSim = False, noise_type = None, noise = None, filtering_dynamic_objects=False, distance_threshold_between_two_frame=1):
    if data_difficulty == 'all':
        path_data_info = '/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json'
    elif data_difficulty in ['easy', 'hard']:
        path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/' + data_difficulty + '_data_info.json'
    elif data_difficulty == 'common_boxes_filtered':
        path_data_info = '/home/massimo/vehicle_infrastructure_calibration/dataset_division/common_boxes_4_data_info.json'
    else:
        raise ValueError('data_difficulty should be easy or hard')
    
    reader = CooperativeBatchingReader(path_data_info = path_data_info)

    cnt = 0

    no_common_box_list = []
    valid_test_list = []
    valid_bad_test_list = []
    invalid_test_list = []

    error_list = []

    if using_V2XSim:
        wrapper = V2XSim_Reader().generate_vehicle_vehicle_bboxes_object_list(noise_type=noise_type, noise=noise)
    else:
        if using_predict_score:
            wrapper = reader.generate_infra_vehicle_bboxes_object_list_predicted()
        elif filtering_dynamic_objects:
            wrapper = reader.generate_infra_vehicle_bboxes_object_list_static_according_last_frame(distance_threshold_between_last_frame=distance_threshold_between_two_frame)
        else:
            wrapper = reader.generate_infra_vehicle_bboxes_object_list()

    
    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in wrapper:
        
        # try:
        if True:

            if verbose:
                print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
                print(f'infra_total_box_cnt: {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}')

            # no_last_frame = False

            # if filtering_dynamic_objects:
            #     last_frame_infra_file_name =  f"{int(infra_file_name) - 1:06}"
            #     last_frame_infra_reader = InfraReader(last_frame_infra_file_name)
            #     if last_frame_infra_reader.exist_infra_label():
            #         last_frame_infra_boxes_object_list = last_frame_infra_reader.get_infra_boxes_object_list()
            #     else:
            #         print(f'no infra last frame: {last_frame_infra_file_name}')
            #         no_last_frame = True
            #         last_frame_infra_boxes_object_list = infra_boxes_object_list
                
            #     last_frame_vehicle_file_name =  f"{int(vehicle_file_name) - 1:06}"
            #     last_frame_vehicle_reader = VehicleReader(last_frame_vehicle_file_name)
            #     if last_frame_vehicle_reader.exist_vehicle_label():
            #         last_frame_vehicle_boxes_object_list = last_frame_vehicle_reader.get_vehicle_boxes_object_list()
            #     else:
            #         print(f'no vehicle last frame: {last_frame_vehicle_file_name}')
            #         no_last_frame = True
            #         last_frame_vehicle_boxes_object_list = vehicle_boxes_object_list

            #     filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_dynamic_object_using_last_frame(last_frame_infra_boxes_object_list, distance_threshold_between_two_frame)
            #     filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_dynamic_object_using_last_frame(last_frame_vehicle_boxes_object_list, distance_threshold_between_two_frame)

            # if no_last_frame:
            #     filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(filter_num)
            #     if using_predict_score:
            #         filtered_infra_boxes_object_list = implement_T_3dbox_object_list(get_reverse_T(T_true), filtered_infra_boxes_object_list)
            #     filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
            
            filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(filter_num)
            if using_predict_score:
                filtered_infra_boxes_object_list = implement_T_3dbox_object_list(get_reverse_T(T_true), filtered_infra_boxes_object_list)
            filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
            
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
            filtered_available_matches = CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches()
            
            converted_original_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
            total_available_matches_cnt = CorrespondingDetector(converted_original_infra_boxes_object_list, vehicle_boxes_object_list).get_matched_num()

            if 'appearance' in similarity_strategy:
                infra_image, vehicle_image = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_image()
                image_list = (infra_image, vehicle_image)
            else:
                image_list = None

            ##################
            start_time = time.time()
            if true_matches_flag:
                result = cal_extrinsic_from_true_matches(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, filtered_available_matches, matches_filter_strategy=matches_filter_strategy)
            else:
                result = cal_extrinsic_from_two_box_object_list(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, verbose=verbose, image_list=image_list,
                                                            test_cpp_speed=test_cpp_speed, true_matches=filtered_available_matches, similarity_strategy=similarity_strategy, 
                                                            corresponding_strategy = corresponding_strategy, optimization_strategy = optimization_strategy, matches_filter_strategy=matches_filter_strategy)

            T_6DOF_result, matches, stability = [0, 0, 0, 0, 0, 0], [], 0
            if result is not None:
                T_6DOF_result, matches, stability = result[0], result[1], int(result[2])
            

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

                valid_test['stability'] = stability

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

                invalid_test['stability'] = stability

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

    # using_V2XSim = True
    # noise_type = 'gaussian'

    filtering_dynamic_objects = False
    distance_threshold_between_two_frame = 2

    using_V2XSim = False
    noise_type = None

    noise = {'pos_std':0, 'rot_std':25, 'pos_mean':0, 'rot_mean':0}

    # method strategy
    optimization_strategy = 'svd8point'
    matches_filter_strategy = 'top matches'
    true_matches_flag = False
    # iou centerpoint_distance vertex_distance
    corresponding_strategy_list = ['centerpoint_distance','vertex_distance']
    if true_matches_flag == False:
        # core category length angle appearance
        similarity_strategy = ['category', 'core']
    else:
        similarity_strategy = []
    
    data_difficulty = 'common_boxes_filtered'

    sim_str = ''

    if using_V2XSim:
        sim_str += 'V2XSim_'
    
    if noise_type != None:
        sim_str = 'noise_' + noise_type + '_'

    if true_matches_flag:
        sim_str += 'true_matches_'
    else:
        for i in similarity_strategy:
            sim_str += i + '_'

    for i in corresponding_strategy_list:
        sim_str += i + '_'

    # data
    
    if filtering_dynamic_objects:
        sim_str += 'filtering_dynamic_objects_rescoring'

    result_folder = f'new_clean_result/extrinsic_' + sim_str + optimization_strategy + '_' + matches_filter_strategy
    output_dict = f'{result_folder}/{data_difficulty}_dataset'

    if filtering_dynamic_objects:
        output_dict += f'/distance_threshold_{distance_threshold_between_two_frame}'

    if noise_type != None and noise != None:
        output_dict += '/'
        for key, value in noise.items():
            output_dict += f'_{key}_{value}'
                
    if not os.path.exists(output_dict):
        os.makedirs(output_dict)

    batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, using_predict_score=False, test_cpp_speed=False,
                                                    output_dict=output_dict, similarity_strategy=similarity_strategy, optimization_strategy=optimization_strategy, corresponding_strategy=corresponding_strategy_list,
                                                    matches_filter_strategy=matches_filter_strategy, data_difficulty=data_difficulty, true_matches_flag=true_matches_flag, using_V2XSim=using_V2XSim, noise_type=noise_type, noise=noise, filtering_dynamic_objects=filtering_dynamic_objects, distance_threshold_between_two_frame=distance_threshold_between_two_frame)

