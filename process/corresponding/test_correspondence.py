import numpy as np
import os
import time
import json
import signal
import sys
sys.path.append('./reader')
sys.path.append('./process')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./process/search')
sys.path.append('./visualize')
from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from BoxesMatch import BoxesMatch
from extrinsic_utils import implement_T_3dbox_object_list, implement_T_points_n_3, get_reverse_T, get_extrinsic_from_two_3dbox_object, convert_6DOF_to_T, get_extrinsic_from_two_3dbox_object, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
from BBoxVisualizer_open3d_standardized import BBoxVisualizer_open3d_standardized
from Matches2Extrinsics import Matches2Extrinsics


# combination & svd
def get_matches_with_score_list(box_object_list1, box_object_list2, verbose = False, true_matches = None, image_list = None,
                                           similarity_strategy = ['core', 'category']):

    '''
        optimization 还可以进一步拆成 rough_strategy 和 finetuning_strategy
        
    '''

    boxes_match = BoxesMatch(box_object_list1, box_object_list2, image_list=image_list, similarity_strategy = similarity_strategy)
    matches_with_score_list = boxes_match.get_matches_with_score()
    return matches_with_score_list


def save_intermediate_match_output(verbose = False, filter_num = 15, test_cpp_speed = False, using_predict_score = False, output_dict = 'intermediate_output',
                                    similarity_strategy = ['core', 'category'], data_difficulty = 'hard'):
    if data_difficulty not in ['easy', 'hard']:
        raise ValueError('data_difficulty should be easy or hard')
    reader = CooperativeBatchingReader(path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/' + data_difficulty + '_data_info.json')

    cnt = 0

    no_common_box_list = []
    valid_test_list = []
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

            matches_with_score_list = get_matches_with_score_list(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, verbose = verbose, true_matches = filtered_available_matches, image_list = image_list,
                                           similarity_strategy = similarity_strategy)
            
            serializable_matches_with_score_list = [[[int(item[0][0]), int(item[0][1])], float(item[1])] for item in matches_with_score_list]

            end_time = time.time()
            ##################

            matches = [match[0] for match in matches_with_score_list]

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

                valid_test['matches_with_score_list'] = serializable_matches_with_score_list

                valid_test['cost_time'] = end_time - start_time

                valid_test_list.append(valid_test)

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

                invalid_test['matches_with_score_list'] = serializable_matches_with_score_list

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
            print(f'cost time: {end_time - start_time}')
            print(f'matches_with_score_list: {matches_with_score_list}')
            print('---------------------------------')

        if cnt % 50 == 0:
            # if len(no_common_box_list):
            with open(os.path.join(output_dict, f'no_common_view_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(no_common_box_list, f)

            # if len(valid_test_list):
            with open(os.path.join(output_dict, f'valid_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(valid_test_list, f)

            # if len(invalid_test_list):
            with open(os.path.join(output_dict, f'invalid_k{filter_num}_cnt{cnt}.json'), 'w') as f:
                json.dump(invalid_test_list, f)

            # if len(error_list):
            # with open(os.path.join(output_dict, f'error_extrinsic_k{filter_num}_cnt{cnt}.json'), 'w') as f:
            #     json.dump(error_list, f)

            no_common_box_list = []
            valid_test_list = []
            valid_bad_test_list = []
            invalid_test_list = []
            error_list = []

            print('----------------write to file---------------------')

def execute_save_intermediate_match_output():
    # method strategy
    similarity_strategy = ['category', 'core']
    
    data_difficulty = 'hard'

    sim_str = ''

    for i in similarity_strategy:
        sim_str += i + '_'

    # ioucorresponding & distancecorresponding
    corresponding_strategy = 'distancecorresponding'

    result_folder = f'new_clean_result/intermediate_box_matches_' + sim_str + corresponding_strategy
    output_dict = f'{result_folder}/{data_difficulty}_dataset'

    if not os.path.exists(output_dict):
        os.makedirs(output_dict)

    save_intermediate_match_output(verbose = False, filter_num = 15, using_predict_score=False, test_cpp_speed=False,
                                    output_dict=output_dict, similarity_strategy=similarity_strategy, data_difficulty=data_difficulty)


def signal_handler(sig, frame):
    print("Exiting visualization...")
    global should_exit
    should_exit = True

signal.signal(signal.SIGINT, signal_handler)
global should_exit
should_exit = False

def read_intermediate_match_and_visualize(difficulty='easy', total_cnt=1300, verbose=False, filter_num=15):
    filename = f'new_clean_result/intermediate_box_matches_category_core_ioucorresponding/{difficulty}_dataset/valid_k15_cnt'
    
    error_high_score_group = []
    right_low_score_group = []

    

    for cnt in range(50, total_cnt + 1, 50):
        if should_exit:
            break
        concated_filename = filename + str(cnt) + '.json'
        if not os.path.exists(concated_filename):
            continue
        
        with open(concated_filename, 'r') as f:
            example_list = json.load(f)

        for example in example_list:
            if should_exit:
                break

            if verbose:
                print(example['infra_file_name'], example['vehicle_file_name'])
                # print(example['result_matched_cnt'], example['filtered_available_matches_cnt'], example['total_available_matches_cnt'], example['total_result_matches_cnt'])
                # print(example['cost_time'])
                # print(example['matches_with_score_list'])
                print('---------------------------------')

            matches_with_score_dict = {tuple(item[0]): item[1] for item in example['matches_with_score_list']}

            infra_boxes_list, vehicle_boxes_list = CooperativeReader(example['infra_file_name'], example['vehicle_file_name']).get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
            infra_pointcloud, vehicle_pointcloud = CooperativeReader(example['infra_file_name'], example['vehicle_file_name']).get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()
            filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_list).filter_according_to_size_topK(filter_num)
            filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_list).filter_according_to_size_topK(filter_num)
            filtered_true_matches_with_score = CorrespondingDetector(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, corresponding_strategy='centerpoint_distance').get_matches_with_score()

            analyze_and_visualize_intermediate_match([filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [infra_pointcloud, vehicle_pointcloud], matches_with_score_dict, filtered_true_matches_with_score)


def get_BoxesMatch_intermediate_match_and_visualize(difficulty='easy', verbose=True, filter_num=15):
    if difficulty == 'all':
        path_data_info = '/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json'
    elif difficulty in ['easy', 'hard']:
        path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/' + difficulty + '_data_info.json'
    else:
        raise ValueError('data_difficulty should be easy or hard')
    
    reader = CooperativeBatchingReader(path_data_info = path_data_info)

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, infra_pointcloud, vehicle_pointcloud, T_true in reader.generate_infra_vehicle_bboxes_object_list_pointcloud():
        if should_exit:
            break

        infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        infra_pointcloud = implement_T_points_n_3(T_true, infra_pointcloud)
        
        filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(filter_num)
        filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
        filtered_true_matches_with_score = CorrespondingDetector(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, corresponding_strategy='centerpoint_distance').get_matches_with_score()

        if verbose:
            print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
            print(f'infra_total_box_cnt: {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}')
            print(f'filtered_infra_boxes_len: {len(filtered_infra_boxes_object_list)}, filtered_vehicle_boxes_len: {len(filtered_vehicle_boxes_object_list)}, filtered_true_matches_with_score: {len(filtered_true_matches_with_score)}')

        matches_with_score_list = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches_with_score()
        matches_with_score_dict = {tuple(item[0]): item[1] for item in matches_with_score_list}

        analyze_and_visualize_intermediate_match([filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [infra_pointcloud, vehicle_pointcloud], matches_with_score_dict, filtered_true_matches_with_score)
        
def visualize_specific_intermediate_match(infra_file_name, vehicle_file_name, infra_number=-1, vehicle_number=-1, filter_num=15):

    infra_boxes_list, vehicle_boxes_list = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_boxes_object_list()
    infra_pointcloud, vehicle_pointcloud = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_pointcloud()

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_list).filter_according_to_size_topK(filter_num)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_list).filter_according_to_size_topK(filter_num)

    T_true = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_T_i2v()
    converted_true_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
    converted_true_infra_pointcloud = implement_T_points_n_3(T_true, infra_pointcloud)
    filtered_true_matches_with_score = CorrespondingDetector(converted_true_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches_with_score()

    if infra_number != -1 and vehicle_number != -1:
        T_specific = get_extrinsic_from_two_3dbox_object(filtered_infra_boxes_object_list[infra_number], filtered_vehicle_boxes_object_list[vehicle_number])
        converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_specific, filtered_infra_boxes_object_list)
        converted_infra_pointcloud = implement_T_points_n_3(T_specific, infra_pointcloud)
        matches_with_score_dict = CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches_with_score()
        print('specific T')
        analyze_and_visualize_intermediate_match([converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [converted_infra_pointcloud, vehicle_pointcloud], matches_with_score_dict, filtered_true_matches_with_score)
    # else:
    matches_with_score_list = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches_with_score()
    # matches_with_score_dict = {tuple(item[0]): item[1] for item in matches_with_score_list}
    matches_with_score_dict = CorrespondingDetector(converted_true_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches_with_score()
    # infra_pointcloud, vehicle_pointcloud = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()
    print('true T')
    analyze_and_visualize_intermediate_match([converted_true_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [converted_true_infra_pointcloud, vehicle_pointcloud], matches_with_score_dict, filtered_true_matches_with_score)

    T_calucated = convert_6DOF_to_T(Matches2Extrinsics(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, matches_score_list=matches_with_score_list).get_combined_extrinsic(matches_filter_strategy='threshold', optimization_strategy='svd8point'))
    T_calucated_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_calucated, filtered_infra_boxes_object_list)
    T_calucated_converted_infra_pointcloud = implement_T_points_n_3(T_calucated, infra_pointcloud)
    T_calucated_matches_with_score_dict = CorrespondingDetector(T_calucated_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches_with_score()
    print('calculated T')
    analyze_and_visualize_intermediate_match([T_calucated_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_calucated_converted_infra_pointcloud, vehicle_pointcloud], T_calucated_matches_with_score_dict, filtered_true_matches_with_score)

    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T_calucated), convert_T_to_6DOF(T_true))
    print(f'RE: {RE}, TE: {TE}')


def analyze_and_visualize_intermediate_match(boxes_object_lists, pointclouds, matches_with_score_dict, filtered_true_matches_with_score={}):

    matches = matches_with_score_dict.keys()

    error_match = []
    high_score_error_match_score = {}
    low_score_right_match_score = {}

    high_score_right_match_score = {}
    low_score_error_match_score = {}

    for match in matches:
        if match not in filtered_true_matches_with_score.keys():
            error_match.append(match)
            if matches_with_score_dict[match] >= 5:
                high_score_error_match_score[match] = matches_with_score_dict[match]
            else:
                low_score_error_match_score[match] = matches_with_score_dict[match]
        else:
            if matches_with_score_dict[match] < 5:
                low_score_right_match_score[match] = matches_with_score_dict[match]
            else:
                high_score_right_match_score[match] = matches_with_score_dict[match]

    # print('len(true_matches): ', len(filtered_true_matches_with_score))
    # print(filtered_true_matches_with_score)
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene(boxes_object_lists, pointclouds, filtered_true_matches_with_score)

    print('len(matches): ', len(matches))
    print('matches_with_score_dict: ', matches_with_score_dict)
    BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene(boxes_object_lists, pointclouds, matches_with_score_dict)

    # if len(high_score_error_match_score):
    #     print('len(high_score_error_match_score): ', len(high_score_error_match_score))
    #     BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene(boxes_object_lists, pointclouds, high_score_error_match_score, k=15)
    # if len(low_score_right_match_score) > 0:
    #     print('len(low_score_right_match_score): ', len(low_score_right_match_score))
    #     BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene(boxes_object_lists, pointclouds, low_score_right_match_score, k=15)


    # if len(high_score_right_match_score) > 0:
    #     print('len(high_score_right_match_score): ', len(high_score_right_match_score))
    #     BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene(boxes_object_lists, pointclouds, high_score_right_match_score , k=15)
    # if len(low_score_error_match_score) > 0:
    #     print('len(low_score_error_match_score): ', len(low_score_error_match_score))
    #     BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene(boxes_object_lists, pointclouds, low_score_error_match_score, k=15)    


def test_batching_predicted_intermediate_match():
    reader = CooperativeBatchingReader()

    cnt = 0

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, infra_boxes_object_list_predicted, vehicle_boxes_object_list_predicted, infra_pointlcoud, vehicle_pointcloud, T_true in reader.generate_infra_vehicle_bboxes_object_list_predicted_and_true_label_pointcloud():
        
        if should_exit:
            break

        true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        true_converted_infra_pointlcoud = implement_T_points_n_3(T_true, infra_pointlcoud)

        matches_with_score_true = CorrespondingDetector(true_converted_infra_boxes_object_list, vehicle_boxes_object_list,corresponding_strategy='vertex_distance').get_matches_with_score()
        matches_with_score_predicted = BoxesMatch(infra_boxes_object_list_predicted, vehicle_boxes_object_list_predicted).get_matches_with_score()
        matches_with_score_predicted_dict = {tuple(item[0]): item[1] for item in matches_with_score_predicted}
        T_calculated_6DOF = Matches2Extrinsics(infra_boxes_object_list_predicted, vehicle_boxes_object_list_predicted, matches_score_list=matches_with_score_predicted)\
            .get_combined_extrinsic(matches_filter_strategy='threshold', optimization_strategy='svd8point')
        T_calculated = convert_6DOF_to_T(T_calculated_6DOF)
        true_back_converted_infra_boxes_object_list = implement_T_3dbox_object_list(get_reverse_T(T_true), infra_boxes_object_list_predicted)
        T_calculated_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_calculated, true_back_converted_infra_boxes_object_list)
        T_calculated_converted_infra_pointcloud = implement_T_points_n_3(T_calculated, infra_pointlcoud)

        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated_6DOF, convert_T_to_6DOF(T_true))

        print(cnt)
        print(f'infra_file_name: {infra_file_name} ; vehicle_file_name: {vehicle_file_name}')
        print(f'len(matches_with_score_true): {len(matches_with_score_true)} ; len(matches_with_score_predicted): {len(matches_with_score_predicted)}')
        print('matches_with_score_true: ', matches_with_score_true)
        print('matches_with_score_predicted: ', matches_with_score_predicted)
        print(f'RE: {RE}, TE: {TE}')
        print('---------------------------------')
        cnt += 1
        
        # BBoxVisualizer_open3d_standardized(vis_names=['true', 'predicted'], vis_num=2).visualize_matches_under_dual_true_predicted_scene([true_converted_infra_boxes_object_list, vehicle_boxes_object_list], [T_calculated_converted_infra_boxes_object_list, vehicle_boxes_object_list_predicted], [true_converted_infra_pointlcoud, vehicle_pointcloud], [T_calculated_converted_infra_pointcloud, vehicle_pointcloud], matches_with_score_true, matches_with_score_predicted_dict)

        BBoxVisualizer_open3d_standardized(vis_names=['true', 'predicted'], vis_num=2)\
        .visualize_matches_under_dual_true_predicted_scene(
            [true_converted_infra_boxes_object_list, vehicle_boxes_object_list], 
            [T_calculated_converted_infra_boxes_object_list, vehicle_boxes_object_list_predicted], 
            [true_converted_infra_pointlcoud, vehicle_pointcloud], 
            [T_calculated_converted_infra_pointcloud, vehicle_pointcloud], 
            {}, {})

        # BBoxVisualizer_open3d_standardized(vis_names=['true', 'predicted'], vis_num=2)\
        # .visualize_matches_under_dual_true_predicted_scene(
        #     [infra_boxes_object_list, vehicle_boxes_object_list], 
        #     [true_back_converted_infra_boxes_object_list, vehicle_boxes_object_list_predicted], 
        #     [infra_pointlcoud, vehicle_pointcloud], 
        #     [infra_pointlcoud, vehicle_pointcloud], 
        #     {}, {})


def test_batching_cooperative_fusioned_vehicle_intermediate_match():
    reader = CooperativeBatchingReader()

    cnt = 0

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, _, vehicle_boxes_object_list_fusioned, infra_pointlcoud, vehicle_pointcloud, T_true in reader.generate_infra_vehicle_bboxes_object_list_cooperative_fusioned_and_ego_true_label_pointcloud():
        
        if should_exit:
            break

        true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        true_converted_infra_pointlcoud = implement_T_points_n_3(T_true, infra_pointlcoud)

        matches_with_score_true = CorrespondingDetector(true_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()
        matches_with_score_vehicle_fusioned = CorrespondingDetector(infra_boxes_object_list, vehicle_boxes_object_list_fusioned).get_matches_with_score()

        print(cnt)
        print(f'infra_file_name: {infra_file_name} ; vehicle_file_name: {vehicle_file_name}')
        print('matches_with_score_true: ', matches_with_score_true)
        print('matches_with_score_vehicle_fusioned: ', matches_with_score_vehicle_fusioned)
        print('---------------------------------')
        cnt += 1
        
        # BBoxVisualizer_open3d_standardized(vis_names=['true', 'predicted'], vis_num=2).visualize_matches_under_dual_true_predicted_scene([true_converted_infra_boxes_object_list, vehicle_boxes_object_list], [T_calculated_converted_infra_boxes_object_list, vehicle_boxes_object_list_predicted], [true_converted_infra_pointlcoud, vehicle_pointcloud], [T_calculated_converted_infra_pointcloud, vehicle_pointcloud], matches_with_score_true, matches_with_score_predicted_dict)

        BBoxVisualizer_open3d_standardized(vis_names=['ego_perception', 'cooperative_perception_enhanced'], vis_num=2)\
        .visualize_matches_under_dual_true_predicted_scene(
            [true_converted_infra_boxes_object_list, vehicle_boxes_object_list], 
            [true_converted_infra_boxes_object_list, vehicle_boxes_object_list_fusioned], 
            [true_converted_infra_pointlcoud, vehicle_pointcloud], 
            [true_converted_infra_pointlcoud, vehicle_pointcloud], 
            matches_with_score_true, matches_with_score_vehicle_fusioned)


if __name__ == '__main__':
    # read_intermediate_match_and_visualize(verbose=True)
    # execute_save_intermediate_match_output()
    # get_BoxesMatch_intermediate_match_and_visualize()
    # visualize_specific_intermediate_match('016844', '010839')

    # test_batching_predicted_intermediate_match()
    # test_batching_cooperative_fusioned_vehicle_intermediate_match()


    visualize_specific_intermediate_match('004557', '019882') 