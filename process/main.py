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
from CooperativeBatchingReader import CooperativeBatchingReader
from BoxesMatch import BoxesMatch
from PSO_Executor import PSO_Executor
from extrinsic_utils import implement_T_3dbox_object_list, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes


def cal_extrinsic_from_two_box_object_list(box_object_list1, box_object_list2, filter_num = 15, verbose = False):
    boxes_match = BoxesMatch(box_object_list1, box_object_list2)
    matches = boxes_match.get_matches()
    if matches == []:
        return None
    pso = PSO_Executor(box_object_list1, box_object_list2, matches=matches, filter_num=filter_num, verbose=verbose)
    return pso.get_best_T_6DOF(), matches


def test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15):
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    valid_test_list = []
    valid_bad_test_list = []
    invalid_test_list = []
    no_common_box_list = []

    error_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in reader.generate_infra_vehicle_bboxes_object_list():
        
        # try:
        if True:

            if verbose:
                print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
                print(f'infra_total_box_cnt: {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}')


            ##################
            start_time = time.time()

            filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(filter_num)
            filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
            T_6DOF_result, matches = cal_extrinsic_from_two_box_object_list(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, filter_num=filter_num, verbose=verbose)
            
            end_time = time.time()
            ##################

            T_6DOF_true = convert_T_to_6DOF(T_true)
            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF_result, T_6DOF_true)

            matched_infra_bboxes_object_list = []
            matched_vehicle_bboxes_object_list = []

            for match in matches:
                matched_infra_bboxes_object_list.append(filtered_infra_boxes_object_list[match[0]])
                matched_vehicle_bboxes_object_list.append(filtered_vehicle_boxes_object_list[match[1]])

            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
            
            available_matches = CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list).corresponding_IoU_dict.keys()
            
            result_matches = []

            for match in matches:
                if match in available_matches:
                    result_matches.append(match)

            matched_cnt, available_matches_cnt, total_matches_cnt =  len(result_matches), len(available_matches), len(matched_infra_bboxes_object_list)

            if available_matches_cnt == 0:
                no_common_box_test = {}
                no_common_box_test['infra_file_name'] = infra_file_name
                no_common_box_test['vehicle_file_name'] = vehicle_file_name
                no_common_box_test['infra_total_box_cnt'] = len(infra_boxes_object_list)
                no_common_box_test['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                no_common_box_test['cost_time'] = end_time - start_time
                no_common_box_list.append(no_common_box_test)
            elif matched_cnt > 0:
                valid_test = {}
                valid_test['infra_file_name'] = infra_file_name
                valid_test['vehicle_file_name'] = vehicle_file_name
                valid_test['infra_total_box_cnt'] = len(infra_boxes_object_list)
                valid_test['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                valid_test['matched_cnt'] = matched_cnt
                valid_test['available_matches_cnt'] = available_matches_cnt
                valid_test['total_matches_cnt'] = total_matches_cnt
                valid_test['delta_T_6DOF'] = (T_6DOF_true - T_6DOF_result).tolist()
                valid_test['RE'] = RE.tolist()
                valid_test['TE'] = TE.tolist()
                valid_test['cost_time'] = end_time - start_time

                if RE < 3 and TE < 3:
                    valid_test_list.append(valid_test)
                else:
                    valid_bad_test_list.append(valid_test)

            else:
                invalid_test = {}
                invalid_test['infra_file_name'] = infra_file_name
                invalid_test['vehicle_file_name'] = vehicle_file_name
                invalid_test['infra_total_box_cnt'] = len(infra_boxes_object_list)
                invalid_test['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                invalid_test['matched_cnt'] = matched_cnt
                invalid_test['available_matches_cnt'] = available_matches_cnt
                invalid_test['total_matches_cnt'] = total_matches_cnt
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
            print(f'matched_cnt: {matched_cnt}, available_matches_cnt: {available_matches_cnt}, total_matches_cnt: {total_matches_cnt}')
            print(f'delta_T_6DOF: {(T_6DOF_true - T_6DOF_result).tolist()}')
            print(f'RE: {RE.tolist()}')
            print(f'TE: {TE.tolist()}')
            print(f'cost time: {end_time - start_time}')
            print('---------------------------------')

        if cnt % 100 == 0:
            if len(no_common_box_list):
                with open(f'intermediate_output/no_common_view_k{filter_num}_cnt{cnt}.json', 'w') as f:
                    json.dump(no_common_box_list, f)

            if len(valid_test_list):
                with open(f'intermediate_output/valid_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
                    json.dump(valid_test_list, f)

            if len(valid_bad_test_list):
                with open(f'intermediate_output/valid_bad_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
                    json.dump(valid_bad_test_list, f)

            if len(invalid_test_list):
                with open(f'intermediate_output/invalid_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
                    json.dump(invalid_test_list, f)

            if len(error_list):
                with open(f'intermediate_output/error_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
                    json.dump(error_list, f)

            no_common_box_list = []
            valid_test_list = []
            valid_bad_test_list = []
            invalid_test_list = []
            error_list = []

            print('----------------write to file---------------------')
    
    if len(no_common_box_list):
        with open(f'intermediate_output/no_common_view_k{filter_num}_cnt{cnt}.json', 'w') as f:
            json.dump(no_common_box_list, f)

    if len(valid_test_list):
        with open(f'intermediate_output/successful_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
            json.dump(valid_test_list, f)

    if len(invalid_test_list):
        with open(f'intermediate_output/invalid_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
            json.dump(invalid_test_list, f)

    if len(error_list):
        with open(f'intermediate_output/error_extrinsic_k{filter_num}_cnt{cnt}.json', 'w') as f:
            json.dump(error_list, f)


if __name__ == '__main__':
    test_extrisic_from_two_box_object_list(verbose = False, filter_num = 10)