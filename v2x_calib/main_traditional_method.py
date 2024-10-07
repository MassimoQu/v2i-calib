import os
import time
import json
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from reader import CooperativeBatchingReader
from reader import V2XSim_Reader
from utils import get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, convert_Rt_to_T

import reglib
import pygicp
from probreg import filterreg

def batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, output_dict = 'intermediate_output', method = 'gicp', data_difficulty = 'hard', data = 'V2XSim'):
    
    if data == 'V2XSim':
        wrapper = V2XSim_Reader().generate_vehicle_vehicle_bboxes_object_list_pointcloud(noise={'pos_std': 0, 'rot_std': 0, 'pos_mean': 0, 'rot_mean': 0})

    elif data == 'DAIR-V2X':
        if data_difficulty not in ['easy', 'hard']:
            raise ValueError('data_difficulty should be easy or hard')
        else:
            reader = CooperativeBatchingReader(path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/' + data_difficulty + '_data_info.json')
            wrapper = reader.generate_infra_vehicle_pointcloud()
    
    cnt = 0 

    valid_test_list = []
    valid_bad_test_list = []

    error_list = []

    for infra_file_name, vehicle_file_name, _, _, infra_pointcloud, vehicle_pointcloud, T_true in wrapper:
        
        # try:
        if True:

            if verbose:
                print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
                
            ##################
            start_time = time.time()

            if method == 'gicp':
                T_6DOF_result = convert_T_to_6DOF(pygicp.align_points(infra_pointcloud, vehicle_pointcloud, method='GICP'))
            elif method == 'ndt':
                # print('type of infra_pointcloud:', type(infra_pointcloud))
                T_6DOF_result = convert_T_to_6DOF(reglib.ndt(source=infra_pointcloud.astype(np.float64), target=vehicle_pointcloud.astype(np.float64)))
            elif method == 'filterreg':
                result = filterreg.registration_filterreg(infra_pointcloud, vehicle_pointcloud)
                T_6DOF_result = convert_T_to_6DOF(convert_Rt_to_T(result.transformation.rot, result.transformation.t))
            else:
                raise ValueError('method should be gicp, ndt or filterreg')
            
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

    # data
    data_difficulty = 'hard'
    result_folder = f'new_clean_result'
    # filterreg ndt gicp
    method = 'gicp'
    # output_dict = f'{result_folder}/{data_difficulty}_dataset'
    output_dict = f'{result_folder}/V2XSim_dataset_{method}'

    if not os.path.exists(output_dict):
        os.makedirs(output_dict)

    batching_test_extrisic_from_two_box_object_list(verbose = False, filter_num = 15, output_dict = output_dict, method = method, data_difficulty = data_difficulty, data = 'V2XSim')
