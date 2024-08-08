import math
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import pandas as pd
import seaborn as sns
import json
import sys
# sys.path.append('./reader')
# sys.path.append('./process')
# sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
# sys.path.append('./process/search')
# sys.path.append('./visualize')
# from CooperativeReader import CooperativeReader
# from BoxesMatch import BoxesMatch
# from Filter3dBoxes import Filter3dBoxes
# from PSO_Executor import PSO_Executor
# from extrinsic_utils import convert_6DOF_to_T, convert_T_to_Rt
# from BBoxVisualizer_open3d import BBoxVisualizer_open3d


def devide_group_according_scene_difficulty(total_cnt = 650, k = 15, folder_name = r'intermediate_output/', output_folder = r'intermediate_output/111/'):

    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    success_cnt = 0
    hard_success_cnt = 0
    valid_bad_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    easy_group = [] # RE < 1, TE < 1 -> valid
    medium_group = [] # 1 <= RE <= 5, 1 <= TE <= 5 -> valid + valid_bad
    hard_group = [] # 5 < RE < 10 or 5 < TE < 10 -> valid_bad
    extreme_group = [] # RE > 10 or TE > 10 -> invalid
    invalid_group = [] # invalid
    no_common_group = [] # no_common

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            if os.path.exists(folder_name + file_name + str(cnt) + '.json') == False:
                continue

            with open(folder_name + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
            
            if file_name == file_name_list[0] or file_name == file_name_list[1]:
                
                # for example in example_list:
                #     if example['RE'] < 1 and example['TE'] < 1:
                #         easy_group.append(example)
                #     elif example['RE'] <= 5 and example['TE'] <= 5:
                #         medium_group.append(example)
                #     elif example['RE'] <= 10 and example['TE'] <= 10:
                #         hard_group.append(example)
                #     else:
                #         extreme_group.append(example)

                for example in example_list:
                    if (example['RE'] < 1 and example['TE'] < 1.1) or (example['RE'] < 1.1 and example['TE'] < 1):
                        if example['TE'] < 1:
                            success_cnt += 1
                        easy_group.append(example)
                    elif example['RE'] <= 5 and example['TE'] <= 5:
                        if example['TE'] < 2:
                            hard_success_cnt += 1
                        medium_group.append(example)
                    elif example['RE'] <= 10 and example['TE'] <= 10:
                        hard_group.append(example)
                    else:
                        extreme_group.append(example)

                if file_name == file_name_list[0]:
                    valid_cnt += len(example_list)
                                
                elif file_name == file_name_list[1]:
                    valid_bad_cnt += len(example_list)

            if file_name == file_name_list[2] :
                invalid_cnt += len(example_list)
                invalid_group += example_list

            elif file_name == file_name_list[3]:
                no_common_cnt += len(example_list)
                no_common_group += example_list

    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    print(f'easy_group: {len(easy_group)}, medium_group: {len(medium_group)}, hard_group: {len(hard_group)}, extreme_group: {len(extreme_group)}, invalid_group: {len(invalid_group)}, no_common_group: {len(no_common_group)}')
    if len(easy_group):
        print(f'success_cnt: {success_cnt}, success_rate: {success_cnt / len(easy_group)}')
    if len(medium_group):
        print(f'hard_success_cnt: {hard_success_cnt}, hard_success_rate: {hard_success_cnt / len(medium_group)}')
    # print(f'easy_group / total_cnt: {len(easy_group) / total_cnt}, medium_group / total_cnt: {len(medium_group) / total_cnt}, hard_group / total_cnt: {len(hard_group) / total_cnt}, extreme_group / total_cnt: {len(extreme_group) / total_cnt}, invalid_group / total_cnt: {len(invalid_group) / total_cnt}, no_common_group / total_cnt: {len(no_common_group) / total_cnt}')
    # print(f'easy_group / (valid + valid_bad): {(len(easy_group) / (valid_cnt + valid_bad_cnt))}, medium_group / (valid + valid_bad): {(len(medium_group) / (valid_cnt + valid_bad_cnt))}, hard_group / (valid + valid_bad): {(len(hard_group) / (valid_cnt + valid_bad_cnt))}')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if len(easy_group):
        with open(os.path.join(output_folder, f'easy_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
            json.dump(easy_group, f)

    if len(medium_group):
        with open(os.path.join(output_folder, f'medium_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
            json.dump(medium_group, f)

    if len(hard_group):
        with open(os.path.join(output_folder, f'hard_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
            json.dump(hard_group, f)

    if len(extreme_group):
        with open(os.path.join(output_folder, f'extreme_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
            json.dump(extreme_group, f)

    if len(invalid_group):
        with open(os.path.join(output_folder, f'invalid_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
            json.dump(invalid_group, f)

    if len(no_common_group):
        with open(os.path.join(output_folder, f'no_common_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
            json.dump(no_common_group, f)

    print('finished')


def count_test_result_according_scene_difficulty(total_num = 650, k = 15, visualize = False, folder_name = r'intermediate_output/111/'):
    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    file_name_list = [f'easy_group_k{k}_totalcnt{total_num}', f'medium_group_k{k}_totalcnt{total_num}', f'hard_group_k{k}_totalcnt{total_num}', f'extreme_group_k{k}_totalcnt{total_num}', f'invalid_group_k{k}_totalcnt{total_num}', f'no_common_group_k{k}_totalcnt{total_num}' ]
    
    easy_cnt = 0
    medium_cnt = 0
    hard_cnt = 0
    extreme_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    RE_list = {}
    TE_list = {}
    time_cost_list = {}
    x_list = {}
    y_list = {}
    z_list = {}
    roll_list = {}
    pitch_list = {}
    yaw_list = {}

    for file_name in file_name_list:
        if os.path.exists(folder_name + file_name + '.json') == False:
            print(f'{folder_name + file_name + ".json"} not exist')
            continue

        with open(folder_name + file_name + '.json', 'r') as f:
            example_list = json.load(f)
            
        if file_name == file_name_list[-1]:
            no_common_cnt += len(example_list)
        else:
            try:
                RE_list_part = [example['RE'] for example in example_list]
            except Exception as e:
                print(file_name)
                print(example)
                print(e)
                continue

            TE_list_part = [example['TE'] for example in example_list]
            time_cost_list_part = [example['cost_time'] for example in example_list]

            x_list_part = [np.abs(example['delta_T_6DOF'][0]) for example in example_list]
            y_list_part = [np.abs(example['delta_T_6DOF'][1]) for example in example_list]
            z_list_part = [np.abs(example['delta_T_6DOF'][2]) for example in example_list]
            roll_list_part = []
            pitch_list_part = []
            yaw_list_part = []
            for example in example_list:
                i = 0
                for alpha in example['delta_T_6DOF'][3:]:
                    if alpha > 180:
                        alpha = alpha - 360                         
                    elif alpha < -180: 
                        alpha = alpha + 360
                    if i == 0:
                        roll_list_part.append(np.abs(alpha))
                    elif i == 1:
                        pitch_list_part.append(np.abs(alpha))
                    elif i == 2:
                        yaw_list_part.append(np.abs(alpha))
                    i += 1

            if file_name == file_name_list[0]:
                easy_cnt += len(example_list)
                RE_list['easy'] = RE_list_part
                TE_list['easy'] = TE_list_part
                time_cost_list['easy'] = time_cost_list_part
                x_list['easy'] = x_list_part
                y_list['easy'] = y_list_part
                z_list['easy'] = z_list_part
                roll_list['easy'] = roll_list_part
                pitch_list['easy'] = pitch_list_part
                yaw_list['easy'] = yaw_list_part
                
            elif file_name == file_name_list[1]:
                medium_cnt += len(example_list)
                RE_list['medium'] = RE_list_part
                TE_list['medium'] = TE_list_part
                time_cost_list['medium'] = time_cost_list_part
                x_list['medium'] = x_list_part
                y_list['medium'] = y_list_part
                z_list['medium'] = z_list_part
                roll_list['medium'] = roll_list_part
                pitch_list['medium'] = pitch_list_part
                yaw_list['medium'] = yaw_list_part

            elif file_name == file_name_list[2]:
                hard_cnt += len(example_list)
                RE_list['hard'] = RE_list_part
                TE_list['hard'] = TE_list_part
                time_cost_list['hard'] = time_cost_list_part
                x_list['hard'] = x_list_part
                y_list['hard'] = y_list_part
                z_list['hard'] = z_list_part
                roll_list['hard'] = roll_list_part
                pitch_list['hard'] = pitch_list_part
                yaw_list['hard'] = yaw_list_part

            elif file_name == file_name_list[3]:
                extreme_cnt += len(example_list)
                RE_list['extreme'] = RE_list_part
                TE_list['extreme'] = TE_list_part
                time_cost_list['extreme'] = time_cost_list_part
                x_list['extreme'] = x_list_part
                y_list['extreme'] = y_list_part
                z_list['extreme'] = z_list_part
                roll_list['extreme'] = roll_list_part
                pitch_list['extreme'] = pitch_list_part
                yaw_list['extreme'] = yaw_list_part

            elif file_name == file_name_list[-2]:
                invalid_cnt += len(example_list)
                RE_list['invalid'] = RE_list_part
                TE_list['invalid'] = TE_list_part
                time_cost_list['invalid'] = time_cost_list_part
                x_list['invalid'] = x_list_part
                y_list['invalid'] = y_list_part
                z_list['invalid'] = z_list_part
                roll_list['invalid'] = roll_list_part
                pitch_list['invalid'] = pitch_list_part
                yaw_list['invalid'] = yaw_list_part

                        
    RE_mean = {}
    TE_mean = {}
    time_cost_mean = {}
    x_mean = {}
    y_mean = {}
    z_mean = {}
    roll_mean = {}
    pitch_mean = {}
    yaw_mean = {}

    time_cost_total_list = []

    for key in RE_list.keys():
        RE_mean[key] = sum(RE_list[key]) / len(RE_list[key])
        TE_mean[key] = sum(TE_list[key]) / len(TE_list[key])
        time_cost_mean[key] = sum(time_cost_list[key]) / len(time_cost_list[key])
        x_mean[key] = sum(x_list[key]) / len(x_list[key])
        y_mean[key] = sum(y_list[key]) / len(y_list[key])
        z_mean[key] = sum(z_list[key]) / len(z_list[key])
        roll_mean[key] = sum(roll_list[key]) / len(roll_list[key])
        pitch_mean[key] = sum(pitch_list[key]) / len(pitch_list[key])
        yaw_mean[key] = sum(yaw_list[key]) / len(yaw_list[key])
        if key != 'extreme':
            time_cost_total_list += time_cost_list[key]

    if visualize:
        # for key in RE_list.keys():
            # plt.figure(figsize=(3, 5))
            # plt.boxplot([RE_list[key], TE_list[key]], labels=['RE(' + str(len(RE_list[key])), 'TE' + str(len(TE_list[key])) + ')'])
            # plt.title(f'RE & TE of {key} group')
            # plt.xticks(rotation=45, ha='right')
            # plt.grid(True)

            # plt.figure(figsize=(5, 5))
            # plt.boxplot([x_list[key], y_list[key], z_list[key], roll_list[key], pitch_list[key], yaw_list[key]], labels=['x(' + str(len(x_list[key])), 'y(' + str(len(y_list[key])), 'z(' + str(len(z_list[key])), 'roll(' + str(len(roll_list[key])), 'pitch(' + str(len(pitch_list[key])), 'yaw(' + str(len(yaw_list[key])) + ')'])
            # plt.title(f'x y z of {key} extrinsic')
            # plt.xticks(rotation=45, ha='right')
            # plt.grid(True)

            # plt.figure(figsize=(3, 5))
            # plt.boxplot(time_cost_list[key], labels=['time_cost(' + str(len(time_cost_list[key])) + ')'])
            # plt.title(f'time_cost of {key} extrinsic')
            # plt.xticks(rotation=45, ha='right')
            # plt.grid(True)

        plt.figure(figsize=(3, 5))
        plt.boxplot(time_cost_total_list, labels=['time_cost(' + str(len(time_cost_total_list)) + ')'])
        plt.title(f'time_cost of python extrinsic')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)

        plt.show()

    total_cnt = easy_cnt + medium_cnt + hard_cnt + extreme_cnt + invalid_cnt + no_common_cnt

    print(f'total: {total_cnt}, easy_cnt: {easy_cnt}, medium_cnt: {medium_cnt}, hard_cnt: {hard_cnt}, extreme_cnt: {extreme_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    # print(f'easy_cnt / total_cnt: {easy_cnt / total_cnt}, medium_cnt / total_cnt: {medium_cnt / total_cnt}, hard_cnt / total_cnt: {hard_cnt / total_cnt}, invalid_cnt / total_cnt: {invalid_cnt / total_cnt}, no_common_cnt / total_cnt: {no_common_cnt / total_cnt}')
    # print(f'easy_cnt / (valid + valid_bad): {(easy_cnt / (total_cnt - invalid_cnt - no_common_cnt))}, medium_cnt / (valid + valid_bad): {(medium_cnt / (total_cnt - invalid_cnt - no_common_cnt))}, hard_cnt / (valid + valid_bad): {(hard_cnt / (total_cnt - invalid_cnt - no_common_cnt))} , extreme_cnt / (valid + valid_bad): {(extreme_cnt / (total_cnt - invalid_cnt - no_common_cnt))}')
    print('----------------------------------------')
    for key in RE_list.keys():
        print(f'{key} group: RE_mean: {RE_mean[key]}, TE_mean: {TE_mean[key]}, time_cost_mean: {time_cost_mean[key]}')
        print(f'{key} group: delta_T_6DOF_mean_seperate_cal(x,y,z,roll,pitch,yaw): {[x_mean[key], y_mean[key], z_mean[key], roll_mean[key], pitch_mean[key], yaw_mean[key]]}')
        print('----------------------------------------')


def count_matches_num_according_to_scene_difficulty(total_num = 650, k = 15, visualize = False):
    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    folder_name = r'/home/massimo/vehicle_infrastructure_calibration/intermediate_output/111/'#extrinsic_test_brute_KP/'
    file_name_list = [f'easy_group_k{k}_totalcnt{total_num}', f'medium_group_k{k}_totalcnt{total_num}', f'hard_group_k{k}_totalcnt{total_num}', f'extreme_group_k{k}_totalcnt{total_num}', f'invalid_group_k{k}_totalcnt{total_num}', f'no_common_group_k{k}_totalcnt{total_num}' ]
    
    easy_cnt = 0
    medium_cnt = 0
    hard_cnt = 0
    extreme_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    matches_num_list = {}
    available_matches_num_list = {}
    total_available_matches_num_list = {}

    for file_name in file_name_list:
        if os.path.exists(folder_name + file_name + '.json') == False:
            print(f'{folder_name + file_name + ".json"} not exist')
            continue

        with open(folder_name + file_name + '.json', 'r') as f:
            example_list = json.load(f)
            
        if file_name == file_name_list[-1]:
            no_common_cnt += len(example_list)
        else:
            matches_num_list_part = [example['matched_cnt'] for example in example_list]
            available_matches_num_list_part = [example['available_matches_cnt'] for example in example_list]
            if 'total_available_matches_cnt' in example_list[0].keys():
                total_available_matches_num_list_part = [example['total_available_matches_cnt'] for example in example_list]
            else:
                total_available_matches_num_list_part = [1 for example in example_list]

            if file_name == file_name_list[0]:
                easy_cnt += len(example_list)
                matches_num_list['easy'] = matches_num_list_part
                available_matches_num_list['easy'] = available_matches_num_list_part
                total_available_matches_num_list['easy'] = total_available_matches_num_list_part
                
            elif file_name == file_name_list[1]:
                medium_cnt += len(example_list)
                matches_num_list['medium'] = matches_num_list_part
                available_matches_num_list['medium'] = available_matches_num_list_part
                total_available_matches_num_list['medium'] = total_available_matches_num_list_part

            elif file_name == file_name_list[2]:
                hard_cnt += len(example_list)
                matches_num_list['hard'] = matches_num_list_part
                available_matches_num_list['hard'] = available_matches_num_list_part
                total_available_matches_num_list['hard'] = total_available_matches_num_list_part

            elif file_name == file_name_list[3]:
                extreme_cnt += len(example_list)
                matches_num_list['extreme'] = matches_num_list_part
                available_matches_num_list['extreme'] = available_matches_num_list_part
                total_available_matches_num_list['extreme'] = total_available_matches_num_list_part

            elif file_name == file_name_list[-2]:
                invalid_cnt += len(example_list)
                matches_num_list['invalid'] = matches_num_list_part
                available_matches_num_list['invalid'] = available_matches_num_list_part
                total_available_matches_num_list['invalid'] = total_available_matches_num_list_part

    matches_num_mean = {}
    available_matches_num_mean = {}
    total_available_matches_num_mean = {}

    for key in matches_num_list.keys():
        matches_num_mean[key] = sum(matches_num_list[key]) / len(matches_num_list[key])
        available_matches_num_mean[key] = sum(available_matches_num_list[key]) / len(available_matches_num_list[key])
        total_available_matches_num_mean[key] = sum(total_available_matches_num_list[key]) / len(total_available_matches_num_list[key])

    if visualize:
        for key in matches_num_list.keys():
            plt.figure(figsize=(4, 5))
            plt.boxplot([matches_num_list[key], available_matches_num_list[key], total_available_matches_num_list[key]], labels=[f'm', f'a_m', f't_a_m'])
            formatted_string = f'{key}({len(matches_num_list[key])});m/a_m:{matches_num_mean[key]/available_matches_num_mean[key]:.3f};a_m/t_a_m:{available_matches_num_mean[key]/total_available_matches_num_mean[key]:.3f}'
            plt.title(formatted_string)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)

        plt.show()


    total_cnt = easy_cnt + medium_cnt + hard_cnt + extreme_cnt + invalid_cnt + no_common_cnt

    print(f'total: {total_cnt}, easy_cnt: {easy_cnt}, medium_cnt: {medium_cnt}, hard_cnt: {hard_cnt}, extreme_cnt: {extreme_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    print('----------------------------------------')
    for key in matches_num_list.keys():
        print(f'{key} group: matches_num_mean: {matches_num_mean[key]}')
        print(f'{key} group: available_matches_num_mean: {available_matches_num_mean[key]}')
        print(f'{key} group: m/a_m: {matches_num_mean[key]/available_matches_num_mean[key]}')
        print(f'{key} group: total_available_matches_num_mean: {total_available_matches_num_mean[key]}')
        print(f'{key} group: a_m/t_a_m: {available_matches_num_mean[key]/total_available_matches_num_mean[key]}')
        print('----------------------------------------')


def count_result_according_difficulty(total_cnt = 650, k = 15, folder_name = r'intermediate_output/'):

    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    success_cnt = 0
    mid_success_cnt = 0
    hard_success_cnt = 0
    valid_bad_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    success_group = []
    mid_success_group = []
    hard_success_group = []


    # easy_group = [] # RE < 1, TE < 1 -> valid
    # medium_group = [] # 1 <= RE <= 5, 1 <= TE <= 5 -> valid + valid_bad
    # hard_group = [] # 5 < RE < 10 or 5 < TE < 10 -> valid_bad
    # extreme_group = [] # RE > 10 or TE > 10 -> invalid
    # invalid_group = [] # invalid
    # no_common_group = [] # no_common

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            if os.path.exists(folder_name + file_name + str(cnt) + '.json') == False:
                print(f'{folder_name + file_name + str(cnt) + ".json"} not exist')
                continue

            with open(folder_name + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
            
            if file_name == file_name_list[0] or file_name == file_name_list[1]:
                
                for example in example_list:
                    
                    if example['TE'] < 1:
                        success_cnt += 1
                        success_group.append(example)
                    if example['TE'] < 2:
                        mid_success_cnt += 1
                        mid_success_group.append(example)
                    if example['TE'] < 5:
                        hard_success_cnt += 1
                        hard_success_group.append(example)
                        
                    # hard_success_cnt += 1
                    # if example['TE'] < 5:
                    #     hard_success_group.append(example)

                if file_name == file_name_list[0]:
                    valid_cnt += len(example_list)
                                
                elif file_name == file_name_list[1]:
                    valid_bad_cnt += len(example_list)

            if file_name == file_name_list[2] :
                invalid_cnt += len(example_list)

            elif file_name == file_name_list[3]:
                no_common_cnt += len(example_list)

    # mean of success_group
    RE_list = [example['RE'] for example in success_group]
    TE_list = [example['TE'] for example in success_group]
    time_cost_list = [example['cost_time'] for example in success_group]
    RE_mean = sum(RE_list) / len(RE_list)
    TE_mean = sum(TE_list) / len(TE_list)
    time_cost_mean = sum(time_cost_list) / len(time_cost_list)

    mid_RE_list = [example['RE'] for example in mid_success_group]
    mid_TE_list = [example['TE'] for example in mid_success_group]
    mid_time_cost_list = [example['cost_time'] for example in mid_success_group]
    mid_RE_mean = sum(mid_RE_list) / len(mid_RE_list)
    mid_TE_mean = sum(mid_TE_list) / len(mid_TE_list)
    mid_time_cost_mean = sum(mid_time_cost_list) / len(mid_time_cost_list)

    hard_RE_list = [example['RE'] for example in hard_success_group]
    hard_TE_list = [example['TE'] for example in hard_success_group]
    hard_time_cost_list = [example['cost_time'] for example in hard_success_group]
    hard_RE_mean = sum(hard_RE_list) / len(hard_RE_list)
    hard_TE_mean = sum(hard_TE_list) / len(hard_TE_list)
    hard_time_cost_mean = sum(hard_time_cost_list) / len(hard_time_cost_list)


    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    print(f'success_cnt: {success_cnt}, success_rate: {success_cnt / total_cnt}')
    print(f'mid_success_cnt: {mid_success_cnt}, mid_success_rate: {mid_success_cnt / total_cnt}')
    print(f'hard_success_cnt: {hard_success_cnt}, hard_success_rate: {hard_success_cnt / total_cnt}')
    print(f'success RE_mean: {RE_mean}, TE_mean: {TE_mean}, time_cost_mean: {time_cost_mean}')
    print(f'mid_success RE_mean: {mid_RE_mean}, TE_mean: {mid_TE_mean}, time_cost_mean: {mid_time_cost_mean}')
    print(f'hard_success RE_mean: {hard_RE_mean}, TE_mean: {hard_TE_mean}, time_cost_mean: {hard_time_cost_mean}')

    # if len(easy_group):
    #     with open(os.path.join(output_folder, f'easy_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(easy_group, f)

    # if len(medium_group):
    #     with open(os.path.join(output_folder, f'medium_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(medium_group, f)

    # if len(hard_group):
    #     with open(os.path.join(output_folder, f'hard_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(hard_group, f)

    # if len(extreme_group):
    #     with open(os.path.join(output_folder, f'extreme_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(extreme_group, f)

    # if len(invalid_group):
    #     with open(os.path.join(output_folder, f'invalid_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(invalid_group, f)

    # if len(no_common_group):
    #     with open(os.path.join(output_folder, f'no_common_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(no_common_group, f)

    print('finished')


def count_time_cost_according_difficulty(total_cnt = 650, k = 15, folder_name = r'intermediate_output/'):

    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    success_cnt = 0
    hard_success_cnt = 0
    valid_bad_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    success_time_cost_list = []
    hard_success_time_cost_list = []
    overall_time_cost_list = []
    overoverall_time_cost_list = []

    # easy_group = [] # RE < 1, TE < 1 -> valid
    # medium_group = [] # 1 <= RE <= 5, 1 <= TE <= 5 -> valid + valid_bad
    # hard_group = [] # 5 < RE < 10 or 5 < TE < 10 -> valid_bad
    # extreme_group = [] # RE > 10 or TE > 10 -> invalid
    # invalid_group = [] # invalid
    # no_common_group = [] # no_common

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            if os.path.exists(folder_name + file_name + str(cnt) + '.json') == False:
                print(f'{folder_name + file_name + str(cnt) + ".json"} not exist')
                continue

            with open(folder_name + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
            
            if file_name == file_name_list[0] or file_name == file_name_list[1]:
                
                for example in example_list:
                    
                    if example['TE'] < 1:
                        success_cnt += 1
                        success_time_cost_list.append(example['cost_time'])
                    if example['TE'] < 2:
                        hard_success_cnt += 1
                        hard_success_time_cost_list.append(example['cost_time'])
                        
                    hard_success_cnt += 1
                    overall_time_cost_list.append(example['cost_time'])

                if file_name == file_name_list[0]:
                    valid_cnt += len(example_list)
                                
                elif file_name == file_name_list[1]:
                    valid_bad_cnt += len(example_list)

            if file_name == file_name_list[2] :
                invalid_cnt += len(example_list)

            elif file_name == file_name_list[3]:
                no_common_cnt += len(example_list)

            for example in example_list:
                overoverall_time_cost_list.append(example['cost_time'])

    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    if len(success_time_cost_list):
        mean_success_time_cost = sum(success_time_cost_list) / len(success_time_cost_list)
        print(f'success_cnt: {success_cnt}, success_rate: {success_cnt / total_cnt} , mean_success_time_cost: {mean_success_time_cost}')
    
    if len(hard_success_time_cost_list):
        mean_hard_success_time_cost = sum(hard_success_time_cost_list) / len(hard_success_time_cost_list)
        print(f'hard_success_cnt: {hard_success_cnt}, hard_success_rate: {hard_success_cnt / total_cnt} , mean_hard_success_time_cost: {mean_hard_success_time_cost}')
    
    if len(overall_time_cost_list):
        mean_overall_time_cost = sum(overall_time_cost_list) / len(overall_time_cost_list)
        print(f'success+hard_success: {valid_bad_cnt+valid_cnt} - mean_overall_time_cost: {mean_overall_time_cost}')
    
    if len(overoverall_time_cost_list):
        mean_overoverall_time_cost = sum(overoverall_time_cost_list) / len(overoverall_time_cost_list)
        print(f'total: {total_cnt}, mean_overoverall_time_cost: {mean_overoverall_time_cost}')

def count_pointcloud_based_result_according_difficulty(total_cnt = 650, k = 15, folder_name = r'intermediate_output/'):

    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    success_cnt = 0
    hard_success_cnt = 0
    valid_bad_cnt = 0

    success_group = []
    hard_success_group = []

    # easy_group = [] # RE < 1, TE < 1 -> valid
    # medium_group = [] # 1 <= RE <= 5, 1 <= TE <= 5 -> valid + valid_bad
    # hard_group = [] # 5 < RE < 10 or 5 < TE < 10 -> valid_bad
    # extreme_group = [] # RE > 10 or TE > 10 -> invalid
    # invalid_group = [] # invalid
    # no_common_group = [] # no_common

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            if os.path.exists(folder_name + "/" + file_name + str(cnt) + '.json') == False:
                print(f'{folder_name + "/" + file_name + str(cnt) + ".json"} not exist')
                continue

            with open(folder_name + "/" + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
                
            for example in example_list:
                if example['TE'] < 10:
                    success_cnt += 1
                    success_group.append(example)
                if example['TE'] < 20:
                    hard_success_cnt += 1
                    hard_success_group.append(example)
            
            if file_name == file_name_list[0]:
                valid_cnt += len(example_list)
                            
            elif file_name == file_name_list[1]:
                valid_bad_cnt += len(example_list)



    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}')
    print(f'success_cnt: {success_cnt}, success_rate: {success_cnt / total_cnt}')
    # mean of success_group
    if len(success_group):
        roll_list = []
        pitch_list = []
        yaw_list = []
        for example in success_group:
            i = 0
            for alpha in example['delta_T_6DOF'][3:]:
                if alpha > 90:
                    alpha = alpha - 180                         
                elif alpha < -90: 
                    alpha = alpha + 180
                if i == 0:
                    roll_list.append(np.abs(alpha))
                elif i == 1:
                    pitch_list.append(np.abs(alpha))
                elif i == 2:
                    yaw_list.append(np.abs(alpha))
                i += 1
        RE_list = [np.linalg.norm([roll_list[i], pitch_list[i], yaw_list[i]]) for i in range(len(roll_list))]
        TE_list = [example['TE'] for example in success_group]
        time_cost_list = [example['cost_time'] for example in success_group]
        RE_mean = sum(RE_list) / len(RE_list)
        TE_mean = sum(TE_list) / len(TE_list)
        time_cost_mean = sum(time_cost_list) / len(time_cost_list)
        print(f'success RE_mean: {RE_mean}, TE_mean: {TE_mean}, time_cost_mean: {time_cost_mean}')        

    print(f'hard_success_cnt: {hard_success_cnt}, hard_success_rate: {hard_success_cnt / total_cnt}')
    if len(hard_success_group):
        roll_list = []
        pitch_list = []
        yaw_list = []
        for example in hard_success_group:
            i = 0
            for alpha in example['delta_T_6DOF'][3:]:
                if alpha > 90:
                    alpha = alpha - 180                         
                elif alpha < -90: 
                    alpha = alpha + 180
                if i == 0:
                    roll_list.append(np.abs(alpha))
                elif i == 1:
                    pitch_list.append(np.abs(alpha))
                elif i == 2:
                    yaw_list.append(np.abs(alpha))
                i += 1
        hard_RE_list = [np.linalg.norm([roll_list[i], pitch_list[i], yaw_list[i]]) for i in range(len(roll_list))]
        hard_TE_list = [example['TE'] for example in hard_success_group]
        hard_time_cost_list = [example['cost_time'] for example in hard_success_group]
        hard_RE_mean = sum(hard_RE_list) / len(hard_RE_list)
        hard_TE_mean = sum(hard_TE_list) / len(hard_TE_list)
        hard_time_cost_mean = sum(hard_time_cost_list) / len(hard_time_cost_list)
        print(f'hard_success RE_mean: {hard_RE_mean}, TE_mean: {hard_TE_mean}, time_cost_mean: {hard_time_cost_mean}')
   
    
    # if len(easy_group):
    #     with open(os.path.join(output_folder, f'easy_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(easy_group, f)

    # if len(medium_group):
    #     with open(os.path.join(output_folder, f'medium_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(medium_group, f)

    # if len(hard_group):
    #     with open(os.path.join(output_folder, f'hard_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(hard_group, f)

    # if len(extreme_group):
    #     with open(os.path.join(output_folder, f'extreme_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(extreme_group, f)

    # if len(invalid_group):
    #     with open(os.path.join(output_folder, f'invalid_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(invalid_group, f)

    # if len(no_common_group):
    #     with open(os.path.join(output_folder, f'no_common_group_k{k}_totalcnt{total_cnt}.json'), 'w') as f:
    #         json.dump(no_common_group, f)

    print('finished')
    
def plot_violin_plot(data, title, x_label, y_label, save_path = None):
    plt.figure(figsize=(5, 5))
    plt.violinplot(data, showmeans=True, showmedians=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()



# 不同策略对比
def get_pair_label(data_folder, total_cnt = 650, k = 15):
    labels = ['valid', 'valid_bad', 'invalid', 'no_common']

    pair_label_dict = {}
    pair_content_dict = {}
    extra_key_content_dict = {}

    for cnt in range(50, total_cnt + 1, 50):
        for label in labels:
            file_name = label + '_extrinsic_k' + str(k) + '_cnt' + str(cnt) + '.json'
            if os.path.exists(data_folder + file_name) == False:
                continue

            with open(data_folder + file_name, 'r') as f:
                example_list = json.load(f)

            for example in example_list:
                
                infra_id = example['infra_file_name']
                vehicle_id = example['vehicle_file_name']

                pair_label_dict[(infra_id, vehicle_id)] = label

                key = ['result_matched_cnt',  'stability', 'RE', 'TE']
                key_content = {k: example[k] for k in key if k in example.keys()}

                extra_key = ['filtered_available_matches_cnt', 'total_available_matches_cnt']
                extra_key_content = {k: example[k] for k in extra_key}

                pair_content_dict[(infra_id, vehicle_id)] = key_content
                extra_key_content_dict[(infra_id, vehicle_id)] = extra_key_content
                

    return pair_label_dict, pair_content_dict, extra_key_content_dict


def save_pair_with_different_effect(true_match_result_data_folder, data_folder, output_folder, total_cnt = 650, k = 15):

    pair_label_dict, pair_content_dict, extra_key_content_dict = get_pair_label(data_folder, total_cnt, k)

    true_match_pair_label_dict, true_match_pair_content_dict, _ = get_pair_label(true_match_result_data_folder, total_cnt, k)

    pair_with_different_effect_list = []

    for pair, label in pair_label_dict.items():
        if pair in true_match_pair_label_dict.keys():
            if true_match_pair_label_dict[pair] != label:
                pair_with_different_effect_content = {}
                pair_with_different_effect_content['pair'] = list(pair)
                pair_with_different_effect_content['result_label'] = label
                pair_with_different_effect_content['true_match_label'] = true_match_pair_label_dict[pair]
                pair_with_different_effect_content['true_match_result_content'] = true_match_pair_content_dict[pair]
                pair_with_different_effect_content['result_content'] = pair_content_dict[pair]
                pair_with_different_effect_content['available_matches_cnt'] = extra_key_content_dict[pair]
                pair_with_different_effect_list.append(pair_with_different_effect_content)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_folder + 'pair_with_different_effect.json', 'w') as f:
        json.dump(pair_with_different_effect_list, f)


def analyze_pair_with_different_effect_json(true_match_folder_name, data_folder_name, output_folder_name):
    # 分析不同方法对比中不同匹配对的变化情况
    total_num = 6600
    true_match_result_data_folder = fr'new_clean_result/{true_match_folder_name}/all_dataset/'
    data_folder = fr'new_clean_result/{data_folder_name}/all_dataset/'
    output_folder = fr'new_clean_result/analyze/{output_folder_name}/'
    save_pair_with_different_effect(true_match_result_data_folder, data_folder, output_folder, total_cnt = total_num, k = 15)

    pairs_path = fr'{output_folder}pair_with_different_effect.json'

    with open(pairs_path, 'r') as f:
        pair_with_different_effect_list = json.load(f)
        combination_content_list_dict = {}
        for pair_with_different_effect in pair_with_different_effect_list:
            result_label = pair_with_different_effect['result_label']
            true_match_label = pair_with_different_effect['true_match_label']
            combination = (result_label, true_match_label)
            if combination not in combination_content_list_dict.keys():
                combination_content_list_dict[combination] = []
            combination_content_list_dict[combination].append(pair_with_different_effect)

    keys = ['valid2valid_bad', 'valid_bad2valid', 'invalid2valid_bad', 'invalid2valid']

    print('len(pair_with_different_effect_list):', len(pair_with_different_effect_list))

    # save
    # for key in keys:
    #     combination = tuple(key.split('2'))
    #     print(f'{key} group: {len(combination_content_list_dict[combination])}')
    #     with open(output_folder + key + '.json', 'w') as f:
    #         json.dump(combination_content_list_dict[combination], f)

    # 初始化一个字典来存储不同 key 的稳定性分布
    stability_distribution_dict = {key: {} for key in keys}

    for key in keys:
        combination = tuple(key.split('2'))
        for pair in combination_content_list_dict[combination]:
            stability = pair['result_content']['stability']
            if stability not in stability_distribution_dict[key]:
                stability_distribution_dict[key][stability] = 0
            stability_distribution_dict[key][stability] += 1

    # print(sorted(stability_distribution_dict.items(), key=lambda x: x[0]))

    # 获取所有可能的 stability 值
    all_stabilities = sorted(set(stability for dist in stability_distribution_dict.values() for stability in dist.keys()))

    # 准备数据以便绘制堆叠柱状图
    stacked_data = {key: [stability_distribution_dict[key].get(stability, 0) for stability in all_stabilities] for key in keys}

    # 绘制堆叠柱状图
    plt.figure(figsize=(10, 6))
    bottom = [0] * len(all_stabilities)

    for key in keys:
        plt.bar(all_stabilities, stacked_data[key], bottom=bottom, label=key)
        bottom = [i + j for i, j in zip(bottom, stacked_data[key])]

    plt.xlabel('Stability')
    plt.ylabel('Count')
    plt.title('Stability Distribution for Different Keys')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_high_stability_pairs(data_folder_name, output_folder_name, total_cnt = 650, k = 15):

    data_folder = fr'new_clean_result/{data_folder_name}/all_dataset/'
    output_folder = fr'new_clean_result/analyze/{output_folder_name}/'

    high_stability_pairs = []

    for cnt in range(50, total_cnt + 1, 50):
        file_name = f'valid_bad_extrinsic_k{k}_cnt{cnt}.json'
        if os.path.exists(data_folder + file_name) == False:
            continue
        
        with open(data_folder + file_name, 'r') as f:
            example_list = json.load(f)

        for example in example_list:
            infra_id = example['infra_file_name']
            vehicle_id = example['vehicle_file_name']
            pair = (infra_id, vehicle_id)
            if example['stability'] >= 3:
                high_stability_pairs.append(example)

    print('len(high_stability_pairs):', len(high_stability_pairs))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_folder + 'valid_bad_high_stability_pairs.json', 'w') as f:
        json.dump(high_stability_pairs, f)



def print_avg_RE_TE_time_cost(example_list, title):
    RE_list = [example['RE'] for example in example_list]
    TE_list = [example['TE'] for example in example_list]
    time_cost_list = [example['cost_time'] for example in example_list]
    RE_mean = sum(RE_list) / len(RE_list) if len(RE_list) else 0
    TE_mean = sum(TE_list) / len(TE_list) if len(TE_list) else 0
    time_cost_mean = sum(time_cost_list) / len(time_cost_list) if len(time_cost_list) else 0
    print(f'{title} RE_mean: {RE_mean}, TE_mean: {TE_mean}, time_cost_mean: {time_cost_mean}')


def devide_group_according_scene_difficulty(stability_threshold = 5, total_cnt = 650, k = 15, folder_name = r'intermediate_output/', output_folder = r'intermediate_output/111/', print_stability = False):

    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    success_cnt = 0
    valid_bad_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    easy_group = [] # RE < 1, TE < 1 -> valid
    medium_group = [] # 1 <= RE <= 5, 1 <= TE <= 5 -> valid + valid_bad
    hard_group = [] # 5 < RE < 10 or 5 < TE < 10 -> valid_bad
    extreme_group = [] # RE > 10 or TE > 10 -> invalid
    invalid_group = [] # invalid
    no_common_group = [] # no_common

    stability_cnt = {}

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            if os.path.exists(folder_name + file_name + str(cnt) + '.json') == False:
                continue

            with open(folder_name + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
            
            if file_name == file_name_list[0] or file_name == file_name_list[1]:
                
                # for example in example_list:
                #     if example['RE'] < 1 and example['TE'] < 1:
                #         easy_group.append(example)
                #     elif example['RE'] <= 5 and example['TE'] <= 5:
                #         medium_group.append(example)
                #     elif example['RE'] <= 10 and example['TE'] <= 10:
                #         hard_group.append(example)
                #     else:
                #         extreme_group.append(example)

                for example in example_list:
                    if 'stability' in example.keys():
                        stability_cnt[example['stability']] = stability_cnt.get(example['stability'], 0) + 1

                        if example['stability'] < stability_threshold:
                            continue

                    if example['RE'] < 1 and example['TE'] < 1:
                        success_cnt += 1
                        easy_group.append(example)
                    elif example['RE'] < 2 and example['TE'] < 2:
                        success_cnt += 1
                        medium_group.append(example)
                    elif example['RE'] < 5 and example['TE'] < 5:
                        hard_group.append(example)
                    else:
                        extreme_group.append(example)

                if file_name == file_name_list[0]:
                    valid_cnt += len(example_list)
                                
                elif file_name == file_name_list[1]:
                    valid_bad_cnt += len(example_list)

            if file_name == file_name_list[2] :
                invalid_cnt += len(example_list)
                invalid_group += example_list

            elif file_name == file_name_list[3]:
                no_common_cnt += len(example_list)
                no_common_group += example_list

    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    
    if print_stability:
        print(stability_cnt)
    print(f'total: {total_cnt}, easy_group(RE,TE<1): {len(easy_group)}, medium_group(1<=RE,TE<2): {len(medium_group)}, hard_group(2<=RE,TE<5): {len(hard_group)}, extreme_group(RE,TE>5): {len(extreme_group)}, invalid_group: {len(invalid_group)}, no_common_group: {len(no_common_group)}')


    print_avg_RE_TE_time_cost(easy_group, '(RE,TE<1)')
    print_avg_RE_TE_time_cost(easy_group+medium_group, '(RE,TE<2)')
    print_avg_RE_TE_time_cost(easy_group+medium_group+hard_group, '(RE,TE<5)')

    print(f'success rate(TE,RE<1):{len(easy_group)/total_cnt}, success rate(TE,RE<2):{ success_cnt / total_cnt}, success rate(TE,RE<5):{(len(easy_group)+len(medium_group)+len(hard_group))/total_cnt}')



if __name__ == "__main__":
    # analyze_bad_test()

    # count_test_result()
    
    total_num = 100
    
    # data_folder = r'new_clean_result/extrinsic_category_core_centerpoint_distance_vertex_distance_svd8point_threshold/common_boxes_filtered_dataset/'
    # intermediate_folder = r'new_clean_result/extrinsic_category_core_centerpoint_distance_vertex_distance_svd8point_threshold/common_boxes_filtered_dataset/analyze/'

    # data_folder = r'new_clean_result/extrinsic_category_core_centerpoint_distance_vertex_distance_svd8point_top matches/common_boxes_filtered_dataset/'
    # intermediate_folder = r'new_clean_result/extrinsic_category_core_centerpoint_distance_vertex_distance_svd8point_top matches/common_boxes_filtered_dataset/analyze/'

    data_folder = r'/home/massimo/vehicle_infrastructure_calibration/new_clean_result/V2XSim_dataset_ndt/'
    intermediate_folder = r'/home/massimo/vehicle_infrastructure_calibration/new_clean_result/V2XSim_dataset_ndt/analyze/'

    # data_folder = r'new_clean_result/extrinsic_category_core_centerpoint_distance_vertex_distance_svd8point_threshold/all_dataset/'
    # intermediate_folder = r'new_clean_result/extrinsic_category_core_centerpoint_distance_vertex_distance_svd8point_threshold/all_dataset/analyze/'


    # if not os.path.exists(intermediate_folder):
    #     os.makedirs(intermediate_folder)


    # for threshold in range(0, 20, 4):
    #     print('threshold:', threshold)
    #     devide_group_according_scene_difficulty(stability_threshold = threshold, total_cnt = total_num, k = 15, folder_name = data_folder, output_folder = intermediate_folder, print_stability=(threshold==0))
    #     print('-----------------------------------')

    devide_group_according_scene_difficulty(total_cnt = total_num, k = 15, folder_name = data_folder, output_folder = intermediate_folder)

    count_test_result_according_scene_difficulty(total_num = total_num, k = 15, visualize=False, folder_name = intermediate_folder)

    # count_result_according_difficulty(total_cnt = total_num, k = 15, folder_name = data_folder)

    # count_time_cost_according_difficulty(total_cnt = total_num, k = 15, folder_name = data_folder)

    # count_pointcloud_based_result_according_difficulty(total_cnt = total_num, k = 15, folder_name = data_folder)

    # count_matches_num_according_to_scene_difficulty(total_num, visualize=True)



    # true_match_folder_name = 'extrinsic_true_matches_distancecorresponding_svd8point_all'
    # data_folder_name = 'extrinsic_category_core_distancecorresponding_svd8point_all'
    # output_folder_name = 'distancecorresponding_svd8point_all_true_matches_comparison'

    # true_match_folder_name = 'extrinsic_true_matches_distancecorresponding_svd8point_threshold'
    # data_folder_name = 'extrinsic_category_core_distancecorresponding_svd8point_threshold'
    # output_folder_name = 'distancecorresponding_svd8point_threshold_true_matches_comparison'

    # # # 分析不同方法对比中不同匹配对的变化情况
    # analyze_pair_with_different_effect_json(true_match_folder_name, data_folder_name, output_folder_name)

    # get_high_stability_pairs(data_folder_name, output_folder_name, total_cnt = 6600, k = 15)    
    
    # pairs_path = fr'new_clean_result/analyze/{output_folder_name}/valid_bad2valid.json'

    # with open(pairs_path, 'r') as f:
    #     valid_bad2valid_list = json.load(f)
    #     high_stability_list = []
    #     # mid_stability_list = []
    #     low_stability_list = []
    #     for pair in valid_bad2valid_list:
    #         if pair['result_content']['stability'] <= 3:
    #             low_stability_list.append(pair)
    #         # elif pair['true_match_result_content']['stability'] <= 3:
    #         #     mid_stability_list.append(pair)
    #         else:
    #             high_stability_list.append(pair)
            
    # # keys = ['low_stability', 'high_stability']
    # # for key in keys:
    # #     with open(pairs_path.replace('.json', f'_{key}.json'), 'w') as f:
    # #         if key == 'low_stability':
    # #             json.dump(low_stability_list, f)
    # #         elif key == 'high_stability':
    # #             json.dump(high_stability_list, f)

    # high_stability_distribution_cnt = {}
    # low_stability_distribution_cnt = {}

    # for pair in high_stability_list:
    #     stability = pair['result_content']['stability']
    #     if stability not in high_stability_distribution_cnt.keys():
    #         high_stability_distribution_cnt[stability] = 0
    #     high_stability_distribution_cnt[stability] += 1    
    # high_stability_distribution_cnt = sorted(high_stability_distribution_cnt.items(), key=lambda x: x[0])

    # for pair in low_stability_list:
    #     stability = pair['result_content']['stability']
    #     if stability not in low_stability_distribution_cnt.keys():
    #         low_stability_distribution_cnt[stability] = 0
    #     low_stability_distribution_cnt[stability] += 1
    # low_stability_distribution_cnt = sorted(low_stability_distribution_cnt.items(), key=lambda x: x[0])

    # print('len(high_stability_list):', len(high_stability_list))
    # print('high_stability_distribution_cnt:', high_stability_distribution_cnt)
    # print('len(low_stability_list):', len(low_stability_list))
    # print('low_stability_distribution_cnt:', low_stability_distribution_cnt)

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(5, 5))
    # plt.bar([str(i[0]) for i in low_stability_distribution_cnt], [i[1] for i in low_stability_distribution_cnt], label='low_stability')
    # plt.bar([str(i[0]) for i in high_stability_distribution_cnt], [i[1] for i in high_stability_distribution_cnt], label='high_stability')
    # plt.legend()
    # plt.xlabel('stability')
    # plt.ylabel('cnt')
    # plt.title('valid_bad2valid')
    # plt.grid(True)
    # plt.show()
