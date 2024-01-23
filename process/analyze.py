import os
import numpy as np
import matplotlib.pyplot as plt
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
from BoxesMatch import BoxesMatch
from Filter3dBoxes import Filter3dBoxes
from PSO_Executor import PSO_Executor
from extrinsic_utils import implement_T_3dbox_object_list, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF,convert_6DOF_to_T
from BBoxVisualizer_open3d import BBoxVisualizer_open3d

def analyze_bad_test():

    with open(r'intermediate_output/extrinsic_test/valid_bad_extrinsic_k15_cnt50.json', 'r') as f:
        valid_bad_example_list = json.load(f)

    cnt = 0

    for valid_bad_example in valid_bad_example_list:

        if cnt >= 2:
            break
        cnt += 1

        infra_file_name = valid_bad_example['infra_file_name']
        vehicle_file_name = valid_bad_example['vehicle_file_name']

        cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name)
        infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
        infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
        
        filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(15)
        filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(15)

        T_true = cooperative_reader.get_cooperative_T_i2v()
        T_6DOF_true = convert_T_to_6DOF(T_true)

        boxes_match = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, T_true, verbose=True)
        matches = boxes_match.get_matches()
        boxes_match.cal_matches_accuracy()

        pso = PSO_Executor(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, true_T_6DOF=T_6DOF_true, matches=matches, filter_num=15, verbose=True)
        T_6DOF_result = pso.get_best_T_6DOF()
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF_result, T_6DOF_true)
        print(f'infra_file_name, vehicle_file_name: {infra_file_name}, {vehicle_file_name}, RE: {RE}, TE: {TE}')
        print('--------------------------------------------------')

        BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([implement_T_3dbox_object_list(convert_6DOF_to_T(T_6DOF_result), filtered_infra_boxes_object_list), filtered_vehicle_boxes_object_list], [infra_pointcloud, vehicle_pointcloud], [[0, 1, 0], [1, 0, 0]])
        BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list), filtered_vehicle_boxes_object_list], [infra_pointcloud, vehicle_pointcloud], [[0, 1, 0], [1, 0, 0]])


def count_test_result():
    total_cnt = 200
    k = 15
    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    folder_name = r'intermediate_output/extrinsic_test_brute_KP_svd_non_divide_volume/'
    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    valid_bad_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    RE_list = []
    TE_list = []
    time_cost_list = []
    x_list = []
    y_list = []
    z_list = []
    roll_list = []
    pitch_list = []
    yaw_list = []

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            with open(folder_name + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
                
            if file_name == file_name_list[0]:
                valid_cnt += len(example_list)
                RE_list_part = [example['RE'] for example in example_list]
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

                RE_list += RE_list_part
                TE_list += TE_list_part
                time_cost_list += time_cost_list_part
                
                x_list += x_list_part
                y_list += y_list_part
                z_list += z_list_part
                roll_list += roll_list_part
                pitch_list += pitch_list_part
                yaw_list += yaw_list_part

            elif file_name == file_name_list[1]:
                valid_bad_cnt += len(example_list)
            elif file_name == file_name_list[2]:
                invalid_cnt += len(example_list)
            elif file_name == file_name_list[3]:
                no_common_cnt += len(example_list)
    
    RE_mean = sum(RE_list) / len(RE_list)
    TE_mean = sum(TE_list) / len(TE_list)
    time_cost_mean = sum(time_cost_list) / len(time_cost_list)
    x_mean = sum(x_list) / len(x_list)
    y_mean = sum(y_list) / len(y_list)
    z_mean = sum(z_list) / len(z_list)
    roll_mean = sum(roll_list) / len(roll_list)
    pitch_mean = sum(pitch_list) / len(pitch_list)
    yaw_mean = sum(yaw_list) / len(yaw_list)
    delta_T_6DOF_mean_seperate_cal = [x_mean, y_mean, z_mean, roll_mean, pitch_mean, yaw_mean]

    plt.figure(figsize=(4, 6))
    plt.boxplot([RE_list, TE_list], labels=['RE(' + str(len(RE_list)), 'TE' + str(len(TE_list)) + ')'])
    plt.title('RE & TE of valid extrinsic')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    # plt.show()

    plt.figure(figsize=(6, 6))
    plt.boxplot([x_list, y_list, z_list, roll_list, pitch_list, yaw_list], labels=['x(' + str(len(x_list)), 'y(' + str(len(y_list)), 'z(' + str(len(z_list)), 'roll(' + str(len(roll_list)), 'pitch(' + str(len(pitch_list)), 'yaw(' + str(len(yaw_list)) + ')'])
    plt.title('delta_T_6DOF of valid extrinsic')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)


    plt.figure(figsize=(4, 6))
    plt.boxplot(time_cost_list, labels=['time_cost(' + str(len(time_cost_list)) + ')'])
    plt.title('time_cost of valid extrinsic')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    print(f'reacall: {valid_cnt / (valid_bad_cnt + valid_cnt)}, valid_cnt / total_cnt: {valid_cnt / total_cnt}, valid_bad_cnt / total_cnt: {valid_bad_cnt / total_cnt},')
    print(f' invalid_cnt / total_cnt: {invalid_cnt / total_cnt}, no_common_cnt / total_cnt: {no_common_cnt / total_cnt}')
    print(f'RE_mean: {RE_mean}, TE_mean: {TE_mean}, time_cost_mean: {time_cost_mean}')
    print(f'delta_T_6DOF_mean_seperate_cal(x,y,z,roll,pitch,yaw): {delta_T_6DOF_mean_seperate_cal}')


def devide_group_according_scene_difficulty(total_cnt = 200, k = 15):

    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    folder_name = r'intermediate_output/'#extrinsic_test_brute_KP_svd_non_divide_volume/'
    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
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
                6*****************************************************
            
            if file_name == file_name_list[0] or file_name == file_name_list[1]:
                
                for example in example_list:
                    if example['RE'] < 1 and example['TE'] < 1:
                        easy_group.append(example)
                    elif example['RE'] <= 5 and example['TE'] <= 5:
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
    print(f'easy_group / total_cnt: {len(easy_group) / total_cnt}, medium_group / total_cnt: {len(medium_group) / total_cnt}, hard_group / total_cnt: {len(hard_group) / total_cnt}, extreme_group / total_cnt: {len(extreme_group) / total_cnt}, invalid_group / total_cnt: {len(invalid_group) / total_cnt}, no_common_group / total_cnt: {len(no_common_group) / total_cnt}')
    print(f'easy_group / (valid + valid_bad): {(len(easy_group) / (valid_cnt + valid_bad_cnt))}, medium_group / (valid + valid_bad): {(len(medium_group) / (valid_cnt + valid_bad_cnt))}, hard_group / (valid + valid_bad): {(len(hard_group) / (valid_cnt + valid_bad_cnt))}')

    if len(easy_group):
        with open(f'intermediate_output/111/easy_group_k{k}_totalcnt{total_cnt}.json', 'w') as f:
            json.dump(easy_group, f)

    if len(medium_group):
        with open(f'intermediate_output/111/medium_group_k{k}_totalcnt{total_cnt}.json', 'w') as f:
            json.dump(medium_group, f)

    if len(hard_group):
        with open(f'intermediate_output/111/hard_group_k{k}_totalcnt{total_cnt}.json', 'w') as f:
            json.dump(hard_group, f)

    if len(extreme_group):
        with open(f'intermediate_output/111/extreme_group_k{k}_totalcnt{total_cnt}.json', 'w') as f:
            json.dump(extreme_group, f)

    if len(invalid_group):
        with open(f'intermediate_output/111/invalid_group_k{k}_totalcnt{total_cnt}.json', 'w') as f:
            json.dump(invalid_group, f)

    if len(no_common_group):
        with open(f'intermediate_output/111/no_common_group_k{k}_totalcnt{total_cnt}.json', 'w') as f:
            json.dump(no_common_group, f)

    print('finished')


def count_test_result_according_scene_difficulty(total_num = 200, k = 15, visualize = False):
    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    folder_name = r'intermediate_output/111/'#extrinsic_test_brute_KP/'
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
            continue

        with open(folder_name + file_name + '.json', 'r') as f:
            example_list = json.load(f)
            
        if file_name == file_name_list[-2]:
            invalid_cnt += len(example_list)
        elif file_name == file_name_list[-1]:
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
                        
    RE_mean = {}
    TE_mean = {}
    time_cost_mean = {}
    x_mean = {}
    y_mean = {}
    z_mean = {}
    roll_mean = {}
    pitch_mean = {}
    yaw_mean = {}

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

    if visualize:
        for key in RE_list.keys():
            plt.figure(figsize=(3, 5))
            plt.boxplot([RE_list[key], TE_list[key]], labels=['RE(' + str(len(RE_list[key])), 'TE' + str(len(TE_list[key])) + ')'])
            plt.title(f'RE & TE of {key} group')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)

            plt.figure(figsize=(5, 5))
            plt.boxplot([x_list[key], y_list[key], z_list[key], roll_list[key], pitch_list[key], yaw_list[key]], labels=['x(' + str(len(x_list[key])), 'y(' + str(len(y_list[key])), 'z(' + str(len(z_list[key])), 'roll(' + str(len(roll_list[key])), 'pitch(' + str(len(pitch_list[key])), 'yaw(' + str(len(yaw_list[key])) + ')'])
            plt.title(f'x y z of {key} extrinsic')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)

            plt.figure(figsize=(3, 5))
            plt.boxplot(time_cost_list[key], labels=['time_cost(' + str(len(time_cost_list[key])) + ')'])
            plt.title(f'time_cost of {key} extrinsic')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True)

        plt.show()

    total_cnt = easy_cnt + medium_cnt + hard_cnt + extreme_cnt + invalid_cnt + no_common_cnt

    print(f'total: {total_cnt}, easy_cnt: {easy_cnt}, medium_cnt: {medium_cnt}, hard_cnt: {hard_cnt}, extreme_cnt: {extreme_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    # print(f'easy_cnt / total_cnt: {easy_cnt / total_cnt}, medium_cnt / total_cnt: {medium_cnt / total_cnt}, hard_cnt / total_cnt: {hard_cnt / total_cnt}, invalid_cnt / total_cnt: {invalid_cnt / total_cnt}, no_common_cnt / total_cnt: {no_common_cnt / total_cnt}')
    print(f'easy_cnt / (valid + valid_bad): {(easy_cnt / (total_cnt - invalid_cnt - no_common_cnt))}, medium_cnt / (valid + valid_bad): {(medium_cnt / (total_cnt - invalid_cnt - no_common_cnt))}, hard_cnt / (valid + valid_bad): {(hard_cnt / (total_cnt - invalid_cnt - no_common_cnt))} , extreme_cnt / (valid + valid_bad): {(extreme_cnt / (total_cnt - invalid_cnt - no_common_cnt))}')
    print('----------------------------------------')
    for key in RE_list.keys():
        print(f'{key} group: RE_mean: {RE_mean[key]}, TE_mean: {TE_mean[key]}, time_cost_mean: {time_cost_mean[key]}')
        print(f'{key} group: delta_T_6DOF_mean_seperate_cal(x,y,z,roll,pitch,yaw): {[x_mean[key], y_mean[key], z_mean[key], roll_mean[key], pitch_mean[key], yaw_mean[key]]}')
        print('----------------------------------------')


if __name__ == "__main__":
    # analyze_bad_test()

    # count_test_result()

    devide_group_according_scene_difficulty()

    count_test_result_according_scene_difficulty(visualize=False)

