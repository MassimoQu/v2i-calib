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
    k = 10
    # folder_name = r'intermediate_output/extrinsic_test/'
    # file_name_list = ['invalid_extrinsic_k15_cnt', 'no_common_view_k15_cnt', 'valid_extrinsic_k15_cnt', 'valid_bad_extrinsic_k15_cnt']

    folder_name = r'intermediate_output/'
    file_name_list = ['valid_extrinsic_k' + str(k) + '_cnt', 'valid_bad_extrinsic_k' + str(k) + '_cnt', 'invalid_extrinsic_k' + str(k) + '_cnt', 'no_common_view_k' + str(k) + '_cnt']
    
    valid_cnt = 0
    valid_bad_cnt = 0
    invalid_cnt = 0
    no_common_cnt = 0

    RE_mean = 0
    TE_mean = 0
    time_cost_mean = 0
    RE_list = []
    TE_list = []
    time_cost_list = []

    for cnt in range(50, total_cnt + 1, 50):
        for file_name in file_name_list:
            with open(folder_name + file_name + str(cnt) + '.json', 'r') as f:
                example_list = json.load(f)
                
            if file_name == file_name_list[0]:
                valid_cnt += len(example_list)
                RE_list_part = [example['RE'] for example in example_list]
                TE_list_part = [example['TE'] for example in example_list]
                time_cost_list_part = [example['cost_time'] for example in example_list]
                RE_list += RE_list_part
                TE_list += TE_list_part
                time_cost_list += time_cost_list_part
                RE_mean = sum(RE_list) / len(RE_list)
                TE_mean = sum(TE_list) / len(TE_list)
                time_cost_mean = sum(time_cost_list) / len(time_cost_list)

            elif file_name == file_name_list[1]:
                valid_bad_cnt += len(example_list)
            elif file_name == file_name_list[2]:
                invalid_cnt += len(example_list)
            elif file_name == file_name_list[3]:
                no_common_cnt += len(example_list)
    
    plt.figure()
    plt.boxplot([RE_list, TE_list], labels=['RE(' + str(len(RE_list)), 'TE' + str(len(TE_list)) + ')'])
    plt.title('RE & TE of valid extrinsic')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    # plt.show()

    plt.figure()
    plt.boxplot(time_cost_list, labels=['time_cost(' + str(len(time_cost_list)) + ')'])
    plt.title('time_cost of valid extrinsic')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

    print(f'total: {total_cnt}, valid_cnt: {valid_cnt}, valid_bad_cnt: {valid_bad_cnt}, invalid_cnt: {invalid_cnt}, no_common_cnt: {no_common_cnt}')
    print(f'reacall: {valid_cnt / (valid_bad_cnt + valid_cnt)}, valid_cnt / total_cnt: {valid_cnt / total_cnt}, valid_bad_cnt / total_cnt: {valid_bad_cnt / total_cnt},')
    print(f' invalid_cnt / total_cnt: {invalid_cnt / total_cnt}, no_common_cnt / total_cnt: {no_common_cnt / total_cnt}')
    print(f'RE_mean: {RE_mean}, TE_mean: {TE_mean}, time_cost_mean: {time_cost_mean}')




if __name__ == "__main__":
    # analyze_bad_test()

    count_test_result()

