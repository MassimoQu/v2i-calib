import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
sys.path.append('./process')
from CooperativeBatchingReader import CooperativeBatchingReader
from CorrespondingDetector import CorrespondingDetector
from BoxesMatch import BoxesMatch
from Filter3dBoxes import Filter3dBoxes
from extrinsic_utils import implement_T_3dbox_object_list

def plot_hist_boxplot(list, str_title = 'Error Matches Scores'):
    # 创建一个图形和两个子图
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # 在第一个子图中绘制直方图
    axs[0].hist(list, bins=10, color='skyblue', edgecolor='black')
    axs[0].set_title('Histogram of' + str_title)
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Frequency')

    # 在第二个子图中绘制箱线图
    axs[1].boxplot(list, vert=True, patch_artist=True)
    axs[1].set_title('Boxplot of' + str_title)
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Value')

    # 显示图形
    plt.tight_layout()
    plt.show()

def batching_test_KP_svd(data_difficulty = 'easy', filter_num = 15, verbose = False, test_num = 200):

    if data_difficulty == 'all':
        path_data_info = '/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json'
    elif data_difficulty in ['easy', 'hard']:
        path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/' + data_difficulty + '_data_info.json'
    else:
        raise ValueError('data_difficulty should be easy or hard')
    
    reader = CooperativeBatchingReader(path_data_info = path_data_info)

    error_matches_score_list = []
    right_matches_score_list = []
    cnt = 0

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in reader.generate_infra_vehicle_bboxes_object_list():
        # try:
        if True:
            
            if verbose:
                print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name}')
                print(f'infra_total_box_cnt: {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}')

            filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(filter_num)
            
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
            filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
            filtered_available_matches = CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list).get_matches()
            
            converted_original_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
            total_available_matches_cnt = CorrespondingDetector(converted_original_infra_boxes_object_list, vehicle_boxes_object_list).get_matched_num()

            boxes_match = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list)
            matches_with_score_list = boxes_match.get_matches_with_score()

            error_matches_score_list += [score for match, score in matches_with_score_list if match not in filtered_available_matches]
            right_matches_score_list += [score for match, score in matches_with_score_list if match in filtered_available_matches]

            print(cnt)
            cnt += 1

            if cnt >= test_num:
                break

    plot_hist_boxplot(error_matches_score_list, 'Error Matches Scores')
    plot_hist_boxplot(right_matches_score_list, 'Right Matches Scores')

if __name__ == '__main__':
    batching_test_KP_svd(data_difficulty = 'easy', filter_num = 15, verbose = False, test_num = 200)
    batching_test_KP_svd(data_difficulty = 'hard', filter_num = 15, verbose = False, test_num = 200)