import matplotlib.pyplot as plt
import json
import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./process/search')
from CooperativeBatchingReader import CooperativeBatchingReader
from Filter3dBoxes import Filter3dBoxes
from BoxesMatch import BoxesMatch
from CorrespondingDetector import CorrespondingDetector
from Matches2Extrinsics import Matches2Extrinsics
from extrinsic_utils import convert_6DOF_to_T, convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true, get_extrinsic_from_two_3dbox_object, implement_T_3dbox_object_list


def validate_BoxesMatches_score():
    '''
    validate the BoxesMatch score
    '''
    
    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader().generate_infra_vehicle_bboxes_object_list():
        
        infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(15)
        vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(15)

        matches_with_score_list = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()
        T_calculated = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=matches_with_score_list).get_combined_extrinsic(matches_filter_strategy='threshold_and_confidence')

        if len(matches_with_score_list) == 0:
            continue

        score_RETE_list = []

        cal_RE, cal_TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated, convert_T_to_6DOF(T_true))
        calculated_T_converted_infra_boxes_object_list = implement_T_3dbox_object_list(convert_6DOF_to_T(T_calculated), infra_boxes_object_list)
        center_point_precision = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_distance_corresponding_precision()
        vertex_point_precision = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, corresponding_strategy='vertex_distance').get_distance_corresponding_precision()
        calculated_score = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_Yscore()
        calculated_score += CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, corresponding_strategy='vertex_distance').get_Yscore()
        
        print(f'calculated: score: {calculated_score}, center_point_precision: {center_point_precision:.2f}, vertex_point_precision: {vertex_point_precision:.2f}, RE: {cal_RE:.2f}, TE: {cal_TE:.2f}')

        print(f'got {len(matches_with_score_list)} candidate matches')

        for match, score in matches_with_score_list:
            infra_box = infra_boxes_object_list[match[0]]
            vehicle_box = vehicle_boxes_object_list[match[1]]
            T_infra2vehicle = get_extrinsic_from_two_3dbox_object(infra_box, vehicle_box)
            
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_infra2vehicle, infra_boxes_object_list)
            center_point_precision = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list).get_distance_corresponding_precision()
            vertex_point_precision = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list, corresponding_strategy='vertex_distance').get_distance_corresponding_precision()

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T_infra2vehicle), convert_T_to_6DOF(T_true))

            score_RETE_list.append((score, RE, TE))

            print(f'matches: {match} ,score: {score}, center_point_precision: {center_point_precision:.2f}, vertex_point_precision: {vertex_point_precision:.2f}, RE: {RE:.2f}, TE: {TE:.2f}')

        input(f'above is the {infra_file_name} and {vehicle_file_name}, press any key to continue the next pair...')



def save_low_combine_accuracy_scenes_list():
    '''
    get the low combine accuracy scenes list
    '''

    cnt = 0
    save_cnt = 0

    low_combine_accuracy_scenes_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader(path_data_info='/home/massimo/vehicle_infrastructure_calibration/dataset_division/common_boxes_4_data_info.json').generate_infra_vehicle_bboxes_object_list():
        
        infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(15)
        vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(15)

        matches_with_score_list = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()
        T_calculated = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=matches_with_score_list).get_combined_extrinsic(matches_filter_strategy='threshold_and_confidence')

        if len(matches_with_score_list) == 0:
            continue

        cal_RE, cal_TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated, convert_T_to_6DOF(T_true))
        calculated_T_converted_infra_boxes_object_list = implement_T_3dbox_object_list(convert_6DOF_to_T(T_calculated), infra_boxes_object_list)
        cal_center_point_precision = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_distance_corresponding_precision()
        cal_vertex_point_precision = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, corresponding_strategy='vertex_distance').get_distance_corresponding_precision()
        calculated_score = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_Yscore()
        calculated_score += CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, corresponding_strategy='vertex_distance').get_Yscore()
        
        found = False
        low_combine_accuracy_scene = {}
        low_combine_accuracy_scene['infra_file_name'] = infra_file_name
        low_combine_accuracy_scene['vehicle_file_name'] = vehicle_file_name
        low_combine_accuracy_scene['combined_RE'] = cal_RE
        low_combine_accuracy_scene['combined_TE'] = cal_TE
        low_combine_accuracy_scene['combined_score'] = calculated_score
        low_combine_accuracy_scene['center_point_precision'] = cal_center_point_precision
        low_combine_accuracy_scene['vertex_point_precision'] = cal_vertex_point_precision
        low_combine_accuracy_scene['better_individual_matches'] = {}

        for match, score in matches_with_score_list:
            infra_box = infra_boxes_object_list[match[0]]
            vehicle_box = vehicle_boxes_object_list[match[1]]
            T_infra2vehicle = get_extrinsic_from_two_3dbox_object(infra_box, vehicle_box)

            converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_infra2vehicle, infra_boxes_object_list)
            center_point_precision = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list).get_distance_corresponding_precision()
            vertex_point_precision = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list, corresponding_strategy='vertex_distance').get_distance_corresponding_precision()

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T_infra2vehicle), convert_T_to_6DOF(T_true))

            if (cal_RE > RE and cal_TE > TE) and RE < 3 and TE < 3:
                found = True
                low_combine_accuracy_scene['better_individual_matches'][str(match)] = {'RE': RE.tolist(), 'TE': TE.tolist(), 'score': int(score), 'center_point_precision': float(center_point_precision), 'vertex_point_precision': float(vertex_point_precision)}
                
        if found:
            low_combine_accuracy_scenes_list.append(low_combine_accuracy_scene)
            save_cnt += 1

        cnt += 1

        # if cnt >= 10:
        #     break

        print(f'processed {cnt} scenes, found {save_cnt} low combine accuracy scenes')

    with open('low_combine_accuracy_scenes_list.json', 'w') as f:
        json.dump(low_combine_accuracy_scenes_list, f)


def save_all_scenes_with_common_boxes_num(filter_flag = True):
    cnt = 0

    common_boxes_scenes_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader().generate_infra_vehicle_bboxes_object_list():
        
        cnt += 1

        if filter_flag:
            infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(15)
            vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(15)

        T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        T_true_matches_with_score_list = CorrespondingDetector(T_true_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()

        # matches_with_score_list = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()
        # T_calculated = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=matches_with_score_list).get_combined_extrinsic(matches_filter_strategy='threshold_and_confidence')
        
        common_boxes_scenes_list.append({'infra_file_name': infra_file_name, 'vehicle_file_name': vehicle_file_name, 'common_boxes_num': len(T_true_matches_with_score_list)})
        print(f'processed {cnt} scenes')

    with open(f'common_boxes_num_scenes_list.json', 'w') as f:
        json.dump(common_boxes_scenes_list, f)

       
def save_scenes_with_more_common_boxes(common_boxes_threshold = 4, filter_flag = True):
    cnt = 0
    save_cnt = {}

    more_common_boxes_scenes_list = {}
    for threshold in range(1, common_boxes_threshold + 1):
        more_common_boxes_scenes_list[threshold] = []
        save_cnt[threshold] = 0

    with open('common_boxes_num_scenes_list.json', 'r') as f:
        common_boxes_num_scenes_list = json.load(f)
        
    # with open('low_combine_accuracy_scenes_list.json', 'r') as f:
    #     low_combine_accuracy_scenes_list = json.load(f)

    with open('error_multi_calib_scene_list.json', 'r') as f:
        error_multi_calib_scene_list = json.load(f)

    error_list_scene_index = [scene['infra_file_name'] for scene in error_multi_calib_scene_list]

    not_in_error_list = []
    
    for scene in common_boxes_num_scenes_list:
        if scene['infra_file_name'] not in error_list_scene_index:
            not_in_error_list.append(scene)

    print(f'got {len(not_in_error_list)} scenes not in error list')

    for scene in not_in_error_list:
        for threshold in range(1, common_boxes_threshold + 1):
            if scene['common_boxes_num'] >= threshold:
                more_common_boxes_scenes_list[threshold].append(scene)
                save_cnt[threshold] += 1
        print(f'got {save_cnt[threshold]} scenes with more than {threshold} common boxes')

    for threshold in range(1, common_boxes_threshold + 1):
        with open(f'more_common_boxes_scenes_list_{threshold}.json', 'w') as f:
            json.dump(more_common_boxes_scenes_list[threshold], f)

        print(f'saved {save_cnt[threshold]} scenes with more than {threshold} common boxes')

    
    


def plot_triple_line_graph(x, y1, y2, y3, x_label, y1_label, y2_label, y3_label,
                           y1_color='b', y2_color='g', y3_color='r',
                           y1_marker='o', y2_marker='^', y3_marker='s',
                           y1_linewidth=2, y2_linewidth=2, y3_linewidth=2,
                           y1_markersize=6, y2_markersize=6, y3_markersize=6,
                           y1_tick_color='b', y2_tick_color='g', y3_tick_color='r',
                           y1_ylabel=None, y2_ylabel=None, y3_ylabel=None,
                           y1_yticks=None, y2_yticks=None, y3_yticks=None,
                           title=None):
    # 创建图像和一个轴
    fig, ax1 = plt.subplots()

    # 绘制y1
    ax1.plot(x, y1, color=y1_color, marker=y1_marker, linestyle='-', label=y1_label, linewidth=y1_linewidth, markersize=y1_markersize)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y1_ylabel if y1_ylabel else y1_label, color=y1_color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=y1_tick_color)
    if y1_yticks:
        ax1.set_yticks(y1_yticks)

    # 创建第二个y轴
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('outward', 0))
    ax2.plot(x, y2, color=y2_color, marker=y2_marker, linestyle='-', label=y2_label, linewidth=y2_linewidth, markersize=y2_markersize)
    ax2.set_ylabel(y2_ylabel if y2_ylabel else y2_label, color=y2_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=y2_tick_color)
    if y2_yticks:
        ax2.set_yticks(y2_yticks)

    # 创建第三个y轴
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x, y3, color=y3_color, marker=y3_marker, linestyle='-', label=y3_label, linewidth=y3_linewidth, markersize=y3_markersize)
    ax3.set_ylabel(y3_ylabel if y3_ylabel else y3_label, color=y3_color, fontsize=12)
    ax3.tick_params(axis='y', labelcolor=y3_tick_color)
    if y3_yticks:
        ax3.set_yticks(y3_yticks)

    # 添加网格、图例和标题
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    if title:
        plt.title(title, fontsize=14)

    # 简化图形外观
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax3.spines['top'].set_visible(False)

    # 调整布局以改善显示
    fig.tight_layout()

    # 显示图表
    plt.show()



def plot_Matches2Extrinsics_filter_threshold(sample_num = 50, threshold_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], plot_type = 'line'):

    cnt = 0

    result = {}

    result['RE'] = {}
    result['TE'] = {}
    result['success_rate'] = {}

    for threshold in threshold_list:
        result['RE'][threshold] = 0
        result['TE'][threshold] = 0
        result['success_rate'][threshold] = 0

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader(path_data_info='/home/massimo/vehicle_infrastructure_calibration/dataset_division/common_boxes_4_data_info.json').generate_infra_vehicle_bboxes_object_list():
        
        infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(15)
        vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(15)

        matches_with_score_list = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()

        if len(matches_with_score_list) == 0:
            continue

        for threshold in threshold_list:
            T_calculated_6DOF = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=matches_with_score_list, threshold=threshold).get_combined_extrinsic(matches_filter_strategy='threshold')
            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated_6DOF, convert_T_to_6DOF(T_true))
            
            if RE < 1 and TE < 1:
                result['success_rate'][threshold] += 1
                result['RE'][threshold] += RE
                result['TE'][threshold] += TE

        cnt += 1

        if cnt >= sample_num:
            break

    for threshold in threshold_list:
        result['RE'][threshold] /= result['success_rate'][threshold]
        result['TE'][threshold] /= result['success_rate'][threshold]
        result['success_rate'][threshold] /= sample_num

    if plot_type == 'line':
        plot_triple_line_graph(threshold_list, [result['RE'][threshold] for threshold in threshold_list], [result['TE'][threshold] for threshold in threshold_list], [result['success_rate'][threshold] for threshold in threshold_list], 'Threshold', 'RE', 'TE', 'Success Rate')
    elif plot_type == 'bar':
        pass



plot_Matches2Extrinsics_filter_threshold()#threshold_list=[3, 3.5, 3.75, 4, 4.5, 5])

# save_low_combine_accuracy_scenes_list()

# save_all_scenes_with_common_boxes_num()

# save_scenes_with_more_common_boxes(common_boxes_threshold=4)