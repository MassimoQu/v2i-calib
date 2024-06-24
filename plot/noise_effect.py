import os
import matplotlib.pyplot as plt
import json
import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./process/search')
from test_V2XSim import V2XSim_Reader
from eval_utils import CalibEvaluator
from BoxesMatch import BoxesMatch
from Matches2Extrinsics import Matches2Extrinsics
from extrinsic_utils import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true


def get_test_noised_V2XSim_result_from_file(k = 15, totol_sample_num = 100, noise = {'pos_std':1.6, 'rot_std':0, 'pos_mean':0, 'rot_mean':0},folder_name = r'new_clean_result/extrinsic_noise_gaussian_category_core_vertex_distance_svd8point_threshold/all_dataset'):
    
    if not os.path.exists(folder_name):
        raise ValueError(f'{folder_name} not exist')

    folder_name = os.path.join(folder_name, f'_pos_std_{noise["pos_std"]}_rot_std_{noise["rot_std"]}_pos_mean_{noise["pos_mean"]}_rot_mean_{noise["rot_mean"]}')
    # folder_name = os.path.join(folder_name, f'_pos_std_{format(Decimal(noise["pos_std"]), 'g')}_rot_std_{format(Decimal(noise["rot_std"]), 'g')}_pos_mean_{format(Decimal(noise["pos_mean"], 'g'))}_rot_mean_{format(Decimal(noise["rot_mean"]), 'g')}')

    TE_list = []
    RE_list = []

    for file_name in os.listdir(folder_name):
        with open(os.path.join(folder_name, file_name), 'r') as f:
            example_list = json.load(f)
            if file_name[0] == 'v':
                for example in example_list:
                    TE_list.append(example['TE'])
                    RE_list.append(example['RE'])

    return TE_list, RE_list


def get_test_noised_V2XSim_result(k = 15, totol_sample_num = 100, noise = {'pos_std':1.6, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}, folder_name = r'new_clean_result/extrinsic_noise_gaussian_category_core_vertex_distance_svd8point_threshold/all_dataset'):
        
        TE_list = []
        RE_list = []
    
        corresponding_strategy = ['centerpoint_distance','vertex_distance']
        matches_filter_strategy = 'threshold'

        for frame_idx, cav_id, bbox3d_object_list_lidar1, bbox3d_object_list_lida2, T_lidar2_lidar1 in V2XSim_Reader().generate_vehicle_vehicle_bboxes_object_list(noise=noise):
            
            matches_with_score_list = BoxesMatch(bbox3d_object_list_lidar1, bbox3d_object_list_lida2, corresponding_strategy=corresponding_strategy).get_matches_with_score()
            T_calculated = Matches2Extrinsics(bbox3d_object_list_lidar1, bbox3d_object_list_lida2, matches_score_list = matches_with_score_list).get_combined_extrinsic(matches_filter_strategy = matches_filter_strategy)
    
            T_6DOF_true = convert_T_to_6DOF(T_lidar2_lidar1)
            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated, T_6DOF_true)

            TE_list.append(TE)
            RE_list.append(RE)

        return TE_list, RE_list


def plot_dual_line_graph(x, y1, y2, x_label, y1_label, y2_label, y1_color = 'b', y2_color = 'r', y1_marker = 'o', y2_marker = 's', y1_linewidth = 2, y2_linewidth = 2, y1_markersize = 6, y2_markersize = 6, y1_tick_color = 'b', y2_tick_color = 'r', y1_ylabel = None, y2_ylabel = None, y1_yticks = None, y2_yticks = None, title = None):
    # Create a new figure and axis
    fig, ax1 = plt.subplots()

    # Plot RE_avg_list with improved style
    ax1.plot(x, y1, f'{y1_color}-{y1_marker}', label=y1_label, linewidth=y1_linewidth, markersize=y1_markersize)
    ax1.set_xlabel(x_label, fontsize=12)
    ax1.set_ylabel(y1_ylabel if y1_ylabel else y1_label, color=y1_color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=y1_tick_color)
    if y1_yticks:
        ax1.set_yticks(y1_yticks)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(x, y2, f'{y2_color}-{y2_marker}', label=y2_label, linewidth=y2_linewidth, markersize=y2_markersize)
    ax2.set_ylabel(y2_ylabel if y2_ylabel else y2_label, color=y2_color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=y2_tick_color)
    if y2_yticks:
        ax2.set_yticks(y2_yticks)

    # Align the tick marks of both y-axes
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.yaxis.set_major_locator(plt.MaxNLocator(5))

    # Add grid, legend, and title
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))
    plt.title(title, fontsize=14)

    # Remove the top and right spines for a cleaner look
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    # Adjust layout for better spacing
    fig.tight_layout()

    # Show the plot
    plt.show()

def plot_triple_line_graph(x, y1, y2, y3, x_label, y1_label, y2_label, y3_label, y1_color='b', y2_color='r', y3_color='g', y1_marker='o', y2_marker='s', y3_marker='^', y1_linewidth=2, y2_linewidth=2, y3_linewidth=2, y1_markersize=6, y2_markersize=6, y3_markersize=6, title=None):
    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Plot the first line
    ax.plot(x, y1, f'{y1_color}-{y1_marker}', label=y1_label, linewidth=y1_linewidth, markersize=y1_markersize)
    # Plot the second line
    ax.plot(x, y2, f'{y2_color}-{y2_marker}', label=y2_label, linewidth=y2_linewidth, markersize=y2_markersize)
    # Plot the third line
    ax.plot(x, y3, f'{y3_color}-{y3_marker}', label=y3_label, linewidth=y3_linewidth, markersize=y3_markersize)

    # Set the labels for x-axis and y-axis
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('mAP', color='black', fontsize=12)  # Unified y-label for mAP

    # Set the tick parameters for the y-axis
    ax.tick_params(axis='y', labelcolor='black')

    # Set the legend to identify the lines
    ax.legend(loc='upper right', bbox_to_anchor=(0.85, 0.85))

    # Add grid for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set the title if provided
    if title:
        plt.title(title, fontsize=14)

    # Remove the top and right spines for a cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Adjust layout to avoid overlap and cutoff
    fig.tight_layout()

    # Show the plot
    plt.show()


def plot_line_graph_for_TE_with_varying_pos_std(std_min = 0, std_max = 2, std_num = 6, folder_name = r'new_clean_result/extrinsic_noise_gaussian_category_core_vertex_distance_svd8point_threshold/all_dataset'):
    
    RE_avg_list = []
    success_rate_list = []

    # pos_noise_std_list = [0, 0.4, 0.8, 1.2, 1.6, 2]
    pos_noise_std_list = [0, 1, 2, 3, 4, 5, 6]

    # for pos_std in np.linspace(std_min, std_max, std_num):
    for pos_std in pos_noise_std_list:
        noise = {'pos_std':pos_std, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}
        TE_list, _ = get_test_noised_V2XSim_result(noise = noise, folder_name = folder_name)
        if len(TE_list) == 0:
            TE_list.append(0)
        RE_avg_list.append(sum(TE_list) / len(TE_list))
        success_rate_list.append(sum(1 for TE in TE_list if TE < 3))
        
    plot_dual_line_graph(pos_noise_std_list, RE_avg_list, success_rate_list, 'Position Noise Std Dev (m)', 'TE_avg', 'Success Rate (%)', y1_color = 'b', y2_color = 'r', y1_marker = 'o', y2_marker = 's', y1_linewidth = 2, y2_linewidth = 2, y1_markersize = 6, y2_markersize = 6, y1_tick_color = 'b', y2_tick_color = 'r', y1_ylabel = 'TE_avg', y2_ylabel = 'Success Rate (%)', y1_yticks = None, y2_yticks = None, title = 'Position Noise Effect')

def plot_line_graph_for_RE_with_varying_rot_std(std_min = 0, std_max = 2, std_num = 6, folder_name = r'new_clean_result/extrinsic_noise_gaussian_category_core_vertex_distance_svd8point_threshold/all_dataset'):
    
    RE_avg_list = []
    success_rate_list = []

    # rot_noise_std_list = [0, 1, 2, 3, 4, 5, 6]
    rot_noise_std_list = [0, 5, 10, 15, 20, 25]

    # for pos_std in np.linspace(std_min, std_max, std_num):
    for rot_std in rot_noise_std_list:
        noise = {'pos_std':0, 'rot_std':rot_std, 'pos_mean':0, 'rot_mean':0}
        _, RE_list = get_test_noised_V2XSim_result(noise = noise, folder_name = folder_name)
        if len(RE_list) == 0:
            RE_list.append(0)
        RE_avg_list.append(sum(RE_list) / len(RE_list))
        success_rate_list.append(sum(1 for RE in RE_list if RE < 10))
        
    plot_dual_line_graph(rot_noise_std_list, RE_avg_list, success_rate_list, 'Rotation Noise Std Dev (°)', 'RE_avg', 'Success Rate (%)', y1_color = 'b', y2_color = 'r', y1_marker = 'o', y2_marker = 's', y1_linewidth = 2, y2_linewidth = 2, y1_markersize = 6, y2_markersize = 6, y1_tick_color = 'b', y2_tick_color = 'r', y1_ylabel = 'RE_avg', y2_ylabel = 'Success Rate (%)', y1_yticks = None, y2_yticks = None, title = 'Rotation Noise Effect')

def convert_bbox3d_object_list_to_v2xsim_format(bbox3d_object_list):
    # FROM: List[box_object]
    # TO: Dict, {'boxes_3d': Array[N, 8, 3], 'labels_3d': Array[N], 'scores_3d': Array[N]}
    result_dict = {'boxes_3d':[], 'labels_3d':[], 'scores_3d':[]}
    for box_object in bbox3d_object_list:
        result_dict['boxes_3d'].append(box_object.get_bbox3d_8_3())
        result_dict['labels_3d'].append(2)
        result_dict['scores_3d'].append(1)
    return result_dict

def get_noised_V2XSim_mAP_result(k = 15, totol_sample_num = 10, noise = {'pos_std':1.6, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}):

    mAP = {'0.3':0, '0.5':0, '0.7':0}

    cnt = 0

    for frame_idx, cav_id, bbox3d_object_list_lidar, noised_bbox3d_object_list_lidar in V2XSim_Reader().generate_gt_and_noised_bboxes_object_list(noise=noise):
        
        # if cnt >= totol_sample_num:
        #     break
        cnt += 1

        # print('start evaluating frame:', frame_idx)
        evaluator = CalibEvaluator()

        evaluator.add_frame(convert_bbox3d_object_list_to_v2xsim_format(noised_bbox3d_object_list_lidar), convert_bbox3d_object_list_to_v2xsim_format(bbox3d_object_list_lidar))
        mAP_dict = evaluator.get_mAP()

        # print('mAP:', mAP_dict)

        mAP['0.3'] += mAP_dict[0.3]
        mAP['0.5'] += mAP_dict[0.5]
        mAP['0.7'] += mAP_dict[0.7]

    mAP['0.3'] /= cnt
    mAP['0.5'] /= cnt
    mAP['0.7'] /= cnt

    return mAP

def plot_line_graph_for_mAP_with_varying_pos_std(std_min = 0, std_max = 2, std_num = 6):
    
    mAP_3_list = []
    mAP_5_list = []
    mAP_7_list = []
    # pos_noise_std_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    # pos_noise_std_list = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    pos_noise_std_list = [0, 1, 2, 3, 4, 5, 6]

    # for pos_std in np.linspace(std_min, std_max, std_num):
    for pos_std in pos_noise_std_list:
        noise = {'pos_std':pos_std, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}
        # print('input noise:', noise)
        # mAP : dict {'0.3': float, '0.5': float, '0.7': float}
        mAP = get_noised_V2XSim_mAP_result(noise = noise)
        
        mAP_3_list.append(mAP['0.3'])
        mAP_5_list.append(mAP['0.5'])
        mAP_7_list.append(mAP['0.7'])

        # print('-------------------')
        
    plot_triple_line_graph(pos_noise_std_list, mAP_3_list, mAP_5_list, mAP_7_list, 'Position Noise Std Dev (m)', 'mAP@0.3', 'mAP@0.5', 'mAP@0.7', y1_color='b', y2_color='r', y3_color='g', y1_marker='o', y2_marker='s', y3_marker='^', y1_linewidth=2, y2_linewidth=2, y3_linewidth=2, y1_markersize=6, y2_markersize=6, y3_markersize=6, title='Position Noise Effect')


def plot_line_graph_for_mAP_with_varying_rot_std(std_min = 0, std_max = 2, std_num = 6):
    
    mAP_3_list = []
    mAP_5_list = []
    mAP_7_list = []
    # rot_noise_std_list = [0, 1, 2, 3, 4, 5, 6]
    rot_noise_std_list = [0, 5, 10, 15, 20, 25]

    # for pos_std in np.linspace(std_min, std_max, std_num):
    for rot_std in rot_noise_std_list:
        noise = {'pos_std':0, 'rot_std':rot_std, 'pos_mean':0, 'rot_mean':0}

        # mAP : dict {'0.3': float, '0.5': float, '0.7': float}
        mAP = get_noised_V2XSim_mAP_result(noise = noise)
        
        mAP_3_list.append(mAP['0.3'])
        mAP_5_list.append(mAP['0.5'])
        mAP_7_list.append(mAP['0.7'])

        # print('-------------------')
        
    plot_triple_line_graph(rot_noise_std_list, mAP_3_list, mAP_5_list, mAP_7_list, 'Rotation Noise Std Dev (°)', 'mAP@0.3', 'mAP@0.5', 'mAP@0.7', y1_color='b', y2_color='r', y3_color='g', y1_marker='o', y2_marker='s', y3_marker='^', y1_linewidth=2, y2_linewidth=2, y3_linewidth=2, y1_markersize=6, y2_markersize=6, y3_markersize=6, title='Rotation Noise Effect')


if __name__ == '__main__':

    # plot_line_graph_for_TE_with_varying_pos_std()

    # plot_line_graph_for_RE_with_varying_rot_std()

    # plot_line_graph_for_mAP_with_varying_pos_std()

    plot_line_graph_for_mAP_with_varying_rot_std()

