import matplotlib.pyplot as plt
import sys
sys.path.append('./reader')
sys.path.append('./process/search')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
from CorrespondingDetector import CorrespondingDetector
from CooperativeBatchingReader import CooperativeBatchingReader
from Matches2Extrinsics import Matches2Extrinsics
from extrinsic_utils import convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true, implement_T_3dbox_object_list


def get_avg_result_from_moving_dynamic_object_last_frame_distance(sample_num = 100, distance_threshold_last_frame = 1, matches_filter_strategy='threshold_and_confidence'):

    RE_list = []
    TE_list = []
    success_cnt = 0

    cnt = 0

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader().generate_infra_vehicle_bboxes_object_list_static_according_last_frame(distance_threshold_between_last_frame=distance_threshold_last_frame):
        
        converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        available_matches_with_distance = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list).get_matches()
        # print('len(available_matches_with_distance): ', len(available_matches_with_distance))
        
        if len(available_matches_with_distance) == 0:
            continue

        if cnt >= sample_num:
            break
        cnt += 1
        
        matches_with_score_list = [(match, 1) for match in available_matches_with_distance]
        T_result = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=matches_with_score_list).get_combined_extrinsic(matches_filter_strategy=matches_filter_strategy)
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_result, convert_T_to_6DOF(T_true))
        if RE < 2 and TE < 2:
            RE_list.append(RE)
            TE_list.append(TE)
            success_cnt += 1

    if len(RE_list) == 0:
        RE_avg = 0
    else:
        RE_avg = sum(RE_list) / len(RE_list)
    if len(TE_list) == 0:
        TE_avg = 0
    else:
        TE_avg = sum(TE_list) / len(TE_list)

    # print('RE_list: ', RE_list)
    # print('TE_list: ', TE_list)

    return RE_avg, TE_avg, success_cnt/sample_num 


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


def plot_result_varing_distance():

    # distance_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    # distance_list = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2]
    distance_list = [10]

    RE_avg_list = []
    TE_avg_list = []

    for distance in distance_list:
        RE_avg, TE_avg, success_rate = get_avg_result_from_moving_dynamic_object_last_frame_distance(distance_threshold_last_frame=distance)
        RE_avg_list.append(RE_avg)
        TE_avg_list.append(TE_avg)
        print(f'distance: {distance}, RE_avg: {RE_avg}, TE_avg: {TE_avg}, success_rate: {success_rate}')

    plot_dual_line_graph(distance_list, RE_avg_list, TE_avg_list, 'Distance Threshold', 'RE', 'TE', y1_color='b', y2_color='r', y1_marker='o', y2_marker='s', y1_linewidth=2, y2_linewidth=2, y1_markersize=6, y2_markersize=6, y1_tick_color='b', y2_tick_color='r', y1_ylabel='RE', y2_ylabel='TE', y1_yticks=[0, 0.5, 1, 1.5, 2], y2_yticks=[0, 0.5, 1, 1.5, 2], title='RE and TE Varing Distance Threshold')


if __name__ == '__main__':
    plot_result_varing_distance()
