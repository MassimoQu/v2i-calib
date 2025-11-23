import json
import matplotlib.pyplot as plt
import sys
sys.path.append('./reader')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./plot')
sys.path.append('./visualize')
sys.path.append('./process/search')
from CooperativeBatchingReader import CooperativeBatchingReader
from extrinsic_utils import implement_T_3dbox_object_list, convert_T_to_6DOF, get_RE_TE_by_compare_T_6DOF_result_true, convert_6DOF_to_T
from eval_utils import CalibEvaluator
from CorrespondingDetector import CorrespondingDetector
from BBoxVisualizer_open3d_standardized import BBoxVisualizer_open3d_standardized
from Matches2Extrinsics import Matches2Extrinsics
from Filter3dBoxes import Filter3dBoxes
from BoxesMatch import BoxesMatch

def load_json(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def create_entry_key(entry):
    """Create a unique key for each entry based on infra and vehicle file names."""
    infra_file_name = entry['infrastructure_pointcloud_path'].split('/')[-1].split('.')[0]
    vehicle_file_name = entry['vehicle_pointcloud_path'].split('/')[-1].split('.')[0]
    return f"{infra_file_name}-{vehicle_file_name}"

def find_matching_entries_efficient(data_dict, group_entries):
    """Find matching entries in data.json based on easy_group.json using a more efficient method."""
    
    filtered_entries = []
    for group in group_entries:
        key = f"{group['infra_file_name']}-{group['vehicle_file_name']}"
        if key in data_dict:
            filtered_entries.append(data_dict[key])

    return filtered_entries


def divide_dair_v2x_according_to_difficulty():
    # Load data from JSON files
    data_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json')
    easy_group_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/intermediate_output/111/easy_group_k15_totalcnt6600.json')
    hard_group_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/intermediate_output/111/medium_group_k15_totalcnt6600.json')

    # Create a dictionary for quick lookup
    data_dict = {create_entry_key(entry): entry for entry in data_entries}

    # Find and filter matching entries with efficient search
    filtered_easy_entries = find_matching_entries_efficient(data_dict, easy_group_entries)
    filtered_hard_entries = find_matching_entries_efficient(data_dict, hard_group_entries)

    # Save the filtered entries to a new JSON file
    save_json(filtered_easy_entries, 'easy_data.json')
    save_json(filtered_hard_entries, 'hard_data.json')

    print("Filtered data has been saved to easy_data.json and hard_data.json")


def divide_dair_v2x_according_to_common_boxes():
    # Load data from JSON files
    data_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json')
    common_boxes_group_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/more_common_boxes_scenes_list_4.json')

    # Create a dictionary for quick lookup
    data_dict = {create_entry_key(entry): entry for entry in data_entries}

    # Find and filter matching entries with efficient search
    filtered_common_boxes_entries = find_matching_entries_efficient(data_dict, common_boxes_group_entries)

    # Save the filtered entries to a new JSON file
    save_json(filtered_common_boxes_entries, 'common_boxes_4_data_info.json')

    print("Filtered data has been saved to common_boxes_4_data_info.json")


def filter_failed_scenes(data_dict, failed_entries):

    failed_entries = []
    filtered_entries = []

    for group in failed_entries:
        key = f"{group['infra_file_name']}-{group['vehicle_file_name']}"
        if key in data_dict:
            failed_entries.append(data_dict[key])

    print(f"Failed entries: {len(failed_entries)}")

    failed_data_dict = {create_entry_key(entry): entry for entry in failed_entries}

    for key, entry in data_dict.items():
        if key not in failed_data_dict:
            filtered_entries.append(entry)

    return filtered_entries


def divide_dair_v2x_according_to_filtering_combine_failed_scenes():
    data_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/common_boxes_4_data_info.json')
    failed_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/dataset_division/low_combine_accuracy_scenes_list.json')

    # Create a dictionary for quick lookup
    data_dict = {create_entry_key(entry): entry for entry in data_entries}

    # Find and filter matching entries with efficient search
    filtered_failed_entries = filter_failed_scenes(data_dict, failed_entries)

    # Save the filtered entries to a new JSON file
    save_json(filtered_failed_entries, 'common_boxes_4_high_combine_accuracy_data_info.json')

    print("Filtered data has been saved to low_combine_accuracy_data_info.json")


############
############
###########
def divide_dair_v2x_according_to_continuity():
    # Load data from JSON files
    data_entries = load_json(f'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json')
    
    # Filter entries based on continuity
    filtered_continuity_entries = []



    # Save the filtered entries to a new JSON file
    save_json(filtered_continuity_entries, 'continuous_data_info.json')


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



def test_dair_v2x_scene_mAP(sample_num = 100):

    cnt = 0

    mAP_difficulty_dict = {}
    mAP_difficulty_dict['easy'] = {}
    mAP_difficulty_dict['medium'] = {}
    mAP_difficulty_dict['hard'] = {}
    mAP_difficulty_dict['extreme'] = {}

    mAP_all_dict = {}
    mAP_all_dict[0.3] = []
    mAP_all_dict[0.5] = []
    mAP_all_dict[0.7] = []

    for difficulty in ['easy', 'medium', 'hard', 'extreme']:
        mAP_difficulty_dict[difficulty][0.3] = []
        mAP_difficulty_dict[difficulty][0.5] = []
        mAP_difficulty_dict[difficulty][0.7] = []


    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader().generate_infra_vehicle_bboxes_object_list():
        converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)

        matches_score_dict = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()
        sorted_matches_score_list = sorted(matches_score_dict.items(), key=lambda x: x[1], reverse=True)
        T_calculated_6DOF = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=sorted_matches_score_list).get_combined_extrinsic(matches_filter_strategy='threshold_and_confidence')
        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated_6DOF, convert_T_to_6DOF(T_true))
        if len(sorted_matches_score_list) == 0 or sorted_matches_score_list[0][1] < 4:
            continue

        true_filtered_infra_boxes_object_list = []
        true_filtered_vehicle_boxes_object_list = []

        for match, score in sorted_matches_score_list:
            true_filtered_infra_boxes_object_list.append(infra_boxes_object_list[match[0]])
            true_filtered_vehicle_boxes_object_list.append(vehicle_boxes_object_list[match[1]])

        converted_true_filtered_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, true_filtered_infra_boxes_object_list)

        # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([converted_infra_boxes_object_list, vehicle_boxes_object_list], [], matches_with_score)

        evaluator = CalibEvaluator()
        evaluator.add_frame_BBox3d_format(converted_true_filtered_infra_boxes_object_list, true_filtered_vehicle_boxes_object_list)
        mAP_dict = evaluator.get_mAP()

        if RE < 1 and TE < 1:
            difficulty = 'easy'
        elif RE < 3 and TE < 3:
            difficulty = 'medium'
        elif RE < 5 and TE < 5:
            difficulty = 'hard'
        else:
            difficulty = 'extreme'

        mAP_difficulty_dict[difficulty][0.3].append((mAP_dict[0.3], RE, TE))
        mAP_difficulty_dict[difficulty][0.5].append((mAP_dict[0.5], RE, TE))
        mAP_difficulty_dict[difficulty][0.7].append((mAP_dict[0.7], RE, TE))

        mAP_all_dict[0.3].append({mAP_dict[0.3]: (RE, TE)})
        mAP_all_dict[0.5].append({mAP_dict[0.5]: (RE, TE)})
        mAP_all_dict[0.7].append({mAP_dict[0.7]: (RE, TE)})


        # if mAP_dict[0.3] != 0 or mAP_dict[0.5] != 0 or mAP_dict[0.7] != 0:

        #     print(infra_file_name, vehicle_file_name)
        #     # print(matches_with_score_list)
        #     print(f'RE: {RE} , TE: {TE}')
        #     print(mAP_dict)
        #     input('Press any key to continue...')

        # else:
        #     print(f'{cnt }: No matches found for: ', infra_file_name, vehicle_file_name)


        cnt += 1
        if cnt >= sample_num:
            break

    # for difficulty in ['easy', 'medium', 'hard', 'extreme']:
    #     for iou_threshold in [0.3, 0.5, 0.7]:
    #         mAP_list = mAP_difficulty_dict[difficulty][iou_threshold]
    #         if len(mAP_list) == 0:
    #             mAP_avg = 0
    #         else:
    #             mAP_avg = sum([mAP for mAP, RE, TE in mAP_list]) / len(mAP_list)
    #             RE_avg = sum([RE for mAP, RE, TE in mAP_list]) / len(mAP_list)
    #             TE_avg = sum([TE for mAP, RE, TE in mAP_list]) / len(mAP_list)
    #         print(f'{difficulty} mAP@{iou_threshold}: {mAP_avg}, RE_avg: {RE_avg}, TE_avg: {TE_avg}')

    mAP_list = {}
    mAP_list[0.3] = []
    mAP_list[0.5] = []
    mAP_list[0.7] = []
    RE_list = []
    TE_list = []

    for iou_threshold in [0.3, 0.5, 0.7]:
        sorted_mAP_threshold_list = sorted(mAP_all_dict[iou_threshold], key=lambda x: list(x.keys())[0], reverse=True)
        RE_list = []
        TE_list = []
        for item in sorted_mAP_threshold_list:
            mAP = list(item.keys())[0]
            RE, TE = list(item.values())[0]
            # if RE > 1 or TE > 1:
            #     continue
            mAP_list[iou_threshold].append(mAP)
            RE_list.append(RE)
            TE_list.append(TE)

    plot_dual_line_graph(mAP_list[0.3], RE_list, TE_list, 'mAP@0.3', 'RE', 'TE')
    plot_dual_line_graph(mAP_list[0.5], RE_list, TE_list, 'mAP@0.5', 'RE', 'TE')
    plot_dual_line_graph(mAP_list[0.7], RE_list, TE_list, 'mAP@0.7', 'RE', 'TE')
            
    
        

    # plot_dual_line_graph(mAP_all_dict[0.3], RE_list, TE_list, 'mAP@0.3', 'RE', 'TE')


def save_dair_v2x_error_multi_calib():

    cnt = 0
    save_cnt = 0

    error_multi_calib_scene_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_true in CooperativeBatchingReader().generate_infra_vehicle_bboxes_object_list():
        
        infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(15)
        vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(15)

        matches_with_score_list = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list).get_matches_with_score()
        T_calculated = Matches2Extrinsics(infra_boxes_object_list, vehicle_boxes_object_list, matches_score_list=matches_with_score_list).get_combined_extrinsic(matches_filter_strategy='threshold_and_confidence')

        if len(matches_with_score_list) == 0:
            continue

        cal_RE, cal_TE = get_RE_TE_by_compare_T_6DOF_result_true(T_calculated, convert_T_to_6DOF(T_true))
        
        calculated_T_converted_infra_boxes_object_list = implement_T_3dbox_object_list(convert_6DOF_to_T(T_calculated), infra_boxes_object_list)
        calculated_center_point_precision = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_distance_corresponding_precision()
        calculated_vertex_point_precision = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, core_similarity_component='vertex_distance').get_distance_corresponding_precision()
        calculated_score = CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_Yscore()
        calculated_score += CorrespondingDetector(calculated_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, core_similarity_component='vertex_distance').get_Yscore()
        
        true_T_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        true_center_point_precision = CorrespondingDetector(true_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_distance_corresponding_precision()
        true_vertex_point_precision = CorrespondingDetector(true_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, core_similarity_component='vertex_distance').get_distance_corresponding_precision()
        true_score = CorrespondingDetector(true_T_converted_infra_boxes_object_list, vehicle_boxes_object_list).get_Yscore()
        true_score += CorrespondingDetector(true_T_converted_infra_boxes_object_list, vehicle_boxes_object_list, core_similarity_component='vertex_distance').get_Yscore()

        if calculated_center_point_precision > true_center_point_precision and calculated_vertex_point_precision > true_vertex_point_precision and calculated_score > true_score:
            error_multi_calib_scene_list.append({'infra_file_name': infra_file_name, 'vehicle_file_name': vehicle_file_name, 
                                                 'cal_RE': cal_RE, 'cal_TE': cal_TE, 
                                                 'calculated_center_point_precision': calculated_center_point_precision, 
                                                 'true_center_point_precision': true_center_point_precision, 
                                                 'calculated_vertex_point_precision': calculated_vertex_point_precision, 
                                                 'true_vertex_point_precision': true_vertex_point_precision, 
                                                 'calculated_score': calculated_score, 
                                                 'true_score': true_score})
            save_cnt += 1   

        cnt += 1

        print(f'{cnt} - {save_cnt}')

        # if save_cnt >= 10:
        #     break

    with open('error_multi_calib_scene_list.json', 'w') as file:
        json.dump(error_multi_calib_scene_list, file)





if __name__ == "__main__":
    # divide_dair_v2x_according_to_difficulty()
    # test_dair_v2x_scene_mAP()
    # save_dair_v2x_error_multi_calib()
    # divide_dair_v2x_according_to_common_boxes()
    divide_dair_v2x_according_to_filtering_combine_failed_scenes()