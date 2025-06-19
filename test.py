import numpy as np
import os
import time
import sys
import argparse
import json
import open3d as o3d
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from v2x_calib.reader import CooperativeBatchingReader
from v2x_calib.reader import CooperativeReader
from v2x_calib.preprocess import Filter3dBoxes
from v2x_calib.reader import V2XSim_Reader, V2XSim_detected_Reader
from v2x_calib.corresponding import CorrespondingDetector
from v2x_calib.corresponding import BoxesMatch
from v2x_calib.search import Matches2Extrinsics
from v2x_calib.utils import implement_T_3dbox_object_list, implement_T_points_n_3, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF, convert_6DOF_to_T, get_extrinsic_from_two_3dbox_object
from config.config import cfg, cfg_from_yaml_file, Logger
from visualize import BBoxVisualizer_open3d
from visualize import BBoxVisualizer_open3d_standardized

# def convert_py_bbox_to_cpp(bbox):
#     bbox_8_3 = np.array(bbox.get_bbox3d_8_3(), dtype=np.float32)
#     return BoxesMatch_cpp.BoxObject(bbox_8_3, bbox.get_bbox_type())

# def get_matches_with_score_cpp(infra_boxes, vehicle_boxes):
#     infra_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in infra_boxes]
#     vehicle_boxes_cpp = [convert_py_bbox_to_cpp(bbox) for bbox in vehicle_boxes]
#     return BoxesMatch_cpp.get_matches_with_score(infra_boxes_cpp, vehicle_boxes_cpp)


def batching_test_extrisic_from_two_box_object_list(cfg):
    
    # saved_record_name = f"{cfg.data.type}_{cfg.v2x_calib.filter_num}_{cfg.v2x_calib.similarity_strategy}_{cfg.v2x_calib.core_similarity_component_list}_{cfg.v2x_calib.matches_filter_strategy}"
    saved_record_name = f"{cfg.data.type}_{cfg.v2x_calib.filter_num}_{cfg.v2x_calib.matches_filter_strategy}"
    if cfg.v2x_calib.matches_filter_strategy == 'threshold':
        saved_record_name += f"_{cfg.v2x_calib.filter_threshold}"
    saved_record_name += f"_{cfg.v2x_calib.matches2extrinsic_strategies}_{cfg.data.data_info_path.split('/')[-1].split('.')[0]}"
    saved_record_name += f"_{cfg.v2x_calib.svd_strategy}"
    if cfg.v2x_calib.parallel_flag == 1:
        saved_record_name += f"_parallel"
    if cfg.v2x_calib.correspoding_parallel_flag == 1:
        saved_record_name += f"_correspodingparallel"
    logger = Logger(saved_record_name)

    if cfg.data.type == 'V2X-Sim':
        wrapper = V2XSim_Reader().generate_vehicle_vehicle_bboxes_object_list(noise=cfg.data.noise)
    elif cfg.data.type == 'V2X-Sim_detected':
        wrapper = V2XSim_detected_Reader().generate_vehicle_vehicle_bboxes_object_list(noise=cfg.data.noise)
    elif cfg.data.type == 'DAIR-V2X':
        wrapper = CooperativeBatchingReader(path_data_info = cfg.data.data_info_path, path_data_folder= cfg.data.data_root_path).generate_infra_vehicle_bboxes_object_list()
    elif cfg.data.type == 'DAIR-V2X_detected':
        wrapper = CooperativeBatchingReader(path_data_info = cfg.data.data_info_path, path_data_folder= cfg.data.data_root_path).generate_infra_vehicle_bboxes_object_list_predicted()
    for id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true in wrapper:

        logger.info(get_processed_solo_result(cfg, id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true)[0])



def get_processed_solo_result(cfg, id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true, time_veerbose = False):

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    
    converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
    filtered_available_matches = CorrespondingDetector(
        converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold, parallel=cfg.v2x_calib.correspoding_parallel_flag).get_matches()
    
    converted_original_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
    total_available_matches_cnt = CorrespondingDetector(
        converted_original_infra_boxes_object_list, vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold, parallel=cfg.v2x_calib.correspoding_parallel_flag).get_matched_num()
    # print(sorted(CorrespondingDetector(converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold).get_matches_with_score().items(), key=lambda x: x[1], reverse=True))
    ##################
    start_time = time.time()

    
    boxes_match = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, similarity_strategy = cfg.v2x_calib.similarity_strategy, distance_threshold=cfg.v2x_calib.distance_threshold, true_matches=filtered_available_matches,
        core_similarity_component=cfg.v2x_calib.core_similarity_component_list, matches_filter_strategy=cfg.v2x_calib.matches_filter_strategy, filter_threshold=cfg.v2x_calib.filter_threshold, svd_starategy=cfg.v2x_calib.svd_strategy, parallel_flag=cfg.v2x_calib.parallel_flag, time_veerbose=time_veerbose, corresponding_parallel=cfg.v2x_calib.correspoding_parallel_flag)

    if time_veerbose:
        end_time = time.time()
        print(f"Time taken for filtering: {end_time - start_time} seconds")
        start_time = time.time()

    matches_with_score_list = boxes_match.get_matches_with_score()
    # print(f"matches_with_score_list: {matches_with_score_list}")
    matches = [match[0] for match in matches_with_score_list]

    if len(matches_with_score_list) > 0:
        T_6DOF_result = Matches2Extrinsics(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, matches_score_list=matches_with_score_list, svd_strategy=cfg.v2x_calib.svd_strategy) \
                            .get_combined_extrinsic(matches2extrinsic_strategies=cfg.v2x_calib.matches2extrinsic_strategies)
        stability = boxes_match.get_stability()

    else:
        T_6DOF_result, stability = [0, 0, 0, 0, 0, 0], 0

    
    if time_veerbose:
        end_time = time.time()
        print(f"Time taken for filtering: {end_time - start_time} seconds")
        start_time = time.time()

    end_time = time.time()
    ##################

    T_6DOF_true = convert_T_to_6DOF(T_true)
    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF_result, T_6DOF_true)

    result_matches = []

    for match in matches:
        if match in filtered_available_matches:
            result_matches.append(match)

    # print(f"result_matches: {result_matches}")
    # print(f"wrong_result_matches: {[match for match in matches if match not in result_matches]}")

    result_matched_cnt, filtered_available_matches_cnt, wrong_result_matches_cnt =  len(result_matches), len(filtered_available_matches), len(matches) - len(result_matches)

    if cfg.data.type == 'V2X-Sim' or cfg.data.type == 'V2X-Sim_detected':
            id_str = f"frame_id: {id1}, cav_id: {id2}"
    elif cfg.data.type == 'DAIR-V2X':
        id_str = f"inf_id: {id1}, veh_id: {id2}"
    elif cfg.data.type == 'DAIR-V2X_detected':
        id_str = f"inf_id: {id1}, veh_id: {id2}"

    basic_result_str = f", RE: {RE}, TE: {TE}, stability: {stability}, time: {end_time - start_time}  "
    detailed_result_str = f" ==details==> infra_total_box_cnt : {len(infra_boxes_object_list)}, vehicle_total_box_cnt: {len(vehicle_boxes_object_list)}, filtered_available_matches_cnt: {filtered_available_matches_cnt}, result_matched_cnt: {result_matched_cnt}, wrong_result_matches_cnt: {wrong_result_matches_cnt}"

    return id_str + basic_result_str + detailed_result_str, T_6DOF_result


def get_pointcloud(path_pointcloud):
    if not os.path.exists(path_pointcloud):
        raise FileNotFoundError(f'path_pointcloud: {path_pointcloud} does not exist')
    if path_pointcloud.endswith('.bin'):
        points = np.fromfile(path_pointcloud, dtype=np.float32).reshape(-1, 4)
        return points[:, :3]
    elif path_pointcloud.endswith('.pcd'):
        pointpillar = o3d.io.read_point_cloud(path_pointcloud)
        points = np.asarray(pointpillar.points)
        return points

def test_solo_with_dataset(cgf, inf_id, veh_id, time_veerbose = False):
# DAIR-V2X
    reader = CooperativeReader(infra_file_name=inf_id, vehicle_file_name=veh_id, data_folder=cfg.data.data_root_path)
    infra_boxes_object_list, vehicle_boxes_object_list = reader.get_cooperative_infra_vehicle_boxes_object_list()
    T_true = reader.get_cooperative_T_i2v()
    
    result_str, T_6DOF_result = get_processed_solo_result(cfg, inf_id, veh_id, infra_boxes_object_list, vehicle_boxes_object_list, T_true, time_veerbose = time_veerbose)
    print(result_str)

    infra_pointcloud, vehicle_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    
    T_result = convert_6DOF_to_T(T_6DOF_result)
    T_result_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_result, filtered_infra_boxes_object_list)
    T_result_converted_infra_pointcloud = implement_T_points_n_3(T_result, infra_pointcloud)

    T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
    T_true_converted_infra_pointcloud = implement_T_points_n_3(T_true, infra_pointcloud)

    # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([T_result_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_result_converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)], 'result_T')
    # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_true_converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)], 'true T')
    
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_result_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_result_converted_infra_pointcloud, vehicle_pointcloud], {}, vis_id=0)
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [T_true_converted_infra_pointcloud, vehicle_pointcloud], {}, vis_id=0)

    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_result_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [], {}, vis_id=0)
    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [], {}, vis_id=0)


def visualize_specific_match_T(cfg, inf_id, veh_id, match):
    reader = CooperativeReader(infra_file_name=inf_id, vehicle_file_name=veh_id, data_folder=cfg.data.data_root_path)
    infra_boxes_object_list, vehicle_boxes_object_list = reader.get_cooperative_infra_vehicle_boxes_object_list()
    T_true = reader.get_cooperative_T_i2v()
    
    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list) \
        .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
    
    T = get_extrinsic_from_two_3dbox_object(filtered_infra_boxes_object_list[match[0]], filtered_vehicle_boxes_object_list[match[1]])
    T_6DOF = convert_T_to_6DOF(T)
    RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(T_6DOF, convert_T_to_6DOF(T_true))

    infra_pointcloud, vehicle_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()
    T_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, filtered_infra_boxes_object_list)
    T_converted_infra_pointcloud = implement_T_points_n_3(T, infra_pointcloud)

    correspondingDetector = CorrespondingDetector(T_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold, core_similarity_component='overall_distance', parallel=cfg.v2x_calib.correspoding_parallel_flag)
    
    if correspondingDetector.get_Yscore() > 0:
        print(f"RE: {RE}, TE: {TE}", f"Y: {correspondingDetector.get_Yscore()}", f"match: {match}, match_score: {correspondingDetector.get_matches_with_score()[match]}")
        print(correspondingDetector.get_matches_with_score())
        print("-------------------")

    T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)
    T_true_converted_infra_pointcloud = implement_T_points_n_3(T_true, infra_pointcloud)

    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([T_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [], {}, vis_id=0)


def extract_flipped_scenerios(cfg):
    wrapper = CooperativeBatchingReader(path_data_info = cfg.data.data_info_path, path_data_folder= cfg.data.data_root_path).generate_infra_vehicle_bboxes_object_list()
    
    selected_pairs = []
    
    for id1, id2, infra_boxes_object_list, vehicle_boxes_object_list, T_true in wrapper:
        # filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list) \
        #     .filter_according_to_size_topK(cfg.v2x_calib.filter_num)
        # filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list) \
        #     .filter_according_to_size_topK(cfg.v2x_calib.filter_num)

        filtered_infra_boxes_object_list = infra_boxes_object_list
        filtered_vehicle_boxes_object_list = vehicle_boxes_object_list

        T_true_converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, filtered_infra_boxes_object_list)

        correspondingDetector = CorrespondingDetector(T_true_converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list, distance_threshold=cfg.v2x_calib.distance_threshold, core_similarity_component='overall_distance', parallel=cfg.v2x_calib.correspoding_parallel_flag)

        co_greater = 2

        if correspondingDetector.get_Yscore() > co_greater:
            selected_pairs.append((id1, id2))

    with open(f"Log/selected_pairs_co_greater_{co_greater}.json", 'w', encoding='utf-8') as f:
        json.dump(selected_pairs, f, indent=4)

    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/config.yaml')
    parser.add_argument('--test_type', type=str, default='single', help='single or batch')
    parser.add_argument('--inf_id', type=str, default='001366')
    parser.add_argument('--veh_id', type=str, default='017314')
    
    parser.add_argument('--filter_num', type=int, default=-1)
    parser.add_argument('--matches_filter_strategy', type=str, default='', help='thresholdRetained or topRetained or trueRetained or allRetained')
    parser.add_argument('--matches2extrinsic_strategies', type=str, default='', help='weightedSVD or evenSVD')
    parser.add_argument('--svd_strategy', type=str, default='', help='svd_with_match or svd_without_match')
    parser.add_argument('--parallel_flag', type=int, default=-1, help='True or False')
    parser.add_argument('--correspoding_parallel_flag', type=int, default=-1, help='True or False, default is True')

    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)

    if args.filter_num != -1:
        cfg.v2x_calib.filter_num = args.filter_num
    if args.matches_filter_strategy != '':
        cfg.v2x_calib.matches_filter_strategy = args.matches_filter_strategy
    if args.matches2extrinsic_strategies != '':
        cfg.v2x_calib.matches2extrinsic_strategies = args.matches2extrinsic_strategies
    if args.svd_strategy != '':
        cfg.v2x_calib.svd_strategy = args.svd_strategy
    if args.parallel_flag != -1:
        cfg.v2x_calib.parallel_flag = args.parallel_flag
    if args.correspoding_parallel_flag != -1:
        cfg.v2x_calib.correspoding_parallel_flag = args.correspoding_parallel_flag

    # if args.test_type == 'single':
    #     test_solo_with_dataset(cfg, args.inf_id, args.veh_id, time_veerbose = False)
    # elif args.test_type == 'batch':
    #     batching_test_extrisic_from_two_box_object_list(cfg = cfg)
    # else:
    #     print('Invalid test_type, please choose single or batch')

    extract_flipped_scenerios(cfg)


    # for i in range(1, 10):
    #     for j in range(1, 10):
            
    #         visualize_specific_match_T(cfg, '006746', '000066', (i, j))

    # visualize_specific_match_T(cfg, '017893', '009328', (9, 3))
    
    # test_solo_with_dataset(cfg, '006746', '000066')