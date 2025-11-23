import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Union, List, Optional
sys.path.append(str(Path(__file__).parent.parent))
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree
import teaserpp_python
import copy
import argparse
import json
from pathlib import Path
from time import perf_counter
from configs.legacy_api import cfg, cfg_from_yaml_file, Logger
from calib.evaluation.metrics import FrameMetrics, aggregate_metrics
from v2x_calib.utils import get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF
from v2x_calib.reader.CooperativeBatchingReader import CooperativeBatchingReader
from v2x_calib import V2XSim_Reader, V2XSetReader


BeamDebug = Dict[str, Union[float, int, bool]]

infra_beam_rng = None
infra_beam_rng_seed = None


def _cfg_value(block, key, default=None):
    if block is None:
        return default
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _get_infra_beam_rng(seed):
    global infra_beam_rng, infra_beam_rng_seed
    if infra_beam_rng is None or infra_beam_rng_seed != seed:
        infra_beam_rng = np.random.default_rng(seed)
        infra_beam_rng_seed = seed
    return infra_beam_rng


def compute_vertical_angles(points: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return np.empty(0)
    xy_norm = np.linalg.norm(points[:, :2], axis=1)
    return np.degrees(np.arctan2(points[:, 2], np.maximum(xy_norm, 1e-6)))


def _filter_points_by_vehicle_angle(points: np.ndarray, veh_points: np.ndarray, beam_cfg) -> Tuple[np.ndarray, BeamDebug]:
    debug_info: BeamDebug = {'angle_filter_applied': False}
    if veh_points is None or veh_points.size == 0:
        return points, debug_info

    percentiles = _cfg_value(beam_cfg, 'angle_percentiles', (1.0, 99.0))
    if not isinstance(percentiles, (list, tuple)) or len(percentiles) != 2:
        percentiles = (1.0, 99.0)
    theta_low, theta_high = percentiles

    veh_angles = compute_vertical_angles(veh_points)
    if veh_angles.size == 0:
        return points, debug_info

    veh_theta_min = float(np.percentile(veh_angles, theta_low))
    veh_theta_max = float(np.percentile(veh_angles, theta_high))

    infra_angles = compute_vertical_angles(points)
    mask = (infra_angles >= veh_theta_min) & (infra_angles <= veh_theta_max)
    filtered = points[mask]
    min_after_angle = int(_cfg_value(beam_cfg, 'min_points_after_angle', 3000))

    debug_info.update({
        'veh_angle_min_deg': veh_theta_min,
        'veh_angle_max_deg': veh_theta_max,
        'infra_points_after_angle_filter': int(filtered.shape[0]),
    })

    if filtered.shape[0] < max(1, min_after_angle):
        return points, debug_info

    debug_info.update({
        'angle_filter_applied': True,
        'infra_angle_min_deg': float(np.min(infra_angles[mask])),
        'infra_angle_max_deg': float(np.max(infra_angles[mask])),
    })
    return filtered, debug_info


def apply_infra_beam_alignment(points: np.ndarray, veh_points: np.ndarray) -> Tuple[np.ndarray, BeamDebug]:
    beam_cfg = _cfg_value(cfg.infra, 'beam_alignment', None)
    debug_info: BeamDebug = {}
    if beam_cfg is None or not _cfg_value(beam_cfg, 'enabled', False):
        return points, debug_info
    cfg_data = _cfg_value(cfg, 'data', None)
    if _cfg_value(cfg_data, 'type', None) != "DAIR-V2X":
        return points, debug_info

    points_aligned = points
    if _cfg_value(beam_cfg, 'match_vehicle_angle', False):
        points_aligned, angle_debug = _filter_points_by_vehicle_angle(points_aligned, veh_points, beam_cfg)
        debug_info.update(angle_debug)

    ratio = _cfg_value(beam_cfg, 'subsample_ratio', None)
    if ratio is None:
        source_lines = _cfg_value(beam_cfg, 'source_lines', None)
        target_lines = _cfg_value(beam_cfg, 'target_lines', None)
        if source_lines and target_lines:
            ratio = float(target_lines) / float(source_lines)
    if ratio is None:
        return points_aligned, debug_info
    ratio = float(np.clip(ratio, 0.0, 1.0))
    if ratio <= 0.0 or ratio >= 0.999:
        return points_aligned, debug_info

    num_points = points_aligned.shape[0]
    min_points = int(_cfg_value(beam_cfg, 'min_points', 0))
    target_points = max(int(num_points * ratio), min_points)
    target_points = min(target_points, num_points)
    if target_points <= 0 or target_points == num_points:
        return points_aligned, debug_info

    rng = _get_infra_beam_rng(_cfg_value(beam_cfg, 'random_seed', None))
    sampled_indices = rng.choice(num_points, size=target_points, replace=False)
    debug_info['infra_points_after_ratio_subsample'] = int(target_points)
    return points_aligned[sampled_indices], debug_info


def _bounded_iterator(iterator, start_idx: int, end_idx: Optional[int]):
    current = 0
    for record in iterator:
        if current < start_idx:
            current += 1
            continue
        if end_idx is not None and current >= end_idx:
            break
        yield record
        current += 1


def maybe_refine_with_icp(T_init: np.ndarray, inf_pcd: o3d.geometry.PointCloud,
                          veh_pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, BeamDebug]:
    refine_cfg = _cfg_value(cfg, 'post_refine', None)
    icp_cfg = _cfg_value(refine_cfg, 'icp', None)
    debug: BeamDebug = {}
    if icp_cfg is None or not _cfg_value(icp_cfg, 'enabled', False):
        return T_init, debug

    max_corr = float(_cfg_value(icp_cfg, 'max_correspondence_distance', 1.0))
    max_iter = int(_cfg_value(icp_cfg, 'max_iterations', 50))
    method = _cfg_value(icp_cfg, 'method', 'point_to_plane')

    estimation: o3d.pipelines.registration.TransformationEstimation = \
        o3d.pipelines.registration.TransformationEstimationPointToPlane() \
        if method == 'point_to_plane' else \
        o3d.pipelines.registration.TransformationEstimationPointToPoint()

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    icp_result = o3d.pipelines.registration.registration_icp(
        source=copy.deepcopy(inf_pcd),
        target=veh_pcd,
        max_correspondence_distance=max_corr,
        init=T_init,
        estimation_method=estimation,
        criteria=criteria
    )
    T_refined = icp_result.transformation
    debug.update({
        'icp_refined': True,
        'icp_fitness': float(icp_result.fitness),
        'icp_inlier_rmse': float(icp_result.inlier_rmse),
    })
    return T_refined, debug

def create_point_cloud(points, color=[0, 0.651, 0.929]):
    # 1.000, 0.706, 0.000
    xyz = points[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(color)
    return pcd

def pcd2xyz(pcd):
    return np.asarray(pcd.points).T

def extract_fpfh(pcd, radius_normal, radius_feature):
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return np.array(fpfh.data).T

def find_knn_cpu(feat0, feat1, knn=1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=knn)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds

def find_correspondences(feats0, feats1, mutual_filter=True):
    nns01 = find_knn_cpu(feats0, feats1, knn=1, return_distance=False)
    corres01_idx0 = np.arange(len(nns01))
    corres01_idx1 = nns01

    if not mutual_filter:
        return corres01_idx0, corres01_idx1

    nns10 = find_knn_cpu(feats1, feats0, knn=1, return_distance=False)
    corres10_idx1 = np.arange(len(nns10))
    corres10_idx0 = nns10

    mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
    corres_idx0 = corres01_idx0[mutual_filter]
    corres_idx1 = corres01_idx1[mutual_filter]

    return corres_idx0, corres_idx1

def Rt2T(R, t):
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def get_teaser_solver(noise_bound, rotation_estimation_algorithm=teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1.0
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.inlier_selection_mode = \
        teaserpp_python.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
    solver_params.rotation_tim_graph = \
        teaserpp_python.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
    solver_params.rotation_estimation_algorithm = rotation_estimation_algorithm
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 10000
    solver_params.rotation_cost_threshold = 1e-16
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    return solver

def fpfh_teaser(inf_pc: np.ndarray, veh_pc: np.ndarray,
                infra_params=None, veh_params=None, visualize=False):
    infra_points_raw = int(inf_pc.shape[0])
    veh_points_raw = int(veh_pc.shape[0])

    inf_pc, beam_debug = apply_infra_beam_alignment(inf_pc, veh_pc)
    infra_points_beam_aligned = int(inf_pc.shape[0])

    inf_pcd = create_point_cloud(inf_pc, color=[0, 0.651, 0.929])
    veh_pcd = create_point_cloud(veh_pc, color=[1.000, 0.706, 0.000])

    infra_fpfh = infra_params or cfg.infra.fpfh
    veh_fpfh = veh_params or cfg.vehicle.fpfh

    inf_voxel = _cfg_value(infra_fpfh, 'voxel_size', cfg.infra.fpfh.voxel_size)
    veh_voxel = _cfg_value(veh_fpfh, 'voxel_size', cfg.vehicle.fpfh.voxel_size)
    inf_pcd = inf_pcd.voxel_down_sample(voxel_size=inf_voxel)
    veh_pcd = veh_pcd.voxel_down_sample(voxel_size=veh_voxel)

    A_xyz = pcd2xyz(inf_pcd)  # np array of size 3 by N
    B_xyz = pcd2xyz(veh_pcd)  # np array of size 3 by M

    # extract FPFH features
    A_feats = extract_fpfh(
        inf_pcd,
        _cfg_value(infra_fpfh, 'radius_normal', cfg.infra.fpfh.radius_normal),
        _cfg_value(infra_fpfh, 'radius_feature', cfg.infra.fpfh.radius_feature))
    B_feats = extract_fpfh(
        veh_pcd,
        _cfg_value(veh_fpfh, 'radius_normal', cfg.vehicle.fpfh.radius_normal),
        _cfg_value(veh_fpfh, 'radius_feature', cfg.vehicle.fpfh.radius_feature))

    # establish correspondences by nearest neighbour search in feature space
    corrs_A, corrs_B = find_correspondences(A_feats, B_feats, mutual_filter=True)
    A_corr = A_xyz[:, corrs_A]  # np array of size 3 by num_corrs
    B_corr = B_xyz[:, corrs_B]  # np array of size 3 by num_corrs

    if cfg.rotation_estimation_algorithm == "GNC_TLS":
        rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    elif cfg.rotation_estimation_algorithm == "FGR":
        rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.FGR
    elif cfg.rotation_estimation_algorithm == "QUATRO":
        rotation_estimation_algorithm = teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.QUATRO

    teaser_solver = get_teaser_solver(cfg.teaser.noise_bound, rotation_estimation_algorithm)
    teaser_solver.solve(A_corr, B_corr)
    solution = teaser_solver.getSolution()
    R_teaser = solution.rotation
    t_teaser = solution.translation
    T_teaser = Rt2T(R_teaser, t_teaser)

    if visualize:
        A_pcd_T_teaser = copy.deepcopy(inf_pcd).transform(T_teaser)
        o3d.visualization.draw_geometries([A_pcd_T_teaser, veh_pcd], window_name="Registration by Teaser++")

    stats = {
        'infra_points_raw': infra_points_raw,
        'infra_points_beam_aligned': infra_points_beam_aligned,
        'infra_points_post_voxel': int(len(inf_pcd.points)),
        'vehicle_points_raw': veh_points_raw,
        'vehicle_points_post_voxel': int(len(veh_pcd.points)),
    }
    T_refined, icp_debug = maybe_refine_with_icp(T_teaser, inf_pcd, veh_pcd)
    stats.update(beam_debug)
    stats.update(icp_debug)

    return T_refined, stats


def numpy_to_point_cloud(points):
    """
    Convert a numpy array to an Open3D PointCloud object.
    
    Parameters:
        points (np.ndarray): A Nx3 numpy array of xyz positions.
    
    Returns:
        o3d.geometry.PointCloud: An Open3D point cloud object.
    """
    # 创建一个空的 PointCloud 对象
    pcd = o3d.geometry.PointCloud()

    # 将 numpy array 中的数据转换成 Open3D 所需的格式
    pcd.points = o3d.utility.Vector3dVector(points)

    return pcd


def ensure_output_dir(tag: str) -> Path:
    root = Path(cfg.ROOT_DIR) / 'outputs' / 'hkust_teaser'
    root.mkdir(parents=True, exist_ok=True)
    out_dir = root / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/hkust_lidar_global_config.yaml')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--max-pairs', type=int, default=None)
    parser.add_argument('--output-tag', type=str, default=None)
    parser.add_argument('--v2xsim-root', type=str, default=None,
                        help='Root directory of V2X-Sim pickled scenes '
                             '(overrides config/env if provided).')
    parser.add_argument('--v2xset-root', type=str, default=None,
                        help='Root directory of the cooperative-vehicle-infrastructure/v2xset tree.')
    parser.add_argument('--v2xset-split', type=str, default=None,
                        help='Optional split override for V2X-Set (train/validate/test).')
    parser.add_argument('--rotation-alg', type=str, choices=['GNC_TLS', 'FGR', 'QUATRO'],
                        help='Override rotation_estimation_algorithm from the config.')
    args = parser.parse_args()
    cfg_from_yaml_file(os.path.join(cfg.ROOT_DIR, args.config), cfg)
    if args.v2xsim_root:
        cfg.data.v2xsim_root = args.v2xsim_root
    if args.v2xset_root:
        cfg.data.v2xset_root = args.v2xset_root
    if args.v2xset_split:
        cfg.data.v2xset_split = args.v2xset_split
    if args.rotation_alg:
        cfg.rotation_estimation_algorithm = args.rotation_alg

    tag_base = args.output_tag or f"{cfg.data.type}_{cfg.rotation_estimation_algorithm}_{Path(cfg.data.data_info_path).stem}"
    output_dir = ensure_output_dir(tag_base)
    logger = Logger(tag_base)
    beam_cfg = _cfg_value(cfg.infra, 'beam_alignment', None)
    if beam_cfg and _cfg_value(beam_cfg, 'enabled', False) and cfg.data.type == "DAIR-V2X":
        ratio = _cfg_value(beam_cfg, 'subsample_ratio', None)
        if ratio is None:
            source_lines = _cfg_value(beam_cfg, 'source_lines', None)
            target_lines = _cfg_value(beam_cfg, 'target_lines', None)
            if source_lines and target_lines:
                ratio = float(target_lines) / float(source_lines)
        if ratio is not None:
            ratio = float(np.clip(ratio, 0.0, 1.0))
        ratio_str = f"{ratio:.3f}" if ratio is not None else "N/A"
        logger.info(
            "Infrastructure beam alignment enabled: "
            f"ratio={ratio_str}, "
            f"min_points={_cfg_value(beam_cfg, 'min_points', 'N/A')}, "
            f"seed={_cfg_value(beam_cfg, 'random_seed', 'N/A')}")
    matches_path = output_dir / 'matches.jsonl'

    end_idx = None if args.end is None or args.end < 0 else args.end

    if cfg.data.type == "V2X-Sim":
        v2xsim_root = getattr(cfg.data, 'v2xsim_root', None) or os.environ.get('V2XSIM_DATA_ROOT')
        reader = V2XSim_Reader(root_dir=v2xsim_root) if v2xsim_root else V2XSim_Reader()
        source_iter = reader.generate_vehicle_vehicle_pointcloud()
        wrapper = _bounded_iterator(source_iter, args.start, end_idx)
    elif cfg.data.type == "V2X-Set":
        v2xset_root = getattr(cfg.data, 'v2xset_root', None) or os.environ.get('V2XSET_DATA_ROOT')
        v2xset_split = getattr(cfg.data, 'v2xset_split', 'validate')
        v2xset_stride = getattr(cfg.data, 'v2xset_frame_stride', 1)
        v2xset_cavs = getattr(cfg.data, 'v2xset_max_cavs', 4)
        reader = V2XSetReader(
            root_dir=v2xset_root or "/mnt/ssd_gw/cooperative-vehicle-infrastructure/v2xset",
            split=v2xset_split,
            max_cavs=v2xset_cavs,
            frame_stride=v2xset_stride,
        )
        wrapper = reader.generate_vehicle_vehicle_pointcloud(start_idx=args.start, end_idx=end_idx)
    elif cfg.data.type == "DAIR-V2X":
        wrapper = CooperativeBatchingReader(
            path_data_info=cfg.data.data_info_path,
            path_data_folder=cfg.data.data_root_path
        ).generate_infra_vehicle_pointcloud(start_idx=args.start, end_idx=args.end)

    modes_cfg = getattr(cfg, 'registration_modes', None)
    if modes_cfg:
        mode_entries = modes_cfg
    else:
        mode_entries = [{'name': 'default'}]

    def build_params(default_block, override):
        if override is None:
            return default_block
        merged = {}
        merged['voxel_size'] = _cfg_value(override, 'voxel_size', _cfg_value(default_block, 'voxel_size', None))
        merged['radius_normal'] = _cfg_value(override, 'radius_normal', _cfg_value(default_block, 'radius_normal', None))
        merged['radius_feature'] = _cfg_value(override, 'radius_feature', _cfg_value(default_block, 'radius_feature', None))
        return merged

    records = []
    processed = 0
    with matches_path.open('w', encoding='utf-8') as match_file:
        for inf_id, veh_id, inf_pc, veh_pc, T_true in wrapper:
            if args.max_pairs is not None and processed >= args.max_pairs:
                break
            candidates: List[Dict[str, Union[str, float, int, bool]]] = []
            best = None
            best_score = -1.0
            best_T = None
            t_start = perf_counter()
            for mode_cfg in mode_entries:
                mode_name = mode_cfg.get('name', 'mode')
                infra_override = mode_cfg.get('infra_fpfh', None)
                veh_override = mode_cfg.get('vehicle_fpfh', None)
                T_mode, stats_mode = fpfh_teaser(
                    inf_pc,
                    veh_pc,
                    infra_params=build_params(cfg.infra.fpfh, infra_override),
                    veh_params=build_params(cfg.vehicle.fpfh, veh_override),
                    visualize=False
                )
                stats_mode = dict(stats_mode)
                stats_mode['mode_name'] = mode_name
                fitness = float(stats_mode.get('icp_fitness', 0.0))
                rmse = stats_mode.get('icp_inlier_rmse', None)
                score = fitness
                if rmse is not None:
                    score = fitness - 0.5 * float(rmse)
                stats_mode['icp_score'] = score
                candidates.append(stats_mode)
                if score > best_score:
                    best_score = score
                    best = stats_mode
                    best_T = T_mode
            t_end = perf_counter()

            if best is None or best_T is None:
                continue

            RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(best_T), convert_T_to_6DOF(T_true))
            ids_str = f"inf_id: {inf_id}, veh_id: {veh_id}" if cfg.data.type == "DAIR-V2X" else f"frame_id: {inf_id}, cav_id: {veh_id}"
            eval_str = f" RE: {RE}, TE: {TE}, time: {t_end - t_start}, best_mode={best.get('mode_name', 'N/A')}, icp_fitness={best_score:.3f}"
            logger.info(ids_str + eval_str)
            match_payload = {
                'infra_id': inf_id,
                'veh_id': veh_id,
                'RE': RE,
                'TE': TE,
                'time': t_end - t_start,
                'best_mode': best.get('mode_name'),
                'icp_fitness': best_score,
                **{k: v for k, v in best.items() if k not in {'mode_name', 'icp_score'}}
            }
            match_payload['mode_candidates'] = [
                {
                    'mode_name': cand.get('mode_name'),
                    'icp_fitness': cand.get('icp_fitness'),
                    'infra_points_post_voxel': cand.get('infra_points_post_voxel'),
                    'vehicle_points_post_voxel': cand.get('vehicle_points_post_voxel')
                } for cand in candidates
            ]
            match_file.write(json.dumps({
                **match_payload
            }) + '\n')
            records.append(FrameMetrics(
                infra_id=str(inf_id),
                veh_id=str(veh_id),
                RE=float(RE),
                TE=float(TE),
                stability=0.0,
                time_cost=float(t_end - t_start),
            ))
            processed += 1

    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
    summary = aggregate_metrics(records, thresholds)
    summary_path = output_dir / 'metrics.json'
    with summary_path.open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")

        # pcd_inf_T_converted = numpy_to_point_cloud(inf_pc)
        # pcd_inf_trueT_converted = numpy_to_point_cloud(inf_pc)
        # pcd_veh = numpy_to_point_cloud(veh_pc)

        # pcd_inf_T_converted.paint_uniform_color([0, 0.651, 0.929])
        # pcd_inf_trueT_converted.paint_uniform_color([0, 0.651, 0.929])
        # pcd_veh.paint_uniform_color([1.000, 0.706, 0.000])

        # pcd_inf_T_converted.transform(T)
        # pcd_inf_trueT_converted.transform(T_true)
        # o3d.visualization.draw_geometries([pcd_inf_trueT_converted, pcd_veh], window_name=f"T_true registration ; inf_id: {inf_id}, veh_id: {veh_id}; RE: {RE}, TE: {TE}")
        # o3d.visualization.draw_geometries([pcd_inf_T_converted, pcd_veh], window_name="T_cal registration")

        # if input("Continue? (y/n)") == 'n':
        #     break
