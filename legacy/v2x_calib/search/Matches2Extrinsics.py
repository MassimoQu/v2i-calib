import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from ..utils import (
    get_extrinsic_from_two_3dbox_object,
    get_extrinsic_from_two_points,
    get_extrinsic_from_two_points_svd_without_match,
    get_extrinsic_from_two_points_weighted,
    get_extrinsic_from_two_points_weighted_svd_without_match,
    implement_T_points_n_3,
    convert_T_to_6DOF,
)  # , optimize_extrinsic_from_two_mixed_3dbox_object_list

class Matches2Extrinsics:
    
    def __init__(
        self,
        infra_boxes_object_list,
        vehicle_boxes_object_list,
        true_T_6DOF=None,
        matches_score_list=None,
        verbose: bool = False,
        svd_strategy: str = 'svd_with_match',
        *,
        resolve_180_ambiguity: bool = False,
        max_iterations: int = 1,
        inlier_threshold_m: float = 0.0,
        mad_scale: float = 2.5,
        min_inliers: int = 1,
    ):
        
        self.matches_score_list = matches_score_list or []
        self.infra_boxes_object_list = infra_boxes_object_list
        self.vehicle_boxes_object_list = vehicle_boxes_object_list
        self.svd_strategy = svd_strategy
        self.resolve_180_ambiguity = bool(resolve_180_ambiguity)
        self.max_iterations = max(1, int(max_iterations or 1))
        self.inlier_threshold_m = float(inlier_threshold_m or 0.0)
        self.mad_scale = float(mad_scale or 0.0)
        self.min_inliers = max(1, int(min_inliers or 1))
        self.threshold = (
            max(self.matches_score_list[0][1] * 0.8, self.matches_score_list[0][1] - 1)
            if len(self.matches_score_list) >= 1
            else 0
        )

        self.true_T_6DOF_format = true_T_6DOF

        if verbose:
            
            if self.true_T_6DOF_format is not None:
                print('true_T_6DOF: ', true_T_6DOF)
            
            print('self.threshold: ', self.threshold)

            adequate_num = len([match[0] for match in self.matches_score_list if match[1] >= self.threshold])
            print('len(adequete_matches): ', adequate_num)

            cnt = 0

            for match, score in self.matches_score_list:
                print(cnt)
                infra_box_object = self.infra_boxes_object_list[match[0]]
                vehicle_box_object = self.vehicle_boxes_object_list[match[1]]
                extrinsic = get_extrinsic_from_two_3dbox_object(infra_box_object, vehicle_box_object)
                print('- score: ', score)
                print('- extrinsic: ', extrinsic)
                cnt += 1
                if score < self.threshold:
                    print('below threshold')

    @staticmethod
    def _box_residual(T, points1, points2) -> float:
        transformed = implement_T_points_n_3(T, points1)
        diff = transformed - points2
        return float(np.linalg.norm(diff, axis=1).mean())

    @staticmethod
    def _point_residuals(T, points1, points2):
        transformed = implement_T_points_n_3(T, points1)
        diff = transformed - points2
        return np.linalg.norm(diff, axis=1)

    def _select_corners(self, T, corners1, corners2):
        if not self.resolve_180_ambiguity:
            residual = self._box_residual(T, corners1, corners2)
            return corners1, residual
        dihedral_4 = [
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2),
            (0, 3, 2, 1),
            (3, 2, 1, 0),
            (2, 1, 0, 3),
            (1, 0, 3, 2),
        ]
        best = corners1
        best_res = self._box_residual(T, corners1, corners2)
        for p in dihedral_4[1:]:
            perm = tuple(p) + tuple(i + 4 for i in p)
            cand = corners1[list(perm)]
            res = self._box_residual(T, cand, corners2)
            if res < best_res:
                best = cand
                best_res = res
        return best, best_res

    def _estimate_T(self, corners1_list, corners2_list, weights, strategy: str):
        if not corners1_list or not corners2_list:
            return np.eye(4)
        pts1 = np.concatenate(corners1_list, axis=0)
        pts2 = np.concatenate(corners2_list, axis=0)
        if self.svd_strategy == 'svd_without_match':
            if strategy == 'weightedSVD':
                w = np.repeat(np.asarray(weights, dtype=np.float64), 8)
                return get_extrinsic_from_two_points_weighted_svd_without_match(pts1, pts2, w)
            return get_extrinsic_from_two_points_svd_without_match(pts1, pts2)
        if strategy == 'weightedSVD':
            w = np.repeat(np.asarray(weights, dtype=np.float64), 8)
            return get_extrinsic_from_two_points_weighted(pts1, pts2, w)
        return get_extrinsic_from_two_points(pts1, pts2)

    def _estimate_T_points(self, points1, points2, weights, strategy: str):
        if points1 is None or points2 is None:
            return np.eye(4)
        pts1 = np.asarray(points1, dtype=np.float64).reshape(-1, 3)
        pts2 = np.asarray(points2, dtype=np.float64).reshape(-1, 3)
        if pts1.size == 0 or pts2.size == 0:
            return np.eye(4)
        if pts1.shape != pts2.shape:
            n = min(int(pts1.shape[0]), int(pts2.shape[0]))
            if n <= 0:
                return np.eye(4)
            pts1 = pts1[:n]
            pts2 = pts2[:n]
            if weights is not None:
                weights = list(weights)[:n]
        if strategy == 'weightedSVD' and weights is not None:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size != pts1.shape[0]:
                w = np.ones((pts1.shape[0],), dtype=np.float64)
            return get_extrinsic_from_two_points_weighted(pts1, pts2, w)
        return get_extrinsic_from_two_points(pts1, pts2)

    def _robust_T(self, corners1_list, corners2_list, weights, strategy: str):
        if not corners1_list or not corners2_list:
            return np.eye(4)
        weights_arr = np.asarray(weights, dtype=np.float64) if weights is not None else None
        if self.resolve_180_ambiguity:
            centers1 = np.vstack([c.mean(axis=0) for c in corners1_list])
            centers2 = np.vstack([c.mean(axis=0) for c in corners2_list])
            if strategy == 'weightedSVD' and weights_arr is not None and len(weights_arr) == len(centers1):
                T = get_extrinsic_from_two_points_weighted(centers1, centers2, weights_arr)
            else:
                T = get_extrinsic_from_two_points(centers1, centers2)
        else:
            T = self._estimate_T(corners1_list, corners2_list, weights, strategy)
        if self.max_iterations <= 1 and not self.resolve_180_ambiguity:
            return T

        for _ in range(self.max_iterations):
            selected1 = []
            residuals = []
            for c1, c2 in zip(corners1_list, corners2_list):
                chosen, res = self._select_corners(T, c1, c2)
                selected1.append(chosen)
                residuals.append(res)

            if self.max_iterations <= 1:
                return self._estimate_T(selected1, corners2_list, weights, strategy)

            residuals_arr = np.asarray(residuals, dtype=np.float64)
            if self.inlier_threshold_m > 0:
                thr = self.inlier_threshold_m
            else:
                median = float(np.median(residuals_arr))
                mad = float(np.median(np.abs(residuals_arr - median)))
                thr = median + max(0.0, self.mad_scale) * mad
                thr = max(thr, median, 0.5)

            inlier_mask = residuals_arr <= thr
            if int(inlier_mask.sum()) < self.min_inliers:
                best = np.argsort(residuals_arr)[: self.min_inliers]
                inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
                inlier_mask[best] = True

            inlier_indices = np.nonzero(inlier_mask)[0].tolist()
            selected1_in = [selected1[i] for i in inlier_indices]
            corners2_in = [corners2_list[i] for i in inlier_indices]
            weights_in = [weights[i] for i in inlier_indices]
            T_new = self._estimate_T(selected1_in, corners2_in, weights_in, strategy)
            if np.allclose(T_new, T):
                T = T_new
                break
            T = T_new
        return T

    def _robust_T_points(self, points1, points2, weights, strategy: str):
        pts1 = np.asarray(points1, dtype=np.float64).reshape(-1, 3)
        pts2 = np.asarray(points2, dtype=np.float64).reshape(-1, 3)
        if pts1.size == 0 or pts2.size == 0:
            return np.eye(4)
        if pts1.shape != pts2.shape:
            n = min(int(pts1.shape[0]), int(pts2.shape[0]))
            if n <= 0:
                return np.eye(4)
            pts1 = pts1[:n]
            pts2 = pts2[:n]
            if weights is not None:
                weights = list(weights)[:n]

        weights_arr = np.asarray(weights, dtype=np.float64) if weights is not None else None
        T = self._estimate_T_points(pts1, pts2, weights_arr, strategy)
        if self.max_iterations <= 1:
            return T

        for _ in range(self.max_iterations):
            residuals = self._point_residuals(T, pts1, pts2)
            residuals_arr = np.asarray(residuals, dtype=np.float64)
            if self.inlier_threshold_m > 0:
                thr = float(self.inlier_threshold_m)
            else:
                median = float(np.median(residuals_arr))
                mad = float(np.median(np.abs(residuals_arr - median)))
                thr = median + max(0.0, self.mad_scale) * mad
                thr = max(thr, median, 0.5)

            inlier_mask = residuals_arr <= thr
            if int(inlier_mask.sum()) < self.min_inliers:
                best = np.argsort(residuals_arr)[: self.min_inliers]
                inlier_mask = np.zeros_like(inlier_mask, dtype=bool)
                inlier_mask[best] = True

            inlier_idx = np.nonzero(inlier_mask)[0]
            pts1_in = pts1[inlier_idx]
            pts2_in = pts2[inlier_idx]
            weights_in = None
            if weights_arr is not None and int(weights_arr.size) == int(pts1.shape[0]):
                weights_in = weights_arr[inlier_idx]
            T_new = self._estimate_T_points(pts1_in, pts2_in, weights_in, strategy)
            if np.allclose(T_new, T):
                T = T_new
                break
            T = T_new
        return T

    def get_combined_extrinsic(self, matches2extrinsic_strategies = 'weightedSVD'):

        infra_boxes_object_list = [self.infra_boxes_object_list[match[0]] for match, _ in self.matches_score_list]
        vehicle_boxes_object_list = [self.vehicle_boxes_object_list[match[1]] for match, _ in self.matches_score_list]
        weights = [float(score) for _, score in self.matches_score_list]
        corners1_list = [np.asarray(box.get_bbox3d_8_3(), dtype=np.float64) for box in infra_boxes_object_list]
        corners2_list = [np.asarray(box.get_bbox3d_8_3(), dtype=np.float64) for box in vehicle_boxes_object_list]
        
        if matches2extrinsic_strategies not in {
            'evenSVD',
            'weightedSVD',
            'centerSVD',
            'centerWeightedSVD',
            'hybridSVD',
            'hybridWeightedSVD',
        }:
            raise ValueError(
                f'matches2extrinsic_strategies={matches2extrinsic_strategies}, matches2extrinsic_strategies should be '
                'evenSVD/weightedSVD/centerSVD/centerWeightedSVD/hybridSVD/hybridWeightedSVD'
            )
        if not corners1_list or not corners2_list:
            return convert_T_to_6DOF(np.eye(4, dtype=np.float64))
        use_center = matches2extrinsic_strategies in {'centerSVD', 'centerWeightedSVD'}
        use_hybrid = matches2extrinsic_strategies in {'hybridSVD', 'hybridWeightedSVD'}
        if use_hybrid and len(corners1_list) >= 3 and len(corners2_list) >= 3:
            use_center = True

        if use_center:
            centers1 = np.vstack([c.mean(axis=0) for c in corners1_list])
            centers2 = np.vstack([c.mean(axis=0) for c in corners2_list])
            weighted = matches2extrinsic_strategies in {'centerWeightedSVD', 'hybridWeightedSVD'}
            strategy = 'weightedSVD' if weighted else 'evenSVD'
            resultT = self._robust_T_points(centers1, centers2, weights, strategy)
        else:
            weighted = matches2extrinsic_strategies in {'weightedSVD', 'hybridWeightedSVD'}
            strategy = 'weightedSVD' if weighted else 'evenSVD'
            resultT = self._robust_T(corners1_list, corners2_list, weights, strategy)

        return convert_T_to_6DOF(resultT)
    
    
