import numpy as np
from pathlib import Path
from ..reader import CooperativeReader
from ..utils import get_lwh_from_bbox3d_8_3
from scipy.optimize import linear_sum_assignment
from . import similarity_utils
import time

class BoxesMatch():

    def __init__(
        self,
        infra_boxes_object_list,
        vehicle_boxes_object_list,
        similarity_strategy=['core', 'category'],
        true_matches=[],
        distance_threshold=3,
        core_similarity_component=['centerpoint_distance', 'vertex_distance'],
        matches_filter_strategy='thresholdRetained',
        filter_threshold=0,
        svd_starategy='svd_with_match',
        parallel_flag=False,
        time_veerbose=False,
        corresponding_parallel=False,
        descriptor_weight=0.0,
        descriptor_metric='cosine',
        seed_top_k=0,
        resolve_180_ambiguity: bool = False,
        confidence_weight_exponent: float = 0.0,
        confidence_weight_min: float = 0.0,
        size_similarity_weight: float = 0.0,
        size_similarity_min: float = 0.0,
        confidence_boost_weight: float = 0.0,
        size_similarity_boost_weight: float = 0.0,
    ):
        '''
        BoxesMatch is a class to obtain corresponding pairs between two sets of bounding boxes without any prior extrinsics.
        param:
                similarity_strategy: ['core', 'length', 'angle', 'size', 'appearance', 'category']
                core_similarity_component:  'iou' 'centerpoint_distance' 'vertex_distance'
        '''
        self.infra_boxes_object_list, self.vehicle_boxes_object_list = infra_boxes_object_list, vehicle_boxes_object_list
        
        infra_node_num, vehicle_node_num = len(self.infra_boxes_object_list), len(self.vehicle_boxes_object_list)
        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)
        # self.norm_KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)

        self.similarity_strategy = similarity_strategy
        self.core_similarity_component = core_similarity_component
        self.matches_filter_strategy = matches_filter_strategy
        self.true_matches = true_matches
        self.distance_threshold = distance_threshold
        self.svd_starategy = svd_starategy
        self.parallel_flag = parallel_flag
        self.corresponding_parallel = corresponding_parallel
        self.descriptor_weight = descriptor_weight
        self.descriptor_metric = descriptor_metric
        self.seed_top_k = int(seed_top_k) if seed_top_k else 0
        self.resolve_180_ambiguity = bool(resolve_180_ambiguity)
        self.confidence_weight_exponent = float(confidence_weight_exponent or 0.0)
        self.confidence_weight_min = float(confidence_weight_min or 0.0)
        self.size_similarity_weight = float(size_similarity_weight or 0.0)
        self.size_similarity_min = float(size_similarity_min or 0.0)
        self.confidence_boost_weight = float(confidence_boost_weight or 0.0)
        self.size_similarity_boost_weight = float(size_similarity_boost_weight or 0.0)

        self.result_matches = []
        self.total_matches = []

        self.time_veerbose = time_veerbose
        if time_veerbose:
            start_time = time.time()

        self.cal_KP()

        if time_veerbose:
            end_time = time.time()
            print(f"Time taken for cal_KP: {end_time - start_time:.2f} seconds")

        self.matches = self.get_matched_boxes_Hungarian_matching()
        self.matches_score_list = self.get_matches_score_list()

        if len(self.matches_score_list) == 0:
            self.filter_threshold = 0
        elif filter_threshold == 0:
            self.filter_threshold = 5 if self.matches_score_list[0][1] >= 5 else self.matches_score_list[0][1]
        else:
            self.filter_threshold = filter_threshold if self.matches_score_list[0][1] > 5 else self.matches_score_list[0][1]

        self.filtered_matches = self.filter_wrong_matches()
        self.matches_score_list = [(match, score) for match, score in self.matches_score_list if match in self.filtered_matches]

    def _get_confidence(self, box):
        if box is None:
            return 1.0
        if hasattr(box, 'get_confidence'):
            try:
                return float(box.get_confidence())
            except Exception:
                pass
        try:
            return float(getattr(box, 'confidence', 1.0))
        except Exception:
            return 1.0

    def _compute_size_similarity_matrix(self):
        num_infra = len(self.infra_boxes_object_list)
        num_vehicle = len(self.vehicle_boxes_object_list)
        if num_infra == 0 or num_vehicle == 0:
            return None
        lwh_infra = np.zeros((num_infra, 3), dtype=np.float32)
        lwh_vehicle = np.zeros((num_vehicle, 3), dtype=np.float32)
        for idx, box in enumerate(self.infra_boxes_object_list):
            try:
                lwh_infra[idx] = np.asarray(get_lwh_from_bbox3d_8_3(box.get_bbox3d_8_3()), dtype=np.float32)
            except Exception:
                lwh_infra[idx] = 0.0
        for idx, box in enumerate(self.vehicle_boxes_object_list):
            try:
                lwh_vehicle[idx] = np.asarray(get_lwh_from_bbox3d_8_3(box.get_bbox3d_8_3()), dtype=np.float32)
            except Exception:
                lwh_vehicle[idx] = 0.0
        min_lwh = np.minimum(lwh_infra[:, None, :], lwh_vehicle[None, :, :])
        max_lwh = np.maximum(lwh_infra[:, None, :], lwh_vehicle[None, :, :])
        ratio = np.divide(min_lwh, np.maximum(max_lwh, 1e-6), out=np.zeros_like(min_lwh), where=max_lwh > 0)
        size_sim = ratio[..., 0] * ratio[..., 1] * ratio[..., 2]
        return size_sim.astype(np.float32, copy=False)

    def _compute_pair_weights(self, *, include_confidence: bool, include_size: bool):
        if not include_confidence and not include_size:
            return None
        num_infra = len(self.infra_boxes_object_list)
        num_vehicle = len(self.vehicle_boxes_object_list)
        if num_infra == 0 or num_vehicle == 0:
            return None
        weights = np.ones((num_infra, num_vehicle), dtype=np.float32)
        if include_confidence and self.confidence_weight_exponent > 0.0:
            infra_conf = np.array([max(0.0, self._get_confidence(b)) for b in self.infra_boxes_object_list], dtype=np.float32)
            veh_conf = np.array([max(0.0, self._get_confidence(b)) for b in self.vehicle_boxes_object_list], dtype=np.float32)
            prod = infra_conf[:, None] * veh_conf[None, :]
            if self.confidence_weight_min > 0.0:
                prod = np.maximum(prod, self.confidence_weight_min)
            weights *= np.power(prod, self.confidence_weight_exponent)
        if include_size and (self.size_similarity_weight > 0.0 or self.size_similarity_min > 0.0):
            size_sim = self._compute_size_similarity_matrix()
            if size_sim is not None:
                if self.size_similarity_min > 0.0:
                    size_sim = np.maximum(size_sim, self.size_similarity_min)
                w = float(self.size_similarity_weight or 0.0)
                if w > 0.0:
                    w = max(0.0, min(w, 1.0))
                    size_factor = (1.0 - w) + w * size_sim
                else:
                    size_factor = np.ones_like(size_sim, dtype=np.float32)
                weights *= size_factor
        if np.allclose(weights, 1.0):
            return None
        return weights

    def _compute_pair_boost(self):
        if self.confidence_boost_weight <= 0.0 and self.size_similarity_boost_weight <= 0.0:
            return None
        num_infra = len(self.infra_boxes_object_list)
        num_vehicle = len(self.vehicle_boxes_object_list)
        if num_infra == 0 or num_vehicle == 0:
            return None
        boost = np.zeros((num_infra, num_vehicle), dtype=np.float32)
        if self.confidence_boost_weight > 0.0:
            infra_conf = np.array([max(0.0, self._get_confidence(b)) for b in self.infra_boxes_object_list], dtype=np.float32)
            veh_conf = np.array([max(0.0, self._get_confidence(b)) for b in self.vehicle_boxes_object_list], dtype=np.float32)
            conf_prod = infra_conf[:, None] * veh_conf[None, :]
            boost += float(self.confidence_boost_weight) * conf_prod
        if self.size_similarity_boost_weight > 0.0:
            size_sim = self._compute_size_similarity_matrix()
            if size_sim is not None:
                boost += float(self.size_similarity_boost_weight) * size_sim
        if np.allclose(boost, 0.0):
            return None
        return boost

    def cal_KP(self):
        infra_node_num = len(self.infra_boxes_object_list)
        vehicle_node_num = len(self.vehicle_boxes_object_list)
        seed_limit = self.seed_top_k
        if seed_limit > 0:
            infra_indices = range(min(seed_limit, infra_node_num))
            vehicle_indices = range(min(seed_limit, vehicle_node_num))
        else:
            infra_indices = range(infra_node_num)
            vehicle_indices = range(vehicle_node_num)
        if 'core' in self.similarity_strategy:
            use_iou = self.core_similarity_component == 'iou' or 'iou' in self.core_similarity_component
            pair_weights = self._compute_pair_weights(
                include_confidence=bool(self.confidence_weight_exponent > 0.0 and not use_iou),
                include_size=bool(self.size_similarity_weight > 0.0 or self.size_similarity_min > 0.0),
            )
            if use_iou:
                KP, max_matches_num = similarity_utils.cal_core_KP_IoU_fast(
                    self.infra_boxes_object_list,
                    self.vehicle_boxes_object_list,
                    category_flag=('category' in self.similarity_strategy),
                    svd_starategy=self.svd_starategy,
                    infra_indices=infra_indices,
                    vehicle_indices=vehicle_indices,
                )
                if pair_weights is not None:
                    KP = KP * pair_weights
                self.KP += KP
            else:
                use_centerpoint = 'centerpoint_distance' in self.core_similarity_component
                use_vertex = 'vertex_distance' in self.core_similarity_component
                centerpoint_max_matches_num, vertexpoint_max_matches_num = -1, -1
                fast_centerpoint = fast_vertex = False
                max_nodes = max(infra_node_num, vehicle_node_num)
                use_fast_path = self.parallel_flag == 0 and (
                    not self.corresponding_parallel or max_nodes <= 25
                )

                if use_fast_path and (use_centerpoint or use_vertex):
                    KP_center, KP_vertex, fast_max_matches = similarity_utils.cal_core_KP_distance_fast_components(
                        self.infra_boxes_object_list,
                        self.vehicle_boxes_object_list,
                        use_centerpoint=use_centerpoint,
                        use_vertex=use_vertex,
                        category_flag=('category' in self.similarity_strategy),
                        distance_threshold=self.distance_threshold,
                        svd_starategy=self.svd_starategy,
                        resolve_180_ambiguity=self.resolve_180_ambiguity,
                        infra_indices=infra_indices,
                        vehicle_indices=vehicle_indices,
                    )
                    if use_centerpoint and KP_center is not None:
                        if pair_weights is not None:
                            KP_center = KP_center * pair_weights
                        self.KP += KP_center
                        centerpoint_max_matches_num = fast_max_matches
                        fast_centerpoint = True
                    if use_vertex and KP_vertex is not None:
                        if pair_weights is not None:
                            KP_vertex = KP_vertex * pair_weights
                        self.KP += KP_vertex
                        vertexpoint_max_matches_num = fast_max_matches
                        fast_vertex = True
                        if fast_centerpoint:
                            if pair_weights is None:
                                self.KP = np.round(self.KP / 2)
                            else:
                                self.KP = self.KP / 2.0

                if use_centerpoint and not fast_centerpoint:
                    if self.parallel_flag == 1:
                        KP_centerpoint, centerpoint_max_matches_num = similarity_utils.cal_core_KP_distance_parallel_refactored(
                            self.infra_boxes_object_list,
                            self.vehicle_boxes_object_list,
                            core_similarity_component='centerpoint_distance',
                            category_flag=('category' in self.similarity_strategy),
                            distance_threshold=self.distance_threshold,
                            parallel=self.corresponding_parallel,
                        )
                    elif self.parallel_flag == 0:
                        KP_centerpoint, centerpoint_max_matches_num = similarity_utils.cal_core_KP_distance(
                            self.infra_boxes_object_list,
                            self.vehicle_boxes_object_list,
                            core_similarity_component='centerpoint_distance',
                            category_flag=('category' in self.similarity_strategy),
                            distance_threshold=self.distance_threshold,
                            svd_starategy=self.svd_starategy,
                            parallel=self.corresponding_parallel,
                        )
                    else:
                        raise ValueError('parallel_flag should be 0 or 1')
                    if pair_weights is not None:
                        KP_centerpoint = KP_centerpoint * pair_weights
                    self.KP += KP_centerpoint

                if use_vertex and not fast_vertex:
                    if self.parallel_flag == 1:
                        KP_vertexpoint, vertexpoint_max_matches_num = similarity_utils.cal_core_KP_distance_parallel_refactored(
                            self.infra_boxes_object_list,
                            self.vehicle_boxes_object_list,
                            core_similarity_component='vertex_distance',
                            category_flag=('category' in self.similarity_strategy),
                            distance_threshold=self.distance_threshold,
                            parallel=self.corresponding_parallel,
                        )
                    elif self.parallel_flag == 0:
                        KP_vertexpoint, vertexpoint_max_matches_num = similarity_utils.cal_core_KP_distance(
                            self.infra_boxes_object_list,
                            self.vehicle_boxes_object_list,
                            core_similarity_component='vertex_distance',
                            category_flag=('category' in self.similarity_strategy),
                            distance_threshold=self.distance_threshold,
                            svd_starategy=self.svd_starategy,
                            parallel=self.corresponding_parallel,
                        )
                    else:
                        raise ValueError('parallel_flag should be 0 or 1')
                    if pair_weights is not None:
                        KP_vertexpoint = KP_vertexpoint * pair_weights
                    self.KP += KP_vertexpoint
                    if use_centerpoint:
                        if pair_weights is None:
                            self.KP = np.round(self.KP / 2)
                        else:
                            self.KP = self.KP / 2.0

                max_matches_num = max(centerpoint_max_matches_num, vertexpoint_max_matches_num)
        else:
            max_matches_num = -1

        pair_boost = self._compute_pair_boost()
        if pair_boost is not None:
            self.KP += pair_boost

        # print(self.KP)

        if 'length' in self.similarity_strategy:
            self.KP += similarity_utils.cal_other_edge_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy), similarity_strategy='length')

        if 'angle' in self.similarity_strategy:
            self.KP += similarity_utils.cal_other_edge_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy), similarity_strategy='angle')

        if 'size' in self.similarity_strategy:
            self.KP += similarity_utils.cal_other_vertex_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy), similarity_strategy='size')

        if 'descriptor' in self.similarity_strategy and self.descriptor_weight > 0:
            self.KP += similarity_utils.cal_descriptor_similarity(
                self.infra_boxes_object_list,
                self.vehicle_boxes_object_list,
                weight=self.descriptor_weight,
                metric=self.descriptor_metric,
            )

        # if 'appearance' in similarity_strategy:
        #     if 0 < max_matches_num < 2:
        #         self.KP += similarity_utils.cal_appearance_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, image_list=image_list)

    # def get_matched_boxes_Hungarian_matching(self):
    #     non_zero_rows = np.any(self.KP, axis=1)
    #     non_zero_columns = np.any(self.KP, axis=0)
    #     reduced_KP = self.KP[non_zero_rows][:, non_zero_columns]

    #     row_ind, col_ind = linear_sum_assignment(reduced_KP, maximize=True)
    #     original_row_ind = np.where(non_zero_rows)[0][row_ind]
    #     original_col_ind = np.where(non_zero_columns)[0][col_ind]
    #     matches = list(zip(original_row_ind, original_col_ind))
    #     return matches
    
    def get_matched_boxes_Hungarian_matching(self):
        row_ind, col_ind = linear_sum_assignment(self.KP, maximize=True)
        matches = list(zip(row_ind, col_ind))
        # matches = np.column_stack((row_ind, col_ind))
        return matches

    def filter_wrong_matches(self):
        if self.matches_filter_strategy == 'trueRetained':
            adequete_matches = [match[0] for match in self.matches_score_list if match[0] in self.true_matches]
        elif self.matches_filter_strategy == 'thresholdRetained':
            adequete_matches = [match[0] for match in self.matches_score_list if match[1] >= self.filter_threshold]
        # elif self.matches_filter_strategy == 'threshold_and_confidence':
        #     adequete_matches = [match[0] for match in self.matches_score_list if match[1] >= self.filter_threshold and self.infra_boxes_object_list[match[0][0]].get_confidence() >= 0.5 and self.vehicle_boxes_object_list[match[0][1]].get_confidence() >= 0.5]
        elif self.matches_filter_strategy == 'topRetained':
            if len(self.matches_score_list) == 0:
                adequete_matches = []
            else:
                score = self.matches_score_list[0][1]
                adequete_matches = [self.matches_score_list[0][0]]
        elif self.matches_filter_strategy == 'allRetained':
            adequete_matches = [match[0] for match in self.matches_score_list]
        else:
            raise ValueError('matches_filter_strategy should be trueRetained, thresholdRetained, topRetained or allRetained')
        
        return adequete_matches


    def get_matches_score_list(self):
        matches_score_dict = {}
        for match in self.matches:
            if self.KP[match[0], match[1]] != 0:
                matches_score_dict[match] = self.KP[match[0], match[1]]
        return sorted(matches_score_dict.items(), key=lambda x: x[1], reverse=True)

    def get_KP(self):
        return self.KP

    def get_matches(self):
        return self.matches

    def get_matches_with_score(self):
        return self.matches_score_list

    def get_stability(self):
        matches_score_dict = {}
        for match in self.matches:
            if self.KP[match[0], match[1]] != 0:
                matches_score_dict[match] = self.KP[match[0], match[1]]
        sorted_matches_score_dict = sorted(matches_score_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches_score_dict[0][1]
