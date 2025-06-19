import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from reader import CooperativeReader
from scipy.optimize import linear_sum_assignment
import similarity_utils as similarity_utils
import time

class BoxesMatch():

    def __init__(self,infra_boxes_object_list, vehicle_boxes_object_list, similarity_strategy = ['core', 'category'], true_matches = [], distance_threshold = 3,
                core_similarity_component = ['centerpoint_distance','vertex_distance'], matches_filter_strategy = 'thresholdRetained' , filter_threshold = 0, svd_starategy = 'svd_with_match', parallel_flag = False, time_veerbose = False, corresponding_parallel=False):
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


    def cal_KP(self):
        if 'core' in self.similarity_strategy:
            if self.core_similarity_component == 'iou' or 'iou' in self.core_similarity_component:
                # KP, norm_KP, max_matches_num = similarity_utils.cal_core_KP_IoU(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy))
                KP, max_matches_num = similarity_utils.cal_core_KP_IoU(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy))
                self.KP += KP
                # self.norm_KP += norm_KP
            else:
                centerpoint_max_matches_num, vertexpoint_max_matches_num = 0, 0
                if 'centerpoint_distance' in self.core_similarity_component:
                    # KP_centerpoint, norm_KP_centerpoint, centerpoint_max_matches_num = similarity_utils.cal_core_KP_distance(self.infra_boxes_object_list, self.vehicle_boxes_object_list, core_similarity_component='centerpoint_distance', category_flag=('category' in self.similarity_strategy), distance_threshold=self.distance_threshold)
                    if self.parallel_flag==1:
                        KP_centerpoint, centerpoint_max_matches_num = similarity_utils.cal_core_KP_distance_parallel_refactored(self.infra_boxes_object_list, self.vehicle_boxes_object_list, core_similarity_component='centerpoint_distance', category_flag=('category' in self.similarity_strategy), distance_threshold=self.distance_threshold,parallel=self.corresponding_parallel)
                    elif self.parallel_flag==0:
                        KP_centerpoint, centerpoint_max_matches_num = similarity_utils.cal_core_KP_distance(self.infra_boxes_object_list, self.vehicle_boxes_object_list, core_similarity_component='centerpoint_distance', category_flag=('category' in self.similarity_strategy), distance_threshold=self.distance_threshold, svd_starategy=self.svd_starategy,parallel=self.corresponding_parallel)
                    else:  
                        raise ValueError('parallel_flag should be 0 or 1')
                    self.KP += KP_centerpoint
                    # self.norm_KP += norm_KP_centerpoint
                if 'vertex_distance' in self.core_similarity_component:
                    # KP_vertexpoint, norm_KP_vertexpoint, vertexpoint_max_matches_num = similarity_utils.cal_core_KP_distance(self.infra_boxes_object_list, self.vehicle_boxes_object_list, core_similarity_component='vertex_distance', category_flag=('category' in self.similarity_strategy), distance_threshold=self.distance_threshold)
                    if self.parallel_flag==1:
                        KP_vertexpoint, vertexpoint_max_matches_num = similarity_utils.cal_core_KP_distance_parallel_refactored(self.infra_boxes_object_list, self.vehicle_boxes_object_list, core_similarity_component='vertex_distance', category_flag=('category' in self.similarity_strategy), distance_threshold=self.distance_threshold,parallel=self.corresponding_parallel)
                    elif self.parallel_flag==0:
                        KP_vertexpoint, vertexpoint_max_matches_num = similarity_utils.cal_core_KP_distance(self.infra_boxes_object_list, self.vehicle_boxes_object_list, core_similarity_component='vertex_distance', category_flag=('category' in self.similarity_strategy), distance_threshold=self.distance_threshold, svd_starategy=self.svd_starategy,parallel=self.corresponding_parallel)
                    else:
                        raise ValueError('parallel_flag should be 0 or 1')
                    self.KP += KP_vertexpoint
                    self.KP = np.round(self.KP / 2)
                    # self.norm_KP += norm_KP_vertexpoint
                    # self.norm_KP = np.round(self.norm_KP / 2)
                
                max_matches_num = max(centerpoint_max_matches_num, vertexpoint_max_matches_num)
        else:
            max_matches_num = -1

        # print(self.KP)

        if 'length' in self.similarity_strategy:
            self.KP += similarity_utils.cal_other_edge_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy), similarity_strategy='length')

        if 'angle' in self.similarity_strategy:
            self.KP += similarity_utils.cal_other_edge_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy), similarity_strategy='angle')

        if 'size' in self.similarity_strategy:
            self.KP += similarity_utils.cal_other_vertex_KP(self.infra_boxes_object_list, self.vehicle_boxes_object_list, category_flag=('category' in self.similarity_strategy), similarity_strategy='size')

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


