import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.append('./reader')
sys.path.append('./process/search')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
from CooperativeReader import CooperativeReader
from Filter3dBoxes import Filter3dBoxes
from PSO_deconstructX import PSO_deconstructX
from CorrespondingDetector import CorrespondingDetector
from read_utils import read_yaml
from extrinsic_utils import get_extrinsic_from_two_3dbox_object, convert_T_to_6DOF, implement_T_3dbox_object_list, convert_6DOF_to_T, get_RE_TE_by_compare_T_6DOF_result_true
from BoxesMatch import BoxesMatch



class PSO_Executor():
    def __init__(self, infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF = None, matches = [], filter_num = 15, boundary_delta = [2, 2, 2, 1, 1, 1], verbose = False, visualize = False, turn_off_pso = False) -> None:
        
        self.read_basic_parameters()

        self.infra_boxes_object_list, self.vehicle_boxes_object_list = infra_boxes_object_list, vehicle_boxes_object_list
        self.true_T_6DOF_format = true_T_6DOF

        self.matches = matches

        if filter_num > 0:
            self.infra_boxes_object_list = Filter3dBoxes(self.infra_boxes_object_list).filter_according_to_size_topK(filter_num)
            self.vehicle_boxes_object_list = Filter3dBoxes(self.vehicle_boxes_object_list).filter_according_to_size_topK(filter_num)
        
        # ub: [-45, 3, 3, 1, 1, 3.1415926]
        # lb: [-50, -3, -3, -1, -1, -3.1415926]
        # self.pso_lower_bound = [-55, -10, -10, -math.pi, -math.pi, -math.pi]
        # self.pso_upper_bound = [-40, 10, 10, math.pi, math.pi, math.pi]
        self.get_pso_init_X()
        self.pso_w = (0.8, 0.8)

        # self.pso_lower_bound = [-100, -100, -10, -math.pi, -math.pi, -math.pi]
        # self.pso_upper_bound = [100, 100, 10, math.pi, math.pi, math.pi]
        # self.pso_init_X = np.eye(4)
        # self.pso_w = (0.8, 1.2)

        # self.cal_pso_bound()
        
        if verbose:
            if self.true_T_6DOF_format is not None:
                print('true_T_6DOF: ', self.true_T_6DOF_format)
            print('len(pso_init_X): ', len(self.pso_init_X))

        best_RE, best_TE = np.inf, np.inf

        for init_X in self.pso_init_X:
            
            ub = init_X + boundary_delta
            lb = init_X - boundary_delta
            if verbose:
                print('- pso_init_x: ', init_X)
                if self.true_T_6DOF_format is not None:
                    print('delta_T_6DOF: ', init_X - self.true_T_6DOF_format)
                corresponding_detector = CorrespondingDetector(implement_T_3dbox_object_list(convert_6DOF_to_T(init_X), self.infra_boxes_object_list), self.vehicle_boxes_object_list)
                corresponding_IoU_dict = corresponding_detector.corresponding_IoU_dict
                y_score = corresponding_detector.get_Yscore()
                print('- corresponding_IoU_dict: ', corresponding_IoU_dict)
                print('- y_score: ', y_score)

                print('- ub: ', ub)
                print('- lb: ', lb)

            pso_best_x = init_X

            if turn_off_pso:
                pso_best_y = CorrespondingDetector(implement_T_3dbox_object_list(convert_6DOF_to_T(init_X), self.infra_boxes_object_list), self.vehicle_boxes_object_list).get_Yscore()
            else:
                pso = PSO_deconstructX(self.infra_boxes_object_list, self.vehicle_boxes_object_list, self.true_T_6DOF_format, np.array([init_X]), 
                                        pop=self.population, max_iter=self.max_iter, ub=ub, lb=lb, w=self.pso_w, verbose=verbose)
                pso_best_x, pso_best_y = pso.run()

            if pso_best_y > self.pso_best_y:
                self.pso_best_x = pso_best_x
                self.pso_best_y = pso_best_y
                if verbose:
                    print('===update pso_best_x===: ', self.pso_best_x)
                    print('===update pso_best_y===: ', self.pso_best_y)

            if self.true_T_6DOF_format is not None:
                cur_RE, cur_TE =  get_RE_TE_by_compare_T_6DOF_result_true(pso_best_x, self.true_T_6DOF_format)
                best_RE, best_TE = min(best_RE, cur_RE), min(best_TE, cur_TE)
                if verbose:
                    print('cur RE, TE: ', cur_RE, cur_TE)
                    print('best RE, TE: ', best_RE, best_TE)

            if verbose:
                print('--------------------------')
        
        if verbose:
            print('======pso_best_x======: ', self.pso_best_x)
            print('======pso_best_y======: ', self.pso_best_y)
            if self.true_T_6DOF_format is not None:
                print('======pso result RE, TE======: ', *get_RE_TE_by_compare_T_6DOF_result_true(self.pso_best_x, self.true_T_6DOF_format))
            print('======best RE, TE======: ', best_RE, best_TE)

        
    def read_basic_parameters(self):
        config = read_yaml('./config.yml')
        self.population = config['pso']['population']
        self.max_iter = config['pso']['max_iter']
        self.pso_lower_bound = config['pso']['lb']
        self.pso_upper_bound = config['pso']['ub']

    # to test
    def cal_pso_bound(self):
        if self.pso_init_X.shape[0] == 0:
            return
        self.pso_lower_bound = np.min(self.pso_init_X, axis=0)
        self.pso_upper_bound = np.max(self.pso_init_X, axis=0)

    # to complete
    def get_pso_init_X(self):
        pso_init_X = []
        pso_y_score = []
        for match in self.matches:
            infra_box_object = self.infra_boxes_object_list[match[0]]
            vehicle_box_object = self.vehicle_boxes_object_list[match[1]]
            extrinsic = get_extrinsic_from_two_3dbox_object(infra_box_object, vehicle_box_object)
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(extrinsic, self.infra_boxes_object_list)

            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, self.vehicle_boxes_object_list)
            if len(corresponding_detector.corresponding_IoU_dict) > 0 and corresponding_detector.get_Yscore() > 0:
                pso_init_X.append(convert_T_to_6DOF(extrinsic))
                pso_y_score.append(corresponding_detector.get_Yscore())

        # pso_init_X.append(np.mean(np.array(pso_init_X), axis=0)) if len(pso_init_X) > 0 else None
        
        if len(pso_init_X) > 0:
            pso_y_score = np.array(pso_y_score)
            pso_init_X = np.array(pso_init_X)
            sorted_indices = np.argsort(pso_y_score)[::-1]
            self.pso_best_x = pso_init_X[sorted_indices[0]]
            self.pso_best_y = pso_y_score[sorted_indices[0]]

            sorted_y_scores = pso_y_score[sorted_indices]
            max_y_score = np.max(pso_y_score)
            filtered_indices = np.where(sorted_y_scores >= max(max_y_score * 0.8, max_y_score - 0.1))[0]
            self.pso_init_X = pso_init_X[sorted_indices]
            self.pso_init_X = self.pso_init_X[filtered_indices]
        else:
            self.pso_init_X = np.array([])
            self.pso_best_x = np.array([])
            self.pso_best_y = -1


    def get_best_T_6DOF(self):
        return self.pso_best_x




if __name__ == "__main__":
    
    # cooperative_reader = CooperativeReader('008663', '002505')
    cooperative_reader = CooperativeReader('009229', '005007')
    infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
    true_T_6DOF_format = convert_T_to_6DOF(cooperative_reader.get_cooperative_T_i2v())

    k = 15 

    infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
    vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)

    boxes_match = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF_format, verbose=True)
    matches = boxes_match.get_matches()
    pso_task = PSO_Executor(infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF_format, matches, verbose=True, visualize=False)
    
