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
from extrinsic_utils import get_extrinsic_from_two_3dbox_object, convert_T_to_6DOF, implement_T_3dbox_object_list




class PSO_Executor():
    def __init__(self, infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF = None, matches = [], filter_num = 15, verbose = False, visualize = False) -> None:
        
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

        self.cal_pso_bound()
        
        if verbose:
            print('len(pso_init_X): ', len(self.pso_init_X))
            print('pso_init_X: ', self.pso_init_X)
            print('pso_lower_bound: ', self.pso_lower_bound)
            print('pso_upper_bound: ', self.pso_upper_bound)


        self.pso = PSO_deconstructX(self.infra_boxes_object_list, self.vehicle_boxes_object_list, self.true_T_6DOF_format, self.pso_init_X, 
                                    pop=self.population, max_iter=self.max_iter, ub=self.pso_upper_bound, lb=self.pso_lower_bound, w=self.pso_w, verbose=verbose)

        self.pso.run()

        if visualize:
            plt.plot(self.pso.gbest_y_hist)
            plt.show()


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


    def get_pso_init_X(self):
        pso_init_X = []        
        for match in self.matches:
            infra_box_object = self.infra_boxes_object_list[match[0]]
            vehicle_box_object = self.vehicle_boxes_object_list[match[1]]
            extrinsic = get_extrinsic_from_two_3dbox_object(infra_box_object, vehicle_box_object)
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(extrinsic, self.infra_boxes_object_list)

            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, self.vehicle_boxes_object_list)
            if len(corresponding_detector.corresponding_IoU_dict) > 0 and corresponding_detector.get_Yscore() > 0.05:
                pso_init_X.append(convert_T_to_6DOF(extrinsic))
            

        pso_init_X.append(np.mean(np.array(pso_init_X), axis=0)) if len(pso_init_X) > 0 else None
        self.pso_init_X = np.array(pso_init_X)

        if self.pso_init_X.shape[0] == 0:
            # ??
            return
        elif self.pso_init_X.shape[0] == 1:
            # ??
            return 

    def get_best_T_6DOF(self):
        return self.pso.best_x


if __name__ == "__main__":
    
    # cooperative_reader = CooperativeReader('008663', '002505')
    cooperative_reader = CooperativeReader()
    infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
    true_T_6DOF_format = convert_T_to_6DOF(cooperative_reader.get_cooperative_T_i2v())

    pso_task = PSO_Executor(infra_boxes_object_list, vehicle_boxes_object_list, true_T_6DOF_format, filter_num = 15, verbose = True, visualize = True)
