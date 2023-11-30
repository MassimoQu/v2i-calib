import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from Task import Task
import sys
sys.path.append('./reader')
from InfraReader import InfraReader
from VehicleReader import VehicleReader
from CooperativeReader import CooperativeReader
from process.Filter3dBoxes import Filter3dBoxes
from Reader import Reader
from seach.PSO import PSO, PSO_new_type
from process.utils.IoU_utils import box3d_iou
from process.utils.extrinsic_utils import convert_Rt_to_T


class PSOTask(Task):
    def __init__(self) -> None:
        self.reader = Reader('config.yml')
        self.infra_reader = InfraReader('config.yml')
        self.vehicle_reader = VehicleReader('config.yml')

        self.infra_boxes = self.infra_reader.get_infra_boxes_dict()
        self.vehicle_boxes = self.vehicle_reader.get_vehicle_boxes_dict()
        # print(self.infra_boxes)
        # print('-------------------')
        # print(self.vehicle_boxes)
        
        # ub: [-45, 3, 3, 1, 1, 3.1415926]
        # lb: [-50, -3, -3, -1, -1, -3.1415926]
        self.pso_lower_bound = [-50, -3, -3, -1, -1, -math.pi]
        self.pso_upper_bound = [-45, 3, 3, 1, 1, math.pi]
        self.pso_init_X = self.cal_init(self.infra_boxes['Bus'][0], self.vehicle_boxes['Bus'][0])
        self.pso_w = (0.1, 0.4)

        # self.pso_lower_bound = [-100, -100, -10, -math.pi, -math.pi, -math.pi]
        # self.pso_upper_bound = [100, 100, 10, math.pi, math.pi, math.pi]
        # self.pso_init_X = np.eye(4)
        # self.pso_w = (0.8, 1.2)
        

        
        self.pso = PSO(self.objective_func, self.infra_boxes, self.vehicle_boxes, self.pso_init_X, 
                       pop=self.reader.parse_pso_pop_yaml(), max_iter=self.reader.parse_pso_max_iter_yaml(),
                       ub=self.pso_upper_bound, lb=self.pso_lower_bound, w=self.pso_w, verbose=True)


    def cal_init(self, infra_box, vehicle_box):
        centroid_infra = np.mean(infra_box, axis=0)
        centroid_vehicle = np.mean(vehicle_box, axis=0)

        infra_box = infra_box - centroid_infra
        vehicle_box = vehicle_box - centroid_vehicle

        # H = np.dot(vehicle_box.T, infra_box) 
        H = np.dot(infra_box.T, vehicle_box) # 这个地方要多斟酌
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1mm
            R = np.dot(Vt.T, U.T)

        t = -np.dot(R, centroid_infra.T) + centroid_vehicle.T

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()  

        return T

    def objective_func(self, infra_box, vehicle_box):
        iou3d, _ = box3d_iou(np.array(infra_box), np.array(vehicle_box))
        return iou3d
    
    def execute(self):
        self.pso.run()
        plt.plot(self.pso.gbest_y_hist)
        plt.show()


class PSOTask_new_type(Task):
    def __init__(self) -> None:
        self.reader = Reader('config.yml')
        self.infra_reader = InfraReader('config.yml')
        self.vehicle_reader = VehicleReader('config.yml')
        self.cooperative_reader = CooperativeReader('config.yml')
        self.filter3dBoxes = Filter3dBoxes()

        self.infra_boxes_object_list = self.filter3dBoxes.filter_according_to_size_distance_occlusion_truncation(self.infra_reader.get_infra_boxes_object_list())
        self.vehicle_boxes_object_list = self.filter3dBoxes.filter_according_to_size_distance_occlusion_truncation(self.vehicle_reader.get_vehicle_boxes_object_list())
        

        # ub: [-45, 3, 3, 1, 1, 3.1415926]
        # lb: [-50, -3, -3, -1, -1, -3.1415926]
        self.pso_lower_bound = [-55, -10, -10, -math.pi, -math.pi, -math.pi]
        self.pso_upper_bound = [-40, 10, 10, math.pi, math.pi, math.pi]
        self.pso_init_X = self.get_pso_init_X()
        self.pso_w = (0.8, 0.8)

        # self.pso_lower_bound = [-100, -100, -10, -math.pi, -math.pi, -math.pi]
        # self.pso_upper_bound = [100, 100, 10, math.pi, math.pi, math.pi]
        # self.pso_init_X = np.eye(4)
        # self.pso_w = (0.8, 1.2)
        
        self.pso = PSO_new_type(self.objective_func, self.infra_boxes_object_list, self.vehicle_boxes_object_list, self.pso_init_X, 
                       pop=self.reader.parse_pso_pop_yaml(), max_iter=self.reader.parse_pso_max_iter_yaml(),
                       ub=self.pso_upper_bound, lb=self.pso_lower_bound, w=self.pso_w, verbose=True)


    def cal_init(self, infra_box, vehicle_box):
        centroid_infra = np.mean(infra_box, axis=0)
        centroid_vehicle = np.mean(vehicle_box, axis=0)

        infra_box = infra_box - centroid_infra
        vehicle_box = vehicle_box - centroid_vehicle

        # H = np.dot(vehicle_box.T, infra_box) 
        H = np.dot(infra_box.T, vehicle_box) # 这个地方要多斟酌
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = -np.dot(R, centroid_infra.T) + centroid_vehicle.T

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()  

        return T

    def get_assignment_from_KP(self, topK = 5):
        filename = './output/KP.csv'
        KP = pd.read_csv(filename, header=None)
        indices_and_values = [(index, value) for index, value in np.ndenumerate(KP)]
        topK_matches = sorted(indices_and_values, key=lambda x: x[1], reverse=True)[:topK]
        return topK_matches

    def get_pso_init_X(self):
        pso_init_X = []
        matches = self.get_assignment_from_KP()
        for match in matches:
            infra_box = self.infra_boxes_object_list[match[0][0]].get_bbox3d_8_3()
            vehicle_box = self.vehicle_boxes_object_list[match[0][1]].get_bbox3d_8_3()
            pso_init_X.append(self.cal_init(infra_box, vehicle_box))
        return np.array(pso_init_X)


    def objective_func(self, infra_box, vehicle_box):
        iou3d, _ = box3d_iou(np.array(infra_box), np.array(vehicle_box))
        return iou3d
    
    def execute(self):
        self.pso.run()
        plt.plot(self.pso.gbest_y_hist)
        plt.show()


if __name__ == "__main__":
    
    pso_task = PSOTask_new_type()
    pso_task.execute()
