import numpy as np
import math
import matplotlib.pyplot as plt

from Task import Task
import sys
sys.path.append('./reader')
from InfraReader import InfraReader
from VehicleReader import VehicleReader
from Reader import Reader
from module.PSO import PSO
from module.calculate_IoU import box3d_iou
from module.convert_utils import get_time


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
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        t = -np.dot(R, centroid_infra.T) + centroid_vehicle.T

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.flatten()  

        return T

    def objective_func(self, infra_box, vehicle_box):
        iou3d, _ = box3d_iou(np.array(infra_box), np.array(vehicle_box))
        return iou3d
    
    @get_time
    def execute(self):
        self.pso.run()
        plt.plot(self.pso.gbest_y_hist)
        plt.show()



if __name__ == "__main__":
    
    pso_task = PSOTask()
    pso_task.execute()

    
