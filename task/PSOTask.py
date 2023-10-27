import numpy as np
import math
import matplotlib.pyplot as plt

from Task import Task
from Reader import Reader
from PSO import PSO
from calculate_IoU import box3d_iou
from utils import get_time


class PSOTask(Task):
    def __init__(self) -> None:

        self.pso_lower_bound = [-100, -100, -10, -math.pi, -math.pi, -math.pi]
        self.pso_upper_bound = [100, 100, 10, math.pi, math.pi, math.pi]
        self.pso_init_X = np.eye(4)

        self.reader = Reader('config.yml')
        self.infra_boxes = self.reader.get_infra_boxes()
        self.vehicle_boxes = self.reader.get_vehicle_boxes()
        
        self.pso = PSO(self.objective_func, self.infra_boxes, self.vehicle_boxes, self.pso_init_X, 
                       pop=self.reader.parse_pso_pop_yaml(), max_iter=self.reader.parse_pso_max_iter_yaml(),
                       ub=self.pso_upper_bound, lb=self.pso_lower_bound, verbose=True)

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

    
