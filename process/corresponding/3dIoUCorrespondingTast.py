import numpy as np
from Task import Task
import sys
sys.path.append('./reader')
sys.path.append('./visualize')
from InfraReader import InfraReader
from VehicleReader import VehicleReader
from CooperativeReader import CooperativeReader
from Reader import Reader
from process.utils.extrinsic_utils import implement_R_t_3dbox_dict_n_8_3
from process.utils.IoU_utils import box3d_iou

# from BBoxVisualizer import BBoxVisualizer


class IoU3dCorrespondingTask(Task):
    def __init__(self):
        self.reader = Reader('config.yml')
        self.infra_reader = InfraReader('config.yml')
        self.vehicle_reader = VehicleReader('config.yml')
        self.cooperative_reader = CooperativeReader('config.yml')
        
        self.infra_boxes_dict = self.infra_reader.get_infra_boxes_dict()
        self.vehicle_boxes_dict = self.vehicle_reader.get_vehicle_boxes_dict()

        # self.bbox_visualizer = BBoxVisualizer()
        
    def cal_single_3dIoU(self, box1, box2):
        iou, _ = box3d_iou(np.array(box1), np.array(box2))
        return iou

    def cal_Y_score(self, infra_boxes_dict, vehicle_boxes_dict, infra_precision=None, vehicle_precision=None):
        type_list = set(list(infra_boxes_dict.keys()) + list(vehicle_boxes_dict.keys()))
    
        Y = 0

        IoU_matrix_list = []

        for box_type in type_list:
            if box_type not in infra_boxes_dict or box_type not in vehicle_boxes_dict:
                continue

            IoU_sum = 0
            mutual_cnt = 0
            IoU_matrix = np.zeros((len(infra_boxes_dict[box_type]), len(vehicle_boxes_dict[box_type])))
            
            for i, infra_box in enumerate(infra_boxes_dict[box_type]):
                for j, vehicle_box in enumerate(vehicle_boxes_dict[box_type]):
                    # boxes shape(8 ,3)

                    coefficient = 1.0
                    if infra_precision is not None and vehicle_precision is None:
                        if infra_precision[i] == 2 or vehicle_precision[j] == 2:
                            coefficient *= 0.1
                        elif infra_precision[i] == 1 or vehicle_precision[j] == 1:
                            coefficient *= 0.5

                    IoU_matrix[i, j] = self.cal_single_3dIoU(infra_box, vehicle_box) * coefficient
                    
                    if IoU_matrix[i, j] > 0:
                        IoU_sum += IoU_matrix[i, j]
                        mutual_cnt += 1
                        
            # print(IoU_matrix)
            if mutual_cnt != 0: 
                IoU_matrix_list.append(IoU_sum / mutual_cnt)
                print('box_type: ', box_type, ' IoU: ', IoU_sum / mutual_cnt)
                print(IoU_matrix)
                print('-----------------------------------')
            else: 
                IoU_matrix_list.append(0)
        
        if len(IoU_matrix_list) != 0:
            Y = np.mean(IoU_matrix_list)
            print('分项得分集合', IoU_matrix_list)
            print('取平均得到最终得分: ', Y)

        return Y


    def get_Y_score_T_infra2vehicle(self, visualize_flag=False):
        R, t = self.cooperative_reader.get_cooperative_lidar_i2v()
        infra_boxes_dict = implement_R_t_3dbox_dict_n_8_3(R, t, self.infra_boxes_dict)

        infra_precision = self.infra_reader.get_infra_occluded_truncated_state_list()
        vehicle_precision = self.vehicle_reader.get_vehicle_occluded_truncated_state_list()

        # if visualize_flag:
        #     self.bbox_visualizer.plot_boxes3d_lists_according_to_precision([infra_boxes_dict, self.vehicle_boxes_dict], 
        #                                             color_lists=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], precision_list=[infra_precision, vehicle_precision])

        return self.cal_Y_score(infra_boxes_dict, self.vehicle_boxes_dict, infra_precision, vehicle_precision)

    def execute(self):
        return self.get_Y_score_T_infra2vehicle(visualize_flag=False)
    

if __name__ == "__main__":
    task = IoU3dCorrespondingTask()
    print(task.execute())
