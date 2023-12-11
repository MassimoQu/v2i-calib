import sys
sys.path.append('./reader')
sys.path.append('./process/utils')

import numpy as np
from IoU_utils import cal_3dIoU


class CorrespondingDetector():
    def __init__(self, infra_bboxes_object_list, vehicle_bboxes_object_list):
        self.infra_bboxes_object_list = infra_bboxes_object_list
        self.vehicle_bboxes_object_list = vehicle_bboxes_object_list

        self.corresponding_IoU_dict = {}
        self.Y = -1

        self.cal_IoU_corresponding()


    def cal_IoU_corresponding(self):

        for i, infra_bbox_object in enumerate(self.infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(self.vehicle_bboxes_object_list):
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    box3d_IoU_score = cal_3dIoU(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    if box3d_IoU_score > 0:
                        self.corresponding_IoU_dict[(i, j)] = box3d_IoU_score

        if len(self.corresponding_IoU_dict) != 0:
            self.Y = np.sum(list(self.corresponding_IoU_dict.values()))

    def get_matched_num(self):
        return len(self.corresponding_IoU_dict)
    
    def get_total_num(self):
        return min(len(self.infra_bboxes_object_list), len(self.vehicle_bboxes_object_list))
    
    def get_Yscore(self):
        return self.Y / self.get_total_num()

