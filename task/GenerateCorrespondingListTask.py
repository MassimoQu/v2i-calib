import sys
sys.path.append('./reader')
sys.path.append('./task/module')

import numpy as np

from CooperativeReader import CooperativeReader
from calculate_IoU import box3d_iou
from Filter3dBoxes import Filter3dBoxes
from CoordinateConversion import CoordinateConversion



# 用程序生成部分匹配真值 
class GenerateCorrespondingListTask():
    def __init__(self, yaml_filename):
        self.cooperative_reader = CooperativeReader(yaml_filename)
        self.infra_bboxes_object_list, self.vehicle_bboxes_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_bboxes_object_list()

    def cal_single_3dIoU(self, box1, box2):
        iou, _ = box3d_iou(np.array(box1), np.array(box2))
        return iou

    def generate_corresponding_list(self, infra_bboxes_object_list, vehicle_bboxes_object_list):
        Y = 0
        infra_vehicle_boxes3d_IoU_list = []
        corresponding_list = []

        for i, infra_bbox_object in enumerate(infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(vehicle_bboxes_object_list):
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    box3d_IoU_score = self.cal_single_3dIoU(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    if box3d_IoU_score > 0:
                        infra_vehicle_boxes3d_IoU_list.append(box3d_IoU_score)
                        corresponding_list.append((i, j))

        if len(infra_vehicle_boxes3d_IoU_list) != 0:
            Y = np.mean(infra_vehicle_boxes3d_IoU_list)
            print('大于0的Y得分数量: ', len(infra_vehicle_boxes3d_IoU_list))
            print('平均得分：', Y)

        return corresponding_list, infra_vehicle_boxes3d_IoU_list, Y

    def cal_Y_score(self, infra_bboxes_object_list, vehicle_bboxes_object_list):
        Y = 0
        infra_vehicle_boxes3d_IoU_list = []

        for i, infra_bbox_object in enumerate(infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(vehicle_bboxes_object_list):
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    box3d_IoU_score = self.cal_single_3dIoU(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    if box3d_IoU_score > 0:
                        infra_vehicle_boxes3d_IoU_list.append(box3d_IoU_score)

        if len(infra_vehicle_boxes3d_IoU_list) != 0:
            Y = np.mean(infra_vehicle_boxes3d_IoU_list)

        return Y


    def get_infra_converted_corresponding_list(self):
        filter3dBoxes = Filter3dBoxes()
        filtered_infra_3dboxes = filter3dBoxes.filter_according_to_occlusion_truncation(self.infra_bboxes_object_list, 1, 1)
        filtered_vehicle_3dboxes = filter3dBoxes.filter_according_to_occlusion_truncation(self.vehicle_bboxes_object_list, 1, 1)
        converted_filtered_infra_3dboxes = CoordinateConversion().convert_bboxes_object_list_infra_lidar_2_vehicle_lidar(filtered_infra_3dboxes)
        infra_vehicle_corresponding_list, corresponding_IoU_list, y_score = self.generate_corresponding_list(converted_filtered_infra_3dboxes, filtered_vehicle_3dboxes)
        return infra_vehicle_corresponding_list, corresponding_IoU_list, y_score

    def save_infra_vehicle_corresponding_list(self):
        infra_vehicle_corresponding_list, _, _ = self.get_infra_converted_corresponding_list()
        # data/cooperative-vehicle-infrastructure/cooperative/label_corresponding
        output_dir = 'data/cooperative-vehicle-infrastructure/cooperative/label_corresponding'
        np.save(output_dir + '/infra_003920_vehicle_020092.npy', infra_vehicle_corresponding_list)

    def plot_corresponding_IoU_list(self):
        _, corresponding_IoU_list, _ = self.get_infra_converted_corresponding_list()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.boxplot(corresponding_IoU_list, patch_artist=True)
        plt.show()

    def load_infra_vehicle_corresponding_list(self):
        # data/cooperative-vehicle-infrastructure/cooperative/label_corresponding
        output_dir = 'data/cooperative-vehicle-infrastructure/cooperative/label_corresponding'
        infra_vehicle_corresponding_list = np.load(output_dir + '/infra_003920_vehicle_020092.npy')
        return infra_vehicle_corresponding_list


if __name__ == "__main__":
    generateCorrespondingListTask = GenerateCorrespondingListTask('config.yml')
    # generateCorrespondingListTask.plot_corresponding_IoU_list()
    filter3dBoxes = Filter3dBoxes()
    cooperative_reader = CooperativeReader('config.yml')
    infra_bboxes_object_list, vehicle_bboxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes3d_object_lists_vehicle_coordinate()
    infra_boxes_object_list = filter3dBoxes.filter_according_to_size_distance_occlusion_truncation(infra_bboxes_object_list)
    vehicle_boxes_object_list = filter3dBoxes.filter_according_to_size_distance_occlusion_truncation(vehicle_bboxes_object_list)
    infra_vehicle_corresponding_list, IoU_list, Y = generateCorrespondingListTask.generate_corresponding_list(infra_boxes_object_list, vehicle_boxes_object_list)
    print('infra_vehicle_corresponding_list: ', infra_vehicle_corresponding_list)
    print('IoU_list: ', IoU_list)
    