import numpy as np
import cv2

import sys
sys.path.append('./reader')
from InfraReader import InfraReader
from vis_utils import get_cam_8_points, vis_label_in_img


class BBoxVisualizer():
    
    def __init__(self) -> None:
        pass


    def draw_2dboxes_image(self, image, boxes2d, color = (255, 255, 0)):
        for box in boxes2d:
            box = box[0]
            cv2.rectangle(image, ((box[0]).astype(np.int32), (box[1].astype(np.int32))), ((box[2]).astype(np.int32), (box[3]).astype(np.int32)), color, 2)
        return image

    def plot_3dboxes_image(self, _3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic, color=(0, 255, 0)):
        img = vis_label_in_img(get_cam_8_points(_3dboxes_list_n_7, path_lidar2camera), path_image, path_camera_intrinsic, color=color)
        cv2.imshow('3dboxes_image', img)
        cv2.waitKey(0)

    
    # to delete the dict type
    def plot_boxes_2d3d_image(self, _3dboxes_list_n_7, boxes_2d_dict, path_lidar2camera, path_image, path_camera_intrinsic, color_list = [(0, 255, 0), (255, 0, 0)]):
        img = vis_label_in_img(get_cam_8_points(_3dboxes_list_n_7, path_lidar2camera), path_image, path_camera_intrinsic, color=color_list[0])

        self.draw_2dboxes_image(img, boxes_2d_dict.values(), color=color_list[1])

        cv2.imshow('3dboxes_image', img)
        cv2.waitKey(0)



if __name__ == "__main__":
    infra_reader = InfraReader('config.yml')
    infra_pointcloud, infra_boxes_object_list = infra_reader.get_infra_pointcloud(), infra_reader.get_infra_boxes_object_list()
    bbox_visualizer = BBoxVisualizer()
    # bbox_visualizer.plot_boxes3d_objects_list_pointcloud_open3d(infra_boxes_object_list, infra_pointcloud)
