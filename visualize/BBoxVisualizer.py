import numpy as np
import cv2

import sys
sys.path.append('./reader')
from VehicleReader import VehicleReader
from vis_utils import get_cam_8_points, vis_label_in_img


class BBoxVisualizer():
    
    def __init__(self) -> None:
        pass


    def draw_2dboxes_image(self, image, boxes2d, color = (255, 255, 0)):
        boxes2d = np.array(boxes2d)
        for box in boxes2d:
            cv2.rectangle(image, ((box[0]).astype(np.int32), (box[1].astype(np.int32))), ((box[2]).astype(np.int32), (box[3]).astype(np.int32)), color, 2)
        return image
    
    def plot_boxes_object_list_image(self, boxes_object_list, image, color = (0, 255, 0)):
        boxes2d = [box_object.get_bbox2d_4() for box_object in boxes_object_list]
        self.draw_2dboxes_image(image, boxes2d, color=color)
        cv2.imshow('boxes_object_list_image', image)
        cv2.waitKey(0)

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
    vehicle_reader = VehicleReader('015904')
    vehicle_object_list = vehicle_reader.get_vehicle_boxes_object_list()
    BBoxVisualizer().plot_boxes_object_list_image(vehicle_object_list, vehicle_reader.get_vehicle_image())
