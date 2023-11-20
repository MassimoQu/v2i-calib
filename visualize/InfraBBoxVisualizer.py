import sys
sys.path.append('./task')
sys.path.append('./reader')
# print(sys.path)
from InfraReader import InfraReader
from BBoxVisualizer_open3d import BBoxVisualizer_open3d as BBoxVisualizer
# import BBoxVisualizer_open3d as BBoxVisualizer

import cv2


class InfraBBoxVisualizer():
    
    def __init__(self, bbox_visualizer=None) -> None:
        self.reader = InfraReader('config.yml')
        self.infra_boxes_3d_dict = self.reader.get_infra_boxes_dict()
        self.bbox_visualizer = BBoxVisualizer()


    def plot_boxes3d_dict_infra(self):
        self.bbox_visualizer.plot_boxes3d_lists([self.infra_boxes_3d_dict], [(0, 1, 0)])

    def plot_boxes3d_dict_pointcloud_infra(self):
        self.bbox_visualizer.plot_boxes3d_pointcloud(self.infra_boxes_3d_dict, self.reader.get_infra_pointcloud())#还是在传dict

    def plot_boxes3d_objects_list_infra(self):
        self.bbox_visualizer.plot_boxes3d_lists([self.reader.get_infra_boxes_object_list()], [(0, 1, 0)])

    def plot_boxes3d_objects_list_pointcloud_infra(self):
        self.bbox_visualizer.plot_boxes3d_pointcloud(self.reader.get_infra_boxes_object_list(), self.reader.get_infra_pointcloud())

    def plot_3dboxes_infra_image(self):
        path_lidar2camera = self.reader.parse_infra_virtuallidar2camera_path()
        path_camera_intrinsic = self.reader.parse_infra_intrinsic_path()
        path_image = self.reader.parse_infra_image_path()
        _3dboxes_list_n_7 = self.reader.get_infra_boxes_list_n_7()
        self.bbox_visualizer.plot_3dboxes_image(_3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic)

    def plot_2dboxes_infra_image(self):
        image = self.bbox_visualizer.draw_2dboxes_image(self.reader.get_infra_image(), self.reader.get_infra_boxes_2d_dict().values(), color=(0, 255, 0))
        cv2.imshow("2dboxes_image", image)
        cv2.waitKey(0)
        
    def plot_boxes_2d3d_infra_image(self):
        path_lidar2camera = self.reader.parse_infra_virtuallidar2camera_path()
        path_camera_intrinsic = self.reader.parse_infra_intrinsic_path()
        path_image = self.reader.parse_infra_image_path()
        _3dboxes_list_n_7 = self.reader.get_infra_boxes_list_n_7()
        self.bbox_visualizer.plot_boxes_2d3d_image(_3dboxes_list_n_7, self.reader.get_infra_boxes_2d_dict(), path_lidar2camera, path_image, path_camera_intrinsic)



if __name__ == "__main__":
    bbox_visualizer = InfraBBoxVisualizer()
    bbox_visualizer.plot_boxes3d_objects_list_infra()
    