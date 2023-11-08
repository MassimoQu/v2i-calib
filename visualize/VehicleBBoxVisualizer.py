import sys
sys.path.append(r'E:\WorkSpace\vehicle-infrastructure-cooperation\vehicle_infrastructure_cooperation_normalized_code\task')
sys.path.append(r'E:\WorkSpace\vehicle-infrastructure-cooperation\vehicle_infrastructure_cooperation_normalized_code\reader')
# print(sys.path)
from VehicleReader import VehicleReader
from BBoxVisualizer import BBoxVisualizer

import cv2


class VehicleBBoxVisualizer():
    
    def __init__(self) -> None:
        self.reader = VehicleReader('config.yml')
        self.vehicle_boxes_3d_dict = self.reader.get_vehicle_boxes_dict()
        self.bbox_visualizer = BBoxVisualizer()


    def plot_boxes3d_vehicle(self):
        self.bbox_visualizer.plot_boxes3d_lists([self.vehicle_boxes_3d_dict],[(1, 0, 0)])

    def plot_boxes3d_pointcloud_vehicle(self):
        self.bbox_visualizer.plot_boxes3d_pointcloud(self.vehicle_boxes_3d_dict, self.reader.get_vehicle_pointcloud())
        
    def plot_3dboxes_vehicle_image(self):
        path_lidar2camera = self.reader.parse_vehicle_lidar2camera_path() 
        path_camera_intrinsic = self.reader.parse_vehicle_intrinsic_path()
        path_image = self.reader.parse_vehicle_image_path()
        _3dboxes_list_n_7 = self.reader.get_vehicle_boxes_list_n_7()
        self.bbox_visualizer.plot_3dboxes_image(_3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic)

    def plot_2dboxes_vehicle_image(self):
        image = self.bbox_visualizer.draw_2dboxes_image(self.reader.get_vehicle_image(), self.reader.get_vehicle_boxes_2d_dict().values(), color=(255, 0, 0))
        cv2.imshow("2dboxes_image", image)
        cv2.waitKey(0)

    def plot_boxes_2d3d_vehicle_image(self):
        path_lidar2camera = self.reader.parse_vehicle_lidar2camera_path() 
        path_camera_intrinsic = self.reader.parse_vehicle_intrinsic_path()
        path_image = self.reader.parse_vehicle_image_path()
        _3dboxes_list_n_7 = self.reader.get_vehicle_boxes_list_n_7()
        self.bbox_visualizer.plot_boxes_2d3d_image(_3dboxes_list_n_7, self.reader.get_vehicle_boxes_2d_dict(), path_lidar2camera, path_image, path_camera_intrinsic)


if __name__ == "__main__":
    bbox_visualizer = VehicleBBoxVisualizer()
    bbox_visualizer.plot_boxes3d_vehicle()#1
    # bbox_visualizer.plot_boxes3d_pointcloud_vehicle()#2
    # bbox_visualizer.plot_3dboxes_vehicle_image()#3
    # bbox_visualizer.plot_2dboxes_vehicle_image()#4
    # bbox_visualizer.plot_boxes_2d3d_vehicle_image()#5
