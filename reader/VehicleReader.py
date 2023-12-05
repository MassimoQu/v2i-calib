import os.path as osp
import cv2
from Reader import Reader
from read_utils import read_json


class VehicleReader(Reader):
    def __init__(self, vehicle_file_name, data_folder = './data'):
        super().__init__(data_folder)
        self.vehicle_file_name = vehicle_file_name

    def parse_vehicle_intrinsic_path(self):
        return osp.join(self.cooperative_folder, 'vehicle-side', 'calib', 'camera_intrinsic', self.vehicle_file_name + '.json')
    
    def parse_vehicle_novatel2world_path(self):
        return osp.join(self.cooperative_folder, 'vehicle-side', 'calib', 'novatel_to_world', self.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2camera_path(self):
        return osp.join(self.cooperative_folder, 'vehicle-side', 'calib', 'lidar_to_camera', self.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2novatel_path(self):
        return osp.join(self.cooperative_folder, 'vehicle-side', 'calib', 'lidar_to_novatel', self.vehicle_file_name + '.json')
    
    # def parse_vehicle_image_path(self):
    #     folder_vehicle_image = self.reader.para_yaml['boxes']['folder_vehicle_images']
    #     return osp.join(folder_vehicle_image, self.vehicle_file_name + '.jpg')

    # def parse_vehicle_pointcloud_path(self):
    #     folder_vehicle_pointcloud = self.reader.para_yaml['boxes']['folder_vehicle_pointcloud']
    #     return osp.join(folder_vehicle_pointcloud, self.vehicle_file_name + '.pcd')

    def parse_vehicle_label_path(self):
        return osp.join(self.cooperative_folder, 'vehicle-side', 'label', 'lidar', self.vehicle_file_name + '.json')


    # def get_vehicle_image(self):
    #     return cv2.imread(self.parse_vehicle_image_path())

    # def get_vehicle_pointcloud(self):
    #     return self.reader.get_pointcloud(self.parse_vehicle_pointcloud_path())
  
    def get_vehicle_boxes_object_list(self):
        return self.get_3dbbox_object_list(self.parse_vehicle_label_path())

    def get_vehicle_intrinsic(self):
        return self.get_intrinsic(self.parse_vehicle_intrinsic_path())
    
    def get_vehicle_novatel2world(self):
        novatel2world = read_json(self.parse_vehicle_novatel2world_path())
        rotation = novatel2world["rotation"]
        translation = novatel2world["translation"]
        return rotation, translation
    
    def get_vehicle_lidar2camera(self):
        lidar2camera = read_json(self.parse_vehicle_lidar2camera_path())
        rotation = lidar2camera["transform"]["rotation"]
        translation = lidar2camera["transform"]["translation"]
        return rotation, translation
    
    def get_lidar2novatel(self):
        lidar2novatel = read_json(self.parse_vehicle_lidar2novatel_path())
        rotation = lidar2novatel["transform"]["rotation"]
        translation = lidar2novatel["transform"]["translation"]
        return rotation, translation
    
