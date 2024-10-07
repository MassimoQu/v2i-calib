import os.path as osp
# import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils import convert_Rt_to_T
from Reader import Reader
from read_utils import read_json

class VehicleReader(Reader):
    def __init__(self, vehicle_file_name, data_folder = './data/cooperative-vehicle-infrastructure'):
        super().__init__(data_folder)
        self.vehicle_file_name = vehicle_file_name

    def parse_vehicle_intrinsic_path(self):
        return osp.join(self.data_folder, 'vehicle-side', 'calib', 'camera_intrinsic', self.vehicle_file_name + '.json')
    
    def parse_vehicle_novatel2world_path(self):
        return osp.join(self.data_folder, 'vehicle-side', 'calib', 'novatel_to_world', self.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2camera_path(self):
        return osp.join(self.data_folder, 'vehicle-side', 'calib', 'lidar_to_camera', self.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2novatel_path(self):
        return osp.join(self.data_folder, 'vehicle-side', 'calib', 'lidar_to_novatel', self.vehicle_file_name + '.json')
    
    def parse_vehicle_image_path(self):
        folder_vehicle_image = osp.join(self.data_folder, 'vehicle-side', 'image')
        return osp.join(folder_vehicle_image, self.vehicle_file_name + '.jpg')

    def parse_vehicle_pointcloud_path(self):
        folder_vehicle_pointcloud = osp.join(self.cfg.data.data_root_path, 'vehicle-side', 'velodyne')
        return osp.join(folder_vehicle_pointcloud, self.vehicle_file_name + '.pcd')

    def parse_vehicle_label_path(self):
        return osp.join(self.data_folder, 'vehicle-side', 'label', 'lidar', self.vehicle_file_name + '.json')
    
    def parse_vehicle_label_path_cooperative_fusioned(self):
        return osp.join(self.data_folder, 'vehicle-side', 'label', 'cooperative', self.vehicle_file_name + '.json')

    def parse_vehicle_label_predicted_path(self):
        '''
        location of detection result of https://github.com/AIR-THU/DAIR-V2X ->  DAIR-V2X/cache/vic-late-lidar/veh/lidar 
        '''
        return osp.join('/home/massimo/DAIR-V2X-calibration/cache/vic-late-lidar/veh/lidar', self.vehicle_file_name + '.json')

    # def get_vehicle_image(self):
    #     return cv2.imread(self.parse_vehicle_image_path())

    def get_vehicle_pointcloud(self):
        return self.get_pointcloud(self.parse_vehicle_pointcloud_path())
  
    def exist_vehicle_label(self):
        return osp.exists(self.parse_vehicle_label_path())

    def get_vehicle_boxes_object_list(self):
        return self.get_3dbbox_object_list(self.parse_vehicle_label_path())
    
    def get_vehicle_boxes_object_list_predicted(self):
        return self.get_3dbbox_object_list_predicted(self.parse_vehicle_label_predicted_path())

    def get_vehicle_boxes_object_list_cooperative_fusioned(self):
        return self.get_3dbbox_object_list(self.parse_vehicle_label_path_cooperative_fusioned())

    def get_vehicle_intrinsic(self):
        return self.get_intrinsic(self.parse_vehicle_intrinsic_path())
    
    def get_vehicle_novatel2world(self):
        novatel2world = read_json(self.parse_vehicle_novatel2world_path())
        rotation = novatel2world["rotation"]
        translation = novatel2world["translation"]
        return rotation, translation
    
    def get_vehicle_lidar2camera(self):
        lidar2camera = read_json(self.parse_vehicle_lidar2camera_path())
        rotation = lidar2camera["rotation"]
        translation = lidar2camera["translation"]
        return convert_Rt_to_T(rotation, translation)
    
    def get_lidar2novatel(self):
        lidar2novatel = read_json(self.parse_vehicle_lidar2novatel_path())
        rotation = lidar2novatel["transform"]["rotation"]
        translation = lidar2novatel["transform"]["translation"]
        return rotation, translation
    
