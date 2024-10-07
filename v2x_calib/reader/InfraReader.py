import os.path as osp
# import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from Reader import Reader
from read_utils import read_json
from utils import convert_Rt_to_T

class InfraReader(Reader):
    def __init__(self, infra_file_name, data_folder = './data/cooperative-vehicle-infrastructure'):
        super().__init__(data_folder)
        self.infra_file_name = infra_file_name
        
    def parse_infra_intrinsic_path(self):
        return osp.join(self.data_folder, 'infrastructure-side', 'calib', 'camera_intrinsic', self.infra_file_name + '.json')

    def parse_infra_virtuallidar2camera_path(self):
        return osp.join(self.data_folder, 'infrastructure-side', 'calib', 'virtuallidar_to_camera', self.infra_file_name + '.json')
    
    def parse_infra_virtuallidar2world_path(self):
        return osp.join(self.data_folder, 'infrastructure-side', 'calib', 'virtuallidar_to_world', self.infra_file_name + '.json')
    
    def parse_infra_image_path(self):
        folder_infra_image = osp.join(self.data_folder, 'infrastructure-side', 'image')
        return osp.join(folder_infra_image, self.infra_file_name + '.jpg')

    def parse_infra_pointcloud_path(self):
        folder_infra_pointcloud = osp.join(self.cfg.data.data_root_path, 'infrastructure-side', 'velodyne')
        return osp.join(folder_infra_pointcloud, self.infra_file_name + '.pcd')

    def parse_infra_label_path(self):
        return osp.join(self.data_folder, 'infrastructure-side', 'label', 'virtuallidar', self.infra_file_name + '.json')
    
    def parse_infra_label_predicted_path(self):
        '''
        location of detection result of https://github.com/AIR-THU/DAIR-V2X ->  DAIR-V2X/cache/vic-late-lidar/inf/lidar
        '''
        return osp.join('/home/massimo/DAIR-V2X-calibration/cache/vic-late-lidar/inf/lidar', self.infra_file_name + '.json')

    # def get_infra_image(self):
    #     return cv2.imread(self.parse_infra_image_path())

    def get_infra_pointcloud(self):
        return self.get_pointcloud(self.parse_infra_pointcloud_path())    
    
    def exist_infra_label(self):
        return osp.exists(self.parse_infra_label_path())

    def get_infra_boxes_object_list(self):
        return self.get_3dbbox_object_list(self.parse_infra_label_path())
    
    def get_infra_boxes_object_list_predicted(self):
        return self.get_3dbbox_object_list_predicted(self.parse_infra_label_predicted_path())

    def get_infra_intrinsic(self):
        return self.get_intrinsic(self.parse_infra_intrinsic_path())
    
    def get_infra_virtuallidar2world(self):
        virtuallidar2world = read_json(self.parse_infra_virtuallidar2world_path())
        rotation = virtuallidar2world["rotation"]
        translation = virtuallidar2world["translation"]
        translation[0][0] += virtuallidar2world["relative_error"]["delta_x"]
        translation[1][0] += virtuallidar2world["relative_error"]["delta_y"]
        return rotation, translation

    def get_infra_lidar2camera(self):
        lidar2camera = read_json(self.parse_infra_virtuallidar2camera_path())
        rotation = lidar2camera["rotation"]
        translation = lidar2camera["translation"]
        return convert_Rt_to_T(rotation, translation)
    