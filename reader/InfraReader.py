import os.path as osp
import cv2
from Reader import Reader
from read_utils import read_json


class InfraReader(Reader):
    def __init__(self, data_folder, infra_file_name):
        super().__init__(data_folder)
        self.infra_file_name = infra_file_name

    def parse_infra_intrinsic_path(self):
        return osp.join(self.cooperative_folder, 'infrastructure-side', 'calib', 'camera_intrinsic', self.infra_file_name + '.json')

    def parse_infra_virtuallidar2camera_path(self):
        return osp.join(self.cooperative_folder, 'infrastructure-side', 'calib', 'virtuallidar_to_camera', self.infra_file_name + '.json')
    
    def parse_infra_virtuallidar2world_path(self):
        return osp.join(self.cooperative_folder, 'infrastructure-side', 'calib', 'virtuallidar_to_world', self.infra_file_name + '.json')
    
    # def parse_infra_image_path(self):
    #     folder_infra_image = self.reader.para_yaml['boxes']['folder_infra_images']
    #     return osp.join(folder_infra_image, self.infra_file_name + '.jpg')

    # def parse_infra_pointcloud_path(self):
    #     folder_infra_pointcloud = self.reader.para_yaml['boxes']['folder_infra_pointcloud']
    #     return osp.join(folder_infra_pointcloud, self.infra_file_name + '.pcd')

    def parse_infra_label_path(self):
        return osp.join(self.cooperative_folder, 'infrastructure-side', 'label', 'virtuallidar', self.infra_file_name + '.json')
    

    # def get_infra_image(self):
    #     return cv2.imread(self.parse_infra_image_path())

    # def get_infra_pointcloud(self):
    #     return self.reader.get_pointcloud(self.parse_infra_pointcloud_path())    
    
    def get_infra_boxes_object_list(self):
        return self.get_3dbbox_object_list(self.parse_infra_label_path())

    def get_infra_intrinsic(self):
        return self.get_intrinsic(self.parse_infra_intrinsic_path())
    
    def get_infra_virtuallidar2world(self):
        virtuallidar2world = read_json(self.parse_infra_virtuallidar2world_path())
        rotation = virtuallidar2world["rotation"]
        translation = virtuallidar2world["translation"]
        translation[0][0] += virtuallidar2world["relative_error"]["delta_x"]
        translation[1][0] += virtuallidar2world["relative_error"]["delta_y"]
        return rotation, translation

