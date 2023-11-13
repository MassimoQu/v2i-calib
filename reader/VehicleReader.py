import os.path as osp
import cv2
from Reader import Reader


class VehicleReader():
    def __init__(self, yaml_filename):
        self.reader = Reader(yaml_filename)


    def parse_vehicle_intrinsic_path(self):
        return osp.join(self.reader.folder_root, 'vehicle-side', 'calib', 'camera_intrinsic', self.reader.vehicle_file_name + '.json')
    
    def parse_vehicle_novatel2world_path(self):
        return osp.join(self.reader.folder_root, 'vehicle-side', 'calib', 'novatel_to_world', self.reader.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2camera_path(self):
        return osp.join(self.reader.folder_root, 'vehicle-side', 'calib', 'lidar_to_camera', self.reader.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2novatel_path(self):
        return osp.join(self.reader.folder_root, 'vehicle-side', 'calib', 'lidar_to_novatel', self.reader.vehicle_file_name + '.json')
    
    def parse_vehicle_image_path(self):
        folder_vehicle_image = self.reader.para_yaml['boxes']['folder_vehicle_images']
        return osp.join(folder_vehicle_image, self.reader.vehicle_file_name + '.jpg')

    def parse_vehicle_pointcloud_path(self):
        folder_vehicle_pointcloud = self.reader.para_yaml['boxes']['folder_vehicle_pointcloud']
        return osp.join(folder_vehicle_pointcloud, self.reader.vehicle_file_name + '.pcd')

    def parse_vehicle_label_path(self):
        return osp.join(self.reader.folder_root, 'vehicle-side', 'label', 'lidar', self.reader.vehicle_file_name + '.json')


    def get_vehicle_image(self):
        return cv2.imread(self.parse_vehicle_image_path())

    def get_vehicle_pointcloud(self):
        return self.reader.get_pointcloud(self.parse_vehicle_pointcloud_path())

    def get_vehicle_boxes_dict(self):
        return self.reader.get_3dboxes_dict_n_8_3(self.parse_vehicle_label_path())
    
    def get_vehicle_boxes_object_list(self):
        return self.reader.get_3dbbox_object_list(self.parse_vehicle_label_path())

    def get_vehicle_boxes_2d_dict(self):
        return self.reader.get_2dbbox_dict_n_4(self.parse_vehicle_label_path())
    
    def get_vehicle_boxes_list_n_7(self):
        return self.reader.get_3dboxes_list_n_7(self.parse_vehicle_label_path())

    def get_vehicle_occluded_truncated_state_list(self):
        return self.reader.get_occluded_truncated_state_list(self.parse_vehicle_label_path())

    def get_vehicle_intrinsic(self):
        return self.reader.get_intrinsic(self.parse_vehicle_intrinsic_path())
    
    def get_vehicle_novatel2world(self):
        novatel2world = self.reader.read_json(self.parse_vehicle_novatel2world_path())
        rotation = novatel2world["rotation"]
        translation = novatel2world["translation"]
        return rotation, translation
    
    def get_vehicle_lidar2camera(self):
        lidar2camera = self.reader.read_json(self.parse_vehicle_lidar2camera_path())
        rotation = lidar2camera["transform"]["rotation"]
        translation = lidar2camera["transform"]["translation"]
        return rotation, translation
    
    def get_lidar2novatel(self):
        lidar2novatel = self.reader.read_json(self.parse_vehicle_lidar2novatel_path())
        rotation = lidar2novatel["transform"]["rotation"]
        translation = lidar2novatel["transform"]["translation"]
        return rotation, translation
    

    
