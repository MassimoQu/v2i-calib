import os.path as osp
import math
import numpy as np
import json
import yaml
import open3d as o3d
import cv2



def get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar):
    center_lidar = [center_lidar[0], center_lidar[1], center_lidar[2]]

    lidar_r = np.matrix(
        [[math.cos(yaw_lidar), -math.sin(yaw_lidar), 0], [math.sin(yaw_lidar), math.cos(yaw_lidar), 0], [0, 0, 1]]
    )
    l, w, h = obj_size
    center_lidar[2] = center_lidar[2] - h / 2
    corners_3d_lidar = np.matrix(
        [
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0, 0, 0, 0, h, h, h, h],
        ]
    )

    corners_3d_lidar = lidar_r * corners_3d_lidar + np.matrix(center_lidar).T

    return corners_3d_lidar.T





class Reader():
    def __init__(self, yaml_filename):
        self.para_yaml = self.read_yaml(yaml_filename)
        self.folder_root = self.para_yaml['boxes']['folder_root']
        self.infra_file_name = self.para_yaml['boxes']['file_name_infra']
        self.vehicle_file_name = self.para_yaml['boxes']['file_name_vehicle']


    def read_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
        
    def read_yaml(self, path):
        with open(path, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)


    def parse_infra_intrinsic_path(self):
        return osp.join(self.folder_root, 'infrastructure-side', 'calib', 'camera_intrinsic', self.infra_file_name + '.json')

    def parse_infra_virtuallidar2camera_path(self):
        return osp.join(self.folder_root, 'infrastructure-side', 'calib', 'virtuallidar_to_camera', self.infra_file_name + '.json')
    
    def parse_infra_virtuallidar2world_path(self):
        return osp.join(self.folder_root, 'infrastructure-side', 'calib', 'virtuallidar_to_world', self.infra_file_name + '.json')

    def parse_vehicle_intrinsic_path(self):
        return osp.join(self.folder_root, 'vehicle-side', 'calib', 'camera_intrinsic', self.vehicle_file_name + '.json')
    
    def parse_vehicle_novatel2world_path(self):
        return osp.join(self.folder_root, 'vehicle-side', 'calib', 'novatel_to_world', self.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2camera_path(self):
        return osp.join(self.folder_root, 'vehicle-side', 'calib', 'lidar_to_camera', self.vehicle_file_name + '.json')
    
    def parse_vehicle_lidar2novatel_path(self):
        return osp.join(self.folder_root, 'vehicle-side', 'calib', 'lidar_to_novatel', self.vehicle_file_name + '.json')
    

    def parse_infra_image_path(self):
        folder_infra_image = self.para_yaml['boxes']['folder_infra_images']
        return osp.join(folder_infra_image, self.infra_file_name + '.jpg')
    
    def parse_vehicle_image_path(self):
        folder_vehicle_image = self.para_yaml['boxes']['folder_vehicle_images']
        return osp.join(folder_vehicle_image, self.vehicle_file_name + '.jpg')

    def parse_infra_pointcloud_path(self):
        folder_infra_pointcloud = self.para_yaml['boxes']['folder_infra_pointcloud']
        return osp.join(folder_infra_pointcloud, self.infra_file_name + '.pcd')
    
    def parse_vehicle_pointcloud_path(self):
        folder_vehicle_pointcloud = self.para_yaml['boxes']['folder_vehicle_pointcloud']
        return osp.join(folder_vehicle_pointcloud, self.vehicle_file_name + '.pcd')

    def parse_infra_label_path(self):
        return osp.join(self.folder_root, 'infrastructure-side', 'label', 'virtuallidar', self.infra_file_name + '.json')

    def parse_vehicle_label_path(self):
        return osp.join(self.folder_root, 'vehicle-side', 'label', 'lidar', self.vehicle_file_name + '.json')


    def parse_pso_max_iter_yaml(self):
        return int(self.para_yaml['pso']['max_iter'])
    
    def parse_pso_pop_yaml(self):
        return int(self.para_yaml['pso']['population'])


    def get_2dbbox_dict_n_4(self, path_label):

        labels = self.read_json(path_label) 
        
        boxes_dict = {}
        for label in labels:
        
            box_type = label["type"]

            if box_type not in boxes_dict.keys():
                boxes_dict[box_type] = []

            box = [label['2d_box']['xmin'], label['2d_box']['ymin'], label['2d_box']['xmax'], label['2d_box']['ymax']]

            boxes_dict[box_type].append(box)

        for box_type, box in boxes_dict.items():
            boxes_dict[box_type] = np.array(box)
            
        return boxes_dict

    def get_3dboxes_dict_n_8_3(self, path_label):

        labels = self.read_json(path_label) 
        
        boxes_dict = {}
        for label in labels:
        
            box_type = label["type"]

            if box_type not in boxes_dict.keys():
                boxes_dict[box_type] = []

            obj_size = [
                float(label["3d_dimensions"]["l"]),
                float(label["3d_dimensions"]["w"]),
                float(label["3d_dimensions"]["h"]),
            ]
            yaw_lidar = float(label["rotation"])
            center_lidar = [
                float(label["3d_location"]["x"]),
                float(label["3d_location"]["y"]),
                float(label["3d_location"]["z"]),
            ]

            box = get_lidar_3d_8points(obj_size, yaw_lidar, center_lidar)

            boxes_dict[box_type].append(box)

        for box_type, box in boxes_dict.items():
            boxes_dict[box_type] = np.array(box)
            
        return boxes_dict    

    def get_3dboxes_list_n_7(self, path_label):
        labels = self.read_json(path_label) 
    
        boxes_list = []
        for label in labels:    
            if "rotation" not in label.keys():
                label["rotation"] = 0.0
            boxes_list.append([label["3d_dimensions"], label["3d_location"], label["rotation"]])

        return boxes_list
    
    def get_pointcloud(self, path_pointcloud):
        pointpillar = o3d.io.read_point_cloud(path_pointcloud)
        points = np.asarray(pointpillar.points)
        return points
    
    def get_infra_image(self):
        return cv2.imread(self.parse_infra_image_path())

    def get_vehicle_image(self):
        return cv2.imread(self.parse_vehicle_image_path())

    def get_infra_pointcloud(self):
        return self.get_pointcloud(self.parse_infra_pointcloud_path())
    
    def get_vehicle_pointcloud(self):
        return self.get_pointcloud(self.parse_vehicle_pointcloud_path())
    

    def get_infra_boxes_dict(self):
        return self.get_3dboxes_dict_n_8_3(self.parse_infra_label_path())
    
    def get_vehicle_boxes_dict(self):
        return self.get_3dboxes_dict_n_8_3(self.parse_vehicle_label_path())
    
    def get_infra_boxes_2d_dict(self):
        return self.get_2dbbox_dict_n_4(self.parse_infra_label_path())

    def get_vehicle_boxes_2d_dict(self):
        return self.get_2dbbox_dict_n_4(self.parse_vehicle_label_path())
    
    def get_infra_boxes_list_n_7(self):
        return self.get_3dboxes_list_n_7(self.parse_infra_label_path())

    def get_vehicle_boxes_list_n_7(self):
        return self.get_3dboxes_list_n_7(self.parse_vehicle_label_path())

    def get_intrinsic(self, path_intrinsic):
        my_json = self.read_json(path_intrinsic)
        cam_K = my_json["cam_K"]
        calib = np.zeros([3, 4])
        calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")
        return calib

    def get_infra_intrinsic(self):
        return self.get_intrinsic(self.parse_infra_intrinsic_path())
        
    def get_vehicle_intrinsic(self):
        return self.get_intrinsic(self.parse_vehicle_intrinsic_path())
    
    