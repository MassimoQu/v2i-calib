import math
import numpy as np
import json
import yaml
import open3d as o3d
from BBox3d import BBox3d



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
            # box = [int(label['2d_box']['xmin']), int(label['2d_box']['ymin']), int(label['2d_box']['xmax']), int(label['2d_box']['ymax'])]

            boxes_dict[box_type].append(box)

        for box_type, box in boxes_dict.items():
            boxes_dict[box_type] = np.array(box)
            
        return boxes_dict

    def is_high_precision(self, label):
        if int(label["truncated_state"]) == 2 or int(label["occluded_state"]) == 2:
            return False
        else :
            return True

    def get_3dboxes_8_3(self, label):
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



    def get_3dboxes_dict_n_8_3(self, path_label, high_precision_constraint_flag):

        labels = self.read_json(path_label) 
        
        boxes_dict = {}
        for label in labels:
            
            if high_precision_constraint_flag and not self.is_high_precision(label):
                continue

            box_type = label["type"]

            if box_type not in boxes_dict.keys():
                boxes_dict[box_type] = []

            box = self.get_3dboxes_8_3(label)

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
    
    def get_bbox3d_list(self, path_label):
        labels = self.read_json(path_label) 
        bbox3d_list = []
        for label in labels:
            bbox3d_list.append(BBox3d(label["label"], self.get_3dboxes_8_3(label)))
        return bbox3d_list

    def get_occluded_truncated_state_list(self, path_label):
        labels = self.read_json(path_label) 
        occluded_truncated_state_list = []
        for label in labels:
            occluded_truncated_state_list.append([int(label["occluded_state"]), int(label["truncated_state"])])
        return occluded_truncated_state_list

    def get_pointcloud(self, path_pointcloud):
        pointpillar = o3d.io.read_point_cloud(path_pointcloud)
        points = np.asarray(pointpillar.points)
        return points

    def get_intrinsic(self, path_intrinsic):
        my_json = self.read_json(path_intrinsic)
        cam_K = my_json["cam_K"]
        calib = np.zeros([3, 4])
        calib[:3, :3] = np.array(cam_K).reshape([3, 3], order="C")
        return calib
