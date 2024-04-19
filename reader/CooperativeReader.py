import os.path as osp
import sys
sys.path.append('./process/utils/')
from extrinsic_utils import implement_R_t_points_n_3, convert_Rt_to_T
from read_utils import read_json
from InfraReader import InfraReader
from VehicleReader import VehicleReader
import sys
sys.path.append('./process/utils')
from extrinsic_utils import convert_Rt_to_T


class CooperativeReader():
    def __init__(self, infra_file_name = '003920', vehicle_file_name = '020092', data_folder = '/mnt/c/Users/10612/Downloads/cooperative_data'):
        self.infra_reader = InfraReader(infra_file_name, data_folder)
        self.vehicle_reader = VehicleReader(vehicle_file_name, data_folder)

    def parse_cooperative_lidar_i2v(self):
        return osp.join(self.vehicle_reader.cooperative_folder, 'cooperative', 'calib', 'lidar_i2v', self.vehicle_reader.vehicle_file_name + '.json')
    
    def get_cooperative_Rt_i2v(self):
        lidar_i2v = read_json(self.parse_cooperative_lidar_i2v())
        rotation = lidar_i2v["rotation"]
        translation = lidar_i2v["translation"]
        return rotation, translation
    
    def get_cooperative_T_i2v(self):
        return convert_Rt_to_T(*self.get_cooperative_Rt_i2v())
    
    def get_infra_vehicle_lidar2camera(self):
        return self.infra_reader.get_infra_lidar2camera(), self.vehicle_reader.get_vehicle_lidar2camera()

    def get_cooperative_infra_vehicle_boxes_object_list(self):
        return self.infra_reader.get_infra_boxes_object_list(), self.vehicle_reader.get_vehicle_boxes_object_list()
    
    def get_cooperative_infra_vehicle_boxes_object_list_predicted(self):
        return self.infra_reader.get_infra_boxes_object_list_predicted(), self.vehicle_reader.get_vehicle_boxes_object_list_predicted()
    
    def get_cooperative_infra_vehicle_pointcloud(self):
        return self.infra_reader.get_infra_pointcloud(), self.vehicle_reader.get_vehicle_pointcloud()
    
    def get_cooperative_infra_vehicle_image(self):
        return self.infra_reader.get_infra_image(), self.vehicle_reader.get_vehicle_image()
    
    def get_infra_vehicle_camera_instrinsics(self):
        return self.infra_reader.get_infra_intrinsic(), self.vehicle_reader.get_vehicle_intrinsic()

    def get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate(self):
        infra_pointcloud, vehicle_pointcloud = self.get_cooperative_infra_vehicle_pointcloud()
        R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar = self.get_cooperative_Rt_i2v()

        converted_infra_pointcloud = implement_R_t_points_n_3(R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar, infra_pointcloud)
        return converted_infra_pointcloud, vehicle_pointcloud
    
    def get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate(self):
        infra_bboxes_object_list, vehicle_bboxes_object_list = self.get_cooperative_infra_vehicle_boxes_object_list()
        R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar = self.get_cooperative_Rt_i2v()

        converted_infra_bboxes_object_list = []
        for bbox_object in infra_bboxes_object_list:
            converted_infra_bboxes_object = bbox_object.copy()
            converted_infra_bboxes_object.bbox3d_8_3 = implement_R_t_points_n_3(R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar, bbox_object.bbox3d_8_3)
            converted_infra_bboxes_object_list.append(converted_infra_bboxes_object)
        return converted_infra_bboxes_object_list, vehicle_bboxes_object_list
    

    