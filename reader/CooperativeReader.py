import os.path as osp
import sys
sys.path.append('./task/module')
from convert_utils import implement_R_t_points_n_3
from Reader import Reader
from InfraReader import InfraReader
from VehicleReader import VehicleReader
from scipy.spatial.transform import Rotation


class CooperativeReader():
    def __init__(self, yaml_filename):
        self.reader = Reader(yaml_filename)
        self.infra_reader = InfraReader(yaml_filename)
        self.vehicle_reader = VehicleReader(yaml_filename)

    def parse_cooperative_lidar_i2v(self):
        return osp.join(self.reader.folder_root, 'cooperative', 'calib', 'lidar_i2v', self.reader.vehicle_file_name + '.json')
    
    def get_cooperative_lidar_i2v(self):
        lidar_i2v = self.reader.read_json(self.parse_cooperative_lidar_i2v())
        rotation = lidar_i2v["rotation"]
        translation = lidar_i2v["translation"]
        return rotation, translation
    
    def get_cooperative_infra_vehicle_bboxes_object_list(self):
        return self.infra_reader.get_infra_boxes_object_list(), self.vehicle_reader.get_vehicle_boxes_object_list()
    
    def get_cooperative_infra_vehicle_pointcloud(self):
        return self.infra_reader.get_infra_pointcloud(), self.vehicle_reader.get_vehicle_pointcloud()
    
    def get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate(self):
        infra_pointcloud, vehicle_pointcloud = self.get_cooperative_infra_vehicle_pointcloud()
        R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar = self.get_cooperative_lidar_i2v()

        converted_infra_pointcloud = implement_R_t_points_n_3(R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar, infra_pointcloud)
        return converted_infra_pointcloud, vehicle_pointcloud
    
    def get_cooperative_infra_vehicle_boxes3d_object_lists_vehicle_coordinate(self):
        infra_bboxes_object_list, vehicle_bboxes_object_list = self.get_cooperative_infra_vehicle_bboxes_object_list()
        R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar = self.get_cooperative_lidar_i2v()

        converted_infra_bboxes_object_list = []
        for bbox_object in infra_bboxes_object_list:
            converted_infra_bboxes_object = bbox_object.copy()
            converted_infra_bboxes_object.bbox3d_8_3 = implement_R_t_points_n_3(R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar, bbox_object.bbox3d_8_3)
            converted_infra_bboxes_object_list.append(converted_infra_bboxes_object)
        return converted_infra_bboxes_object_list, vehicle_bboxes_object_list
    
if __name__ == '__main__':
    cooperative_reader = CooperativeReader('config.yml')
    # R, _ = cooperative_reader.get_cooperative_lidar_i2v()
    # r = Rotation.from_matrix(R)
    # euler = r.as_euler('xyz', degrees=True) # roll pitch yaw
    # print(euler)

    