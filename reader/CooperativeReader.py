import os.path as osp
from Reader import Reader
from InfraReader import InfraReader
from VehicleReader import VehicleReader

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