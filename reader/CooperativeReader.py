import os.path as osp
from Reader import Reader

class CooperativeReader():
    def __init__(self, yaml_filename):
        self.reader = Reader(yaml_filename)

    def parse_cooperative_lidar_i2v(self):
        return osp.join(self.reader.folder_root, 'cooperative', 'calib', 'lidar_i2v', self.reader.vehicle_file_name + '.json')
    
    def get_cooperative_lidar_i2v(self):
        lidar_i2v = self.reader.read_json(self.parse_cooperative_lidar_i2v())
        rotation = lidar_i2v["rotation"]
        translation = lidar_i2v["translation"]
        return rotation, translation
    
