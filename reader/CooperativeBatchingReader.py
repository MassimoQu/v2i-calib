import os.path as osp
from read_utils import read_yaml, read_json
from CooperativeReader import CooperativeReader


class CooperativeBatchingReader:
    def __init__(self, yaml_filename = 'config.yml'):
        para_yaml = read_yaml(yaml_filename)
        self.data_folder = para_yaml['data_folder']
        self.path_data_info = osp.join(self.data_folder, 'cooperative-vehicle-infrastructure', 'cooperative', 'data_info.json')
        
        self.infra_file_names, self.vehicle_file_names = self.get_infra_vehicle_file_names(self.path_data_info)

    
    def get_infra_vehicle_file_names(self, path_data_info = None):
        if path_data_info is None:
            path_data_info = self.path_data_info
        data_infos = read_json(path_data_info)
        infra_file_names = []
        vehicle_file_names = []
        for data_info in data_infos:
            infra_img_path = data_info["infrastructure_image_path"]
            vehicle_img_path = data_info["vehicle_image_path"]
            infra_file_names.append(infra_img_path.split('/')[-1].split('.')[0])
            vehicle_file_names.append(vehicle_img_path.split('/')[-1].split('.')[0])
        return infra_file_names, vehicle_file_names
    
    def generate_infra_vehicle_bboxes_object_list(self, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(self.infra_file_names)
        if start_idx < 0 or start_idx >= end_idx:
            raise ValueError('start_idx should be in [0, end_idx)')
        if end_idx > len(self.infra_file_names):
            raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
        infra_file_names = self.infra_file_names[start_idx:end_idx]
        vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
        for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name)
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), self.cooperative_reader.get_cooperative_T_i2v()

    def generate_infra_vehicle_bboxes_object_list_pointcloud(self, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(self.infra_file_names)
        if start_idx < 0 or start_idx >= end_idx:
            raise ValueError('start_idx should be in [0, end_idx)')
        if end_idx > len(self.infra_file_names):
            raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
        infra_file_names = self.infra_file_names[start_idx:end_idx]
        vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
        for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name)
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()