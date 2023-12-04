import os.path as osp
from read_utils import read_yaml, read_json
from CooperativeReader import CooperativeReader


class CooperativeBatchingReader:
    def __init__(self, yaml_filename):
        para_yaml = read_yaml(yaml_filename)
        self.data_folder = para_yaml['data_folder']
        path_data_info = osp.join(self.data_folder, 'cooperative-vehicle-infrastructure', 'cooperative', 'data_info.json')
        
        self.infra_file_names, self.vehicle_file_names = self.get_infra_vehicle_file_names(path_data_info)

    
    def get_infra_vehicle_file_names(self, path_data_info):
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
        for infra_file_name, vehicle_file_name in zip(self.infra_file_names, self.vehicle_file_names):
            self.cooperative_reader = CooperativeReader(self.data_folder, infra_file_name, vehicle_file_name)
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_bboxes_object_list()

    
