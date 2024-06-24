import os.path as osp
from read_utils import read_yaml, read_json
import sys
sys.path.append('./process/corresponding')
from CorrespondingDetector import CorrespondingDetector
from CooperativeReader import CooperativeReader
from InfraReader import InfraReader
from VehicleReader import VehicleReader


class CooperativeBatchingReader:
    def __init__(self, yaml_filename = 'config.yml', path_data_info = f'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/data_info.json'):
        para_yaml = read_yaml(yaml_filename)
        self.data_folder = para_yaml['data_folder']
        self.path_data_info = path_data_info
        
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

    @staticmethod
    def rescore_boxes_object_list(boxes_object_list, matches_with_distance, distance_threshold_between_last_frame):
        boxes_score_dict = {}
        for match, distance in matches_with_distance.items():
            distance = -distance
            score = 0.5                    
            # score = 1 if 0 < distance < distance_threshold_between_last_frame
            # score = -x + 1 + threshold if distance_threshold_between_last_frame < distance < distance_threshold_between_last_frame + 1
            # score = 0 if distance > distance_threshold_between_last_frame + 1
            if distance < distance_threshold_between_last_frame:
                score = 1
            # elif distance < distance_threshold_between_last_frame + 1:
            #     score = -distance + 1 + distance_threshold_between_last_frame
            else:
                score = 0
            boxes_score_dict[match[0]] = score

        rescored_boxes_object_list = []
        for i, box_object in enumerate(boxes_object_list):
            if i in boxes_score_dict.keys():
                score = boxes_score_dict[i]
            else:
                score = 0
            rescored_box_objct = box_object.copy()
            rescored_box_objct.set_confidence(score)
            rescored_boxes_object_list.append(rescored_box_objct)
        return rescored_boxes_object_list


    def generate_infra_vehicle_bboxes_object_list_static_according_last_frame(self, start_idx=0, end_idx=-1, distance_threshold_between_last_frame = 1):
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

            last_infra_file_name = f'{int(infra_file_name) - 1:06}'
            last_vehicle_file_name = f'{int(vehicle_file_name) - 1:06}'
            if last_infra_file_name not in self.infra_file_names or last_vehicle_file_name not in self.vehicle_file_names:
                continue

            last_infra_boxes_object_list = InfraReader(last_infra_file_name).get_infra_boxes_object_list()
            infra_boxes_object_list = InfraReader(infra_file_name).get_infra_boxes_object_list()
            infra_matches_with_distance = CorrespondingDetector(infra_boxes_object_list, last_infra_boxes_object_list).get_matches_with_score()
            rescored_infra_boxes_object_list = CooperativeBatchingReader.rescore_boxes_object_list(infra_boxes_object_list, infra_matches_with_distance, distance_threshold_between_last_frame)
            
            last_vehicle_boxes_object_list = VehicleReader(last_vehicle_file_name).get_vehicle_boxes_object_list()
            vehicle_boxes_object_list = VehicleReader(vehicle_file_name).get_vehicle_boxes_object_list()
            vehicle_matches_with_distance = CorrespondingDetector(vehicle_boxes_object_list, last_vehicle_boxes_object_list).get_matches_with_score()
            rescored_vehicle_boxes_object_list = CooperativeBatchingReader.rescore_boxes_object_list(vehicle_boxes_object_list, vehicle_matches_with_distance, distance_threshold_between_last_frame)

            yield infra_file_name, vehicle_file_name, rescored_infra_boxes_object_list, rescored_vehicle_boxes_object_list, self.cooperative_reader.get_cooperative_T_i2v()


    def generate_infra_vehicle_bboxes_object_list_predicted(self, start_idx=0, end_idx=-1):
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
            try:
                yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted(), self.cooperative_reader.get_cooperative_T_i2v()
            except FileNotFoundError as e:
                print(f'error: {infra_file_name}, {vehicle_file_name}')           
                print(e)
                print('-------------------')

    def generate_infra_vehicle_bboxes_object_list_cooperative_fusioned(self, start_idx=0, end_idx=-1):
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
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_cooperative_fusioned(), self.cooperative_reader.get_cooperative_T_i2v()


    def generate_infra_vehicle_bboxes_object_list_predicted_pointcloud(self, start_idx=0, end_idx=-1):
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
            try:
                yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()
            except FileNotFoundError as e:
                print(f'error: {infra_file_name}, {vehicle_file_name}')           
                print(e)
                print('-------------------')

    def generate_infra_vehicle_bboxes_object_list_cooperative_fusioned_and_ego_true_label_pointcloud(self, start_idx=0, end_idx=-1):
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
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_cooperative_fusioned(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()

    def generate_infra_vehicle_bboxes_object_list_predicted_and_true_label_pointcloud(self, start_idx=0, end_idx=-1):
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
            try:
                yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(),*self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()
            except FileNotFoundError as e:
                print(f'error: {infra_file_name}, {vehicle_file_name}')           
                print(e)
                print('-------------------')

    def generate_infra_vehicle_pointcloud(self, start_idx=0, end_idx=-1):
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
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()

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

    def generate_infra_vehicle_image(self, start_idx=0, end_idx=-1):
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
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_image()

    def generate_infra_vehicle_bboxes_object_list_image(self, start_idx=0, end_idx=-1):
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
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_image(), self.cooperative_reader.get_cooperative_T_i2v()

    def generate_infra_vehicle_bboxes_object_list_pointcloud_image(self, start_idx=0, end_idx=-1):
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
            yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), *self.cooperative_reader.get_cooperative_infra_vehicle_image(), self.cooperative_reader.get_cooperative_T_i2v()
