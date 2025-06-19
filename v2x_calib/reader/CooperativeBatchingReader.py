from .read_utils import read_yaml, read_json
from .CooperativeReader import CooperativeReader


class CooperativeBatchingReader:
    def __init__(self, path_data_info = f'./data/DAIR-V2X/cooperative/data_info.json', path_data_folder = None):
        self.path_data_info = path_data_info
        self.path_data_folder = path_data_folder
        if path_data_folder is None:
            self.path_data_folder = '/'.join(path_data_info.split('/')[:-2])
        
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
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
            inf_bbox_object_list, veh_bbox_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
            yield infra_file_name, vehicle_file_name, inf_bbox_object_list, veh_bbox_object_list, self.cooperative_reader.get_cooperative_T_i2v()

    def generate_infra_vehicle_2dbboxes_object_list(self, start_idx=0, end_idx=-1):
        if end_idx == -1:
            end_idx = len(self.infra_file_names)
        if start_idx < 0 or start_idx >= end_idx:
            raise ValueError('start_idx should be in [0, end_idx)')
        if end_idx > len(self.infra_file_names):
            raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')

        infra_file_names = self.infra_file_names[start_idx:end_idx]
        vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
        for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
            inf_bbox_object_list, veh_bbox_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
            yield infra_file_name, vehicle_file_name, inf_bbox_object_list, veh_bbox_object_list, self.cooperative_reader.get_cooperative_camera_T_i2v()

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
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
            inf_bbox_object_list, veh_bbox_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted()
            try:
                yield infra_file_name, vehicle_file_name, inf_bbox_object_list, veh_bbox_object_list, self.cooperative_reader.get_cooperative_T_i2v()
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
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
            inf_bbox_object_list, veh_bbox_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_cooperative_fusioned()
            yield infra_file_name, vehicle_file_name, inf_bbox_object_list, veh_bbox_object_list, self.cooperative_reader.get_cooperative_T_i2v()


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
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
            inf_bbox_object_list, veh_bbox_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted()
            inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
            try:
                yield infra_file_name, vehicle_file_name, inf_bbox_object_list, veh_bbox_object_list, inf_pointcloud, veh_pointcloud, self.cooperative_reader.get_cooperative_T_i2v()
            except FileNotFoundError as e:
                print(f'error: {infra_file_name}, {vehicle_file_name}')           
                print(e)
                print('-------------------')

    # def generate_infra_vehicle_bboxes_object_list_cooperative_fusioned_and_ego_true_label_pointcloud(self, start_idx=0, end_idx=-1):
    #     if end_idx == -1:
    #         end_idx = len(self.infra_file_names)
    #     if start_idx < 0 or start_idx >= end_idx:
    #         raise ValueError('start_idx should be in [0, end_idx)')
    #     if end_idx > len(self.infra_file_names):
    #         raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
    #     infra_file_names = self.infra_file_names[start_idx:end_idx]
    #     vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
    #     for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
    #         self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
    #         inf_bbox_object_list, veh_bbox_object_list = self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_cooperative_fusioned()
    #         yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_cooperative_fusioned(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()

    # def generate_infra_vehicle_bboxes_object_list_predicted_and_true_label_pointcloud(self, start_idx=0, end_idx=-1):
    #     if end_idx == -1:
    #         end_idx = len(self.infra_file_names)
    #     if start_idx < 0 or start_idx >= end_idx:
    #         raise ValueError('start_idx should be in [0, end_idx)')
    #     if end_idx > len(self.infra_file_names):
    #         raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
    #     infra_file_names = self.infra_file_names[start_idx:end_idx]
    #     vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
    #     for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
    #         self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
    #         inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
    #         try:
    #             yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(),*self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()
    #         except FileNotFoundError as e:
    #             print(f'error: {infra_file_name}, {vehicle_file_name}')           
    #             print(e)
    #             print('-------------------')

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
            self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
            try:
                inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
                yield infra_file_name, vehicle_file_name, inf_pointcloud, veh_pointcloud, self.cooperative_reader.get_cooperative_T_i2v()
            except FileNotFoundError as e:
                print(f'error: {infra_file_name}, {vehicle_file_name}')           
                print(e)
                print('-------------------')

    # def generate_infra_vehicle_bboxes_object_list_pointcloud(self, start_idx=0, end_idx=-1):
    #     if end_idx == -1:
    #         end_idx = len(self.infra_file_names)
    #     if start_idx < 0 or start_idx >= end_idx:
    #         raise ValueError('start_idx should be in [0, end_idx)')
    #     if end_idx > len(self.infra_file_names):
    #         raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
    #     infra_file_names = self.infra_file_names[start_idx:end_idx]
    #     vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
    #     for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
    #         self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
    #         inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
    #         yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), self.cooperative_reader.get_cooperative_T_i2v()

    # def generate_infra_vehicle_image(self, start_idx=0, end_idx=-1):
    #     if end_idx == -1:
    #         end_idx = len(self.infra_file_names)
    #     if start_idx < 0 or start_idx >= end_idx:
    #         raise ValueError('start_idx should be in [0, end_idx)')
    #     if end_idx > len(self.infra_file_names):
    #         raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
    #     infra_file_names = self.infra_file_names[start_idx:end_idx]
    #     vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
    #     for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
    #         self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
    #         inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
    #         yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_image()

    # def generate_infra_vehicle_bboxes_object_list_image(self, start_idx=0, end_idx=-1):
    #     if end_idx == -1:
    #         end_idx = len(self.infra_file_names)
    #     if start_idx < 0 or start_idx >= end_idx:
    #         raise ValueError('start_idx should be in [0, end_idx)')
    #     if end_idx > len(self.infra_file_names):
    #         raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
    #     infra_file_names = self.infra_file_names[start_idx:end_idx]
    #     vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
    #     for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
    #         self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
    #         inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
    #         yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_image(), self.cooperative_reader.get_cooperative_T_i2v()

    # def generate_infra_vehicle_bboxes_object_list_pointcloud_image(self, start_idx=0, end_idx=-1):
    #     if end_idx == -1:
    #         end_idx = len(self.infra_file_names)
    #     if start_idx < 0 or start_idx >= end_idx:
    #         raise ValueError('start_idx should be in [0, end_idx)')
    #     if end_idx > len(self.infra_file_names):
    #         raise ValueError('end_idx should be in [start_idx, len(infra_file_names)]')
        
    #     infra_file_names = self.infra_file_names[start_idx:end_idx]
    #     vehicle_file_names = self.vehicle_file_names[start_idx:end_idx]
    #     for infra_file_name, vehicle_file_name in zip(infra_file_names, vehicle_file_names):
    #         self.cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name, self.path_data_folder)
    #         inf_pointcloud, veh_pointcloud = self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud()
    #         yield infra_file_name, vehicle_file_name, *self.cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list(), *self.cooperative_reader.get_cooperative_infra_vehicle_pointcloud(), *self.cooperative_reader.get_cooperative_infra_vehicle_image(), self.cooperative_reader.get_cooperative_T_i2v()
