import os.path as osp
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from ..utils import implement_R_t_points_n_3, convert_Rt_to_T
import numpy as np
from .read_utils import read_json
from .InfraReader import InfraReader
from .VehicleReader import VehicleReader


class CooperativeReader():
    def __init__(self, infra_file_name = '003920', vehicle_file_name = '020092', data_folder = './data/DAIR-V2X'):
        self.infra_reader = InfraReader(infra_file_name, data_folder)
        self.vehicle_reader = VehicleReader(vehicle_file_name, data_folder)

    def parse_cooperative_lidar_i2v(self):
        return osp.join(self.vehicle_reader.data_folder, 'cooperative', 'calib', 'lidar_i2v', self.vehicle_reader.vehicle_file_name + '.json')
    
    def parse_cooperative_camera_i2v(self):
        return osp.join(self.vehicle_reader.data_folder, 'cooperative', 'calib', 'camera_i2v', self.vehicle_reader.vehicle_file_name + '.json')

    def _compute_lidar_i2v_from_individual(self):
        infra_R, infra_t = self.infra_reader.get_infra_virtuallidar2world()
        veh_R, veh_t = self.vehicle_reader.get_vehicle_novatel2world()
        lidar2novatel_R, lidar2novatel_t = self.vehicle_reader.get_lidar2novatel()

        T_infra = convert_Rt_to_T(infra_R, infra_t)
        T_vehicle = convert_Rt_to_T(veh_R, veh_t)
        T_lidar2novatel = convert_Rt_to_T(lidar2novatel_R, lidar2novatel_t)
        T_vehicle_lidar2world = T_vehicle @ T_lidar2novatel
        T_i2v = np.linalg.inv(T_vehicle_lidar2world) @ T_infra
        rotation = T_i2v[:3, :3].tolist()
        translation = [[T_i2v[0, 3]], [T_i2v[1, 3]], [T_i2v[2, 3]]]
        return rotation, translation

    def _compute_lidar_i2v_from_individual_unadjusted(self):
        infra_R, infra_t = self.infra_reader.get_infra_virtuallidar2world_unadjusted()
        veh_R, veh_t = self.vehicle_reader.get_vehicle_novatel2world()
        lidar2novatel_R, lidar2novatel_t = self.vehicle_reader.get_lidar2novatel()

        T_infra = convert_Rt_to_T(infra_R, infra_t)
        T_vehicle = convert_Rt_to_T(veh_R, veh_t)
        T_lidar2novatel = convert_Rt_to_T(lidar2novatel_R, lidar2novatel_t)
        T_vehicle_lidar2world = T_vehicle @ T_lidar2novatel
        T_i2v = np.linalg.inv(T_vehicle_lidar2world) @ T_infra
        rotation = T_i2v[:3, :3].tolist()
        translation = [[T_i2v[0, 3]], [T_i2v[1, 3]], [T_i2v[2, 3]]]
        return rotation, translation

    def get_cooperative_lidar_Rt_i2v(self):
        path = self.parse_cooperative_lidar_i2v()
        if Path(path).exists():
            lidar_i2v = read_json(path)
            rotation = lidar_i2v["rotation"]
            translation = lidar_i2v["translation"]
            return rotation, translation
        return self._compute_lidar_i2v_from_individual()

    def get_cooperative_lidar_Rt_i2v_unadjusted(self):
        return self._compute_lidar_i2v_from_individual_unadjusted()
    
    def get_cooperative_camera_Rt_i2v(self):
        path = self.parse_cooperative_camera_i2v()
        if Path(path).exists():
            camera_i2v = read_json(path)
            rotation = camera_i2v["rotation"]
            translation = camera_i2v["translation"]
            return rotation, translation
        return self._compute_camera_i2v_from_lidar_i2v()

    def _compute_camera_i2v_from_lidar_i2v(self):
        """
        Derive camera extrinsic (infra camera -> vehicle camera) from:
          - LiDAR i2v (infra LiDAR -> vehicle LiDAR)
          - Per-agent LiDAR->camera extrinsics
        """
        T_lidar_i2v = self.get_cooperative_T_i2v()
        T_inf_lidar2cam, T_veh_lidar2cam = self.get_infra_vehicle_lidar2camera()
        if T_inf_lidar2cam is None or T_veh_lidar2cam is None:
            raise FileNotFoundError('missing lidar2camera calib for camera_i2v fallback')
        T_cam_i2v = np.asarray(T_veh_lidar2cam) @ np.asarray(T_lidar_i2v) @ np.linalg.inv(
            np.asarray(T_inf_lidar2cam)
        )
        rotation = T_cam_i2v[:3, :3].tolist()
        translation = [[T_cam_i2v[0, 3]], [T_cam_i2v[1, 3]], [T_cam_i2v[2, 3]]]
        return rotation, translation

    def get_cooperative_T_i2v(self):
        return convert_Rt_to_T(*self.get_cooperative_lidar_Rt_i2v())

    def get_cooperative_T_i2v_unadjusted(self):
        return convert_Rt_to_T(*self.get_cooperative_lidar_Rt_i2v_unadjusted())
    
    def get_cooperative_camera_T_i2v(self):
        return convert_Rt_to_T(*self.get_cooperative_camera_Rt_i2v())

    def get_infra_vehicle_lidar2camera(self):
        return self.infra_reader.get_infra_lidar2camera(), self.vehicle_reader.get_vehicle_lidar2camera()

    def get_cooperative_infra_vehicle_boxes_object_list(self):
        return self.infra_reader.get_infra_boxes_object_list(), self.vehicle_reader.get_vehicle_boxes_object_list()
    
    def get_cooperative_infra_vehicle_boxes_object_list_predicted(self):
        return self.infra_reader.get_infra_boxes_object_list_predicted(), self.vehicle_reader.get_vehicle_boxes_object_list_predicted()
    
    def get_cooperative_infra_vehicle_boxes_object_list_cooperative_fusioned(self):
        return self.infra_reader.get_infra_boxes_object_list(), self.vehicle_reader.get_vehicle_boxes_object_list_cooperative_fusioned()

    def get_cooperative_infra_vehicle_pointcloud(self):
        return self.infra_reader.get_infra_pointcloud(), self.vehicle_reader.get_vehicle_pointcloud()
    
    def get_cooperative_infra_vehicle_image(self):
        return self.infra_reader.get_infra_image(), self.vehicle_reader.get_vehicle_image()
    
    def get_infra_vehicle_camera_instrinsics(self):
        return self.infra_reader.get_infra_intrinsic(), self.vehicle_reader.get_vehicle_intrinsic()

    def get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate(self):
        infra_pointcloud, vehicle_pointcloud = self.get_cooperative_infra_vehicle_pointcloud()
        R_infra_lidar_2_vehicle_lidar, t_infra_lidar_2_vehicle_lidar = self.get_cooperative_lidar_Rt_i2v()

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
    

    
