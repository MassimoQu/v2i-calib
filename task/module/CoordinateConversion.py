import numpy as np
import sys
sys.path.append(r'E:\WorkSpace\vehicle-infrastructure-cooperation\vehicle_infrastructure_cooperation_normalized_code\reader')
sys.path.append(r'E:\WorkSpace\vehicle-infrastructure-cooperation\vehicle_infrastructure_cooperation_normalized_code\visualize')
from VehicleReader import VehicleReader
from InfraReader import InfraReader
from CooperativeReader import CooperativeReader
from PointCloudVisualizer import PointCloudVisualizer

from utils import get_reverse_R_t, multiply_extrinsics, implement_T_3dbox_dict_n_8_3, implement_T_points_n_3, get_reverse_T, convert_Rt_to_T

class CoordinateConversion():
    def __init__(self):
        self.infra_reader = InfraReader('config.yml')
        self.vehicle_reader = VehicleReader('config.yml')
        self.cooperative_reader = CooperativeReader('config.yml')

    # def convert_bboxes_3d_2d(self, _3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic, color=(0, 255, 0)):
    #     img = vis_label_in_img(get_cam_8_points(_3dboxes_list_n_7, path_lidar2camera), path_image, path_camera_intrinsic, color=color)




    def convert_bboxes_lidar_2_image(self, bboxes_dict):
        # lidar_2_camera -> camera_2_image
        


        pass

    # multiply extrisincs #tofix
    # use cooperative/calib/lidar_i2v #todo
    def convert_point_3_n_infra_lidar_2_vehicle_lidar(self, points):
        # infra_lidar_2_world -> world_2_vehicle_novatel(rev) -> vehicle_novatel_2_vehicle_lidar(rev)
        
        '''
        # T_infra_lidar_2_world = convert_Rt_to_T(*self.infra_reader.get_infra_virtuallidar2world())
        # T_world_2_vehicle_novatel = get_reverse_T(convert_Rt_to_T(*self.vehicle_reader.get_vehicle_novatel2world()))
        # T_vehicle_novatel_2_vehicle_lidar = get_reverse_T(convert_Rt_to_T(*self.vehicle_reader.get_lidar2novatel()))
        
        # convert1
        # T_infra_lidar_2_vehicle_novatel = multiply_extrinsics(T_infra_lidar_2_world, T_world_2_vehicle_novatel)
        # T_infra_lidar_2_vehicle_lidar = multiply_extrinsics(T_infra_lidar_2_vehicle_novatel, T_vehicle_novatel_2_vehicle_lidar)

        # return implement_T_points_n_3(T_infra_lidar_2_vehicle_lidar, points)

        # convert2
        # points = implement_T_points_n_3(T_infra_lidar_2_world, points)
        # points = implement_T_points_n_3(T_world_2_vehicle_novatel, points)
        # points = implement_T_points_n_3(T_vehicle_novatel_2_vehicle_lidar, points)
        # return points
        '''

        # using cooperative/calib/lidar_i2v directly
        T_infra_lidar_2_vehicle_lidar = convert_Rt_to_T(*self.cooperative_reader.get_cooperative_lidar_i2v())
        return implement_T_points_n_3(T_infra_lidar_2_vehicle_lidar, points)


        



    def convert_bboxes_dict_n_8_3_infra_lidar_2_vehicle_image(self, bboxes_dict):
        # infra_lidar_2_world -> world_2_vehicle_novatel(rev) -> vehicle_novatel_2_vehicle_lidar(rev) -> vehicle_lidar_2_vehicle_camera -> vehicle_camera_2_vehicle_image        
        # convert_bboxes_infra_lidar_2_vehicle_lidar -> convert_bboxes_lidar_2_image (vehicle)

        self.convert_point_3_n_infra_lidar_2_vehicle_lidar(bboxes_dict)
        T_vehicle_lidar_2_vehicle_camera = convert_Rt_to_T(self.vehicle_reader.get_vehicle_lidar2camera())
        bboxes_dict = implement_T_3dbox_dict_n_8_3(T_vehicle_lidar_2_vehicle_camera, bboxes_dict)
        return self.convert_bboxes_lidar_2_image(bboxes_dict)



if __name__ == '__main__':
    conversion = CoordinateConversion()
    infra_pointcloud = conversion.infra_reader.get_infra_pointcloud()
    vehicle_pointcloud = conversion.vehicle_reader.get_vehicle_pointcloud()

    infra_pointcloud = conversion.convert_point_3_n_infra_lidar_2_vehicle_lidar(infra_pointcloud)
    pointcloud_visualizer = PointCloudVisualizer()
    pointcloud_visualizer.plot_pointclouds([infra_pointcloud, vehicle_pointcloud], [(0, 1, 0), (1, 0, 0)])

