import sys
import time

sys.path.append('./reader')
sys.path.append('./process/utils')

from extrinsic_utils import get_reverse_T, implement_T_3dbox_object_list, implement_T_points_n_3
from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from BBoxVisualizer_open3d import BBoxVisualizer_open3d

import open3d as o3d
import numpy as np


class ConsecutiveVisualizer:
    def __init__(self, inf_start_id=-1, veh_start_id=-1, len=-1):
        self.bbox_visualizer = BBoxVisualizer_open3d()
        if inf_start_id > 0 and veh_start_id > 0 and len > 0:
            self.visualize_file_sequence()
        else:
            self.visualize_specify(int(inf_start_id), int(veh_start_id), len)

    def visualize_frame(self, vis, infra_pointcloud, vehicle_pointcloud, infra_bboxes_object_list, vehicle_bboxes_object_list):
        
        def get_pointcloud_open3d(vis, pointcloud, color):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            num_points = len(pcd.points)
            colors = np.tile(color, (num_points, 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(pcd)
            # 获取渲染选项并设置点大小
            opt = vis.get_render_option()
            opt.point_size = 1.0  # 可以调整这个值来改变点的大小
            return pcd

        infra_pcd = get_pointcloud_open3d(vis, infra_pointcloud, (0.5, 0, 0))
        vehicle_pcd = get_pointcloud_open3d(vis, vehicle_pointcloud, (0, 0.5, 0))

        vis.update_geometry(infra_pcd)
        vis.update_geometry(vehicle_pcd)

        for box3d_object in infra_bboxes_object_list:
            box3d = self.bbox_visualizer.draw_box3d_open3d(box3d_object.get_bbox3d_8_3(), color=(1, 0, 0))
            vis.add_geometry(box3d)

        for box3d_object in vehicle_bboxes_object_list:
            box3d = self.bbox_visualizer.draw_box3d_open3d(box3d_object.get_bbox3d_8_3(), color=(0, 1, 0))
            vis.add_geometry(box3d)
            
        return infra_pcd, vehicle_pcd
        
    def visualize_file_sequence(self):
        # 初始化视窗
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.update_renderer()

        cnt = 0

        for infra_file_name, vehicle_file_name, infra_bboxes_object_list, vehicle_bboxes_object_list, infra_pointcloud, vehicle_pointcloud, T_i2v in CooperativeBatchingReader(path_data_info=r'/home/massimo/vehicle_infrastructure_calibration/data/cooperative-vehicle-infrastructure/cooperative/sorted_data_info.json').generate_infra_vehicle_bboxes_object_list_pointcloud(start_idx=0, end_idx=-1):
            
            # if not vis.poll_events():
            #     break
            
            try:

                view_control: o3d.visualization.ViewControl = vis.get_view_control()
                view_control.set_front([ -0.16369650945088704, 0.0082033463888920785, -0.98647663829490639 ])
                view_control.set_lookat([ -0.5126953125, 0.0146484375, 1.3552540928219587 ])
                view_control.set_up([ -0.14927274718300235, 0.98824562139584249, 0.032988463746963764 ])
                view_control.set_zoom(0.1)



                T_v2i = get_reverse_T(T_i2v)
                # converted_infra_pointcloud = implement_T_points_n_3(T_i2v, infra_pointcloud)
                # converted_infra_bbox_object_list = implement_T_3dbox_object_list(T_i2v, infra_bboxes_object_list)
                # self.visualize_frame(vis, converted_infra_pointcloud, vehicle_pointcloud, converted_infra_bbox_object_list, vehicle_bboxes_object_list)
                
                converted_vehicle_pointcloud = implement_T_points_n_3(T_v2i, vehicle_pointcloud)
                converted_vehicle_bbox_object_list = implement_T_3dbox_object_list(T_v2i, vehicle_bboxes_object_list)
                self.visualize_frame(vis, infra_pointcloud, converted_vehicle_pointcloud, infra_bboxes_object_list, converted_vehicle_bbox_object_list)

                # 更新视窗以显示当前帧
                vis.poll_events()
                vis.update_renderer()
                
                # 设置视窗参数
                # ctr = vis.get_view_control()

                # ctr.set_zoom(0.1)

                # cam_params = ctr.convert_to_pinhole_camera_parameters()

                # 设置相机的位置和朝向以实现逆时针旋转90度并放大三倍
                # cam_params.extrinsic = np.array([
                #     [0, -1, 0, 0],  # 逆时针旋转90度
                #     [1, 0, 0, 0],
                #     [0, 0, 0.333, 0],
                #     [0, 0, 0, 1]
                # ])

                # 根据你的需求调整相机位置以放大视图
                # 这里示例仅修改 z 轴上的位置
                # cam_params.extrinsic[2, 3] = cam_params.extrinsic[2, 3] / 3

                # ctr.convert_from_pinhole_camera_parameters(cam_params)

                


                print(cnt, infra_file_name, vehicle_file_name)
                # print(infra_file_name)

                cnt += 1

            except Exception as e:
                print(e)
                continue

            # # 简单的延时来模拟视频流播放
            time.sleep(0.1)

            # clear the visualizer
            vis.clear_geometries()

            # stop when press esc

        # 关闭视窗
        vis.destroy_window()



    # 一边读一遍处理
    def visualize_specify(self, inf_start_id, veh_start_id, len):
        
        # 初始化视窗
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        cnt = 0

        for i in range(len):
            try:
                inf_id = inf_start_id + i
                veh_id = veh_start_id + i
                cooperative_reader = CooperativeReader(infra_file_name = str(inf_id).zfill(6), vehicle_file_name = str(veh_id).zfill(6))
                infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()
                infra_bboxes_object_list, vehicle_bboxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
                
                self.visualize_frame(vis, infra_pointcloud, vehicle_pointcloud, infra_bboxes_object_list, vehicle_bboxes_object_list)
                
                # 更新视窗以显示当前帧
                vis.poll_events()
                vis.update_renderer()
                
                print(cnt, inf_id, veh_id)

                cnt += 1

            except Exception as e:
                print(e)
                continue

            # 简单的延时来模拟视频流播放
            time.sleep(0.1)

            # clear the visualizer
            vis.clear_geometries()

        # 关闭视窗
        vis.destroy_window()


# ConsecutiveVisualizer(3920, 20092, 200).visualize()
ConsecutiveVisualizer().visualize_file_sequence()

