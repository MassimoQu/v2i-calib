import numpy as np
import open3d as o3d
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from v2x_calib.search import Matches2Extrinsics
from v2x_calib.reader import V2XSim_Reader
from v2x_calib.reader import CooperativeReader
from v2x_calib.reader import CooperativeBatchingReader
from v2x_calib.corresponding import BoxesMatch
from v2x_calib.utils import implement_T_3dbox_object_list, implement_T_points_n_3, get_reverse_T, convert_6DOF_to_T, get_RE_TE_by_compare_T_6DOF_result_true, convert_T_to_6DOF
from v2x_calib.preprocess import Filter3dBoxes

class BBoxVisualizer_open3d():

    def __init__(self) -> None:
        pass

    def draw_box3d_open3d(self, box3d, color=(1, 0, 0)):
        """
        用 Open3D 绘制 3D 框。
        """
        lines = []
        colors = [color for _ in range(12)] # 每个框需要12条线
        # 4个底部点
        bottom_points = [0, 1, 2, 3, 0]
        for i in range(4):
            lines.append([bottom_points[i], bottom_points[i + 1]])
        
        # 4个顶部点
        top_points = [4, 5, 6, 7, 4]
        for i in range(4):
            lines.append([top_points[i], top_points[i + 1]])
        
        # 从底部到顶部的4条线
        for i in range(4):
            lines.append([bottom_points[i], top_points[i]])
        lines = np.array(lines, dtype=np.int32)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack(box3d)),
            lines=o3d.utility.Vector2iVector(np.array(lines)),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        return line_set

    def plot_boxes3d_pointcloud(self, boxes3d_object_list, pointcloud):
        self.plot_boxes3d_lists_pointcloud_lists([boxes3d_object_list], [pointcloud], [[(1, 0, 0)], [(0, 1, 0)]])

    def plot_boxes3d_lists_pointcloud_lists(self, boxes_lists, pointclouds_list, boxes_color_list, win_name = 'point cloud with bbox'):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=win_name)
        # 绘制点云
        pointcloud_colors = [(0.5, 0, 0), (0, 0.5, 0)]
        # pointcloud_colors = [(0, 0.25, 0), (0.25, 0, 0)]
        
        # pointcloud_colors = [(1, 0, 0), (0, 1, 0)]

        for i, pointcloud in enumerate(pointclouds_list):
            # print(pointcloud.shape)  # Should output (n, 3)
            # print(pointcloud.dtype)  # Ideally should be np.float64 or np.float32
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            num_points = len(pcd.points)
            colors = np.tile(pointcloud_colors[i%2], (num_points, 1))
            pcd.colors = o3d.utility.Vector3dVector(colors)

            vis.add_geometry(pcd)
            # 获取渲染选项并设置点大小
            opt = vis.get_render_option()
            opt.point_size = 1.0  # 可以调整这个值来改变点的大小

        for color_, boxes_list in zip(boxes_color_list, boxes_lists):
            for box3d_object in boxes_list:
                box3d = self.draw_box3d_open3d(box3d_object.get_bbox3d_8_3(), color=color_)
                vis.add_geometry(box3d)

        vis.run()
        vis.destroy_window()

    def plot_boxes_8_3_list(self, boxes_8_3_list, color_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for color_, boxes_8_3 in zip(color_list, boxes_8_3_list):
            box3d = self.draw_box3d_open3d(boxes_8_3, color=color_)
            vis.add_geometry(box3d)
        vis.run()
        vis.destroy_window()

    def plot_boxes3d_lists(self, boxes_lists, color_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for color_, boxes_list in zip(color_list, boxes_lists):
            for box3d_object in boxes_list:
                box3d = self.draw_box3d_open3d(box3d_object.get_bbox3d_8_3(), color=color_)
                vis.add_geometry(box3d)
        vis.run()  # 开始事件循环
        vis.destroy_window()

    def visualize_specific_type_boxes_object_within_infra_vehicle_boxes_object_list(self, boxes_object_list, pointcloud_list, specific_type = 'car', colors_list = [[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]]):
        specific_boxes_lists = []
        specific_boxes_lists.append(boxes_object_list[0])
        specific_boxes_lists.append(boxes_object_list[1])
        specific_boxes_lists.append(Filter3dBoxes(boxes_object_list[0]).filter_according_to_category(specific_type))
        specific_boxes_lists.append(Filter3dBoxes(boxes_object_list[1]).filter_according_to_category(specific_type))
        self.plot_boxes3d_lists_pointcloud_lists(specific_boxes_lists, pointcloud_list, colors_list)


def test_alpha_property():
    reader = CooperativeBatchingReader('config.yml')
    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, infra_pointcloud, vehicle_pointcloud, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list():
        converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_infra2vehicle, infra_boxes_object_list)
        converted_infra_pointcloud = implement_T_points_n_3(T_infra2vehicle, infra_pointcloud)
        boxes_color_list = [[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]]
        BBoxVisualizer_open3d().visualize_specific_type_boxes_object_within_infra_vehicle_boxes_object_list([converted_infra_boxes_object_list, vehicle_boxes_object_list], [converted_infra_pointcloud, vehicle_pointcloud], specific_type='car', colors_list=boxes_color_list)
        print('infra_file_name: ', infra_file_name)
        print('vehicle_file_name: ', vehicle_file_name)
        print('infra_alpha: ', infra_boxes_object_list[0].alpha)


def test_original_dataset(infra_file_name, vehicle_file_name, k = 15):
    cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name)
    # infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted()
    infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
    infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud()

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_topK_confidence(k=k)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_topK_confidence(k=k)

    # [-4.48010435e+01 -3.54835729e+01 -1.94283870e-01 -2.35885590e-14  2.54444375e-14  4.05479517e+01]
    # T_6DOF = np.array([-4.48010435e+01, -3.54835729e+01, -1.94283870e-01, -2.35885590e-14,  2.54444375e-14,  4.05479517e+01])
    # T = convert_6DOF_to_T(T_6DOF)
    T = cooperative_reader.get_cooperative_T_i2v()
    converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, filtered_infra_boxes_object_list)
    converted_infra_pointcloud = implement_T_points_n_3(T, infra_pointcloud)

    BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([converted_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)])

def test_predicted_dataset(infra_file_name, vehicle_file_name, k = 15):
    cooperative_reader = CooperativeReader(infra_file_name, vehicle_file_name)
    infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list_predicted()
    infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud()

    filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_topK_confidence(k=k)
    filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_topK_confidence(k=k)

    # [-4.48010435e+01 -3.54835729e+01 -1.94283870e-01 -2.35885590e-14  2.54444375e-14  4.05479517e+01]
    # T_6DOF = np.array([-4.48010435e+01, -3.54835729e+01, -1.94283870e-01, -2.35885590e-14,  2.54444375e-14,  4.05479517e+01])
    # T = convert_6DOF_to_T(T_6DOF)
    T = cooperative_reader.get_cooperative_T_i2v()
    infra_boxes_object_list = implement_T_3dbox_object_list(get_reverse_T(T), infra_boxes_object_list)
    converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, filtered_infra_boxes_object_list)
    converted_infra_pointcloud = implement_T_points_n_3(T, infra_pointcloud)

    print('len(filtered_infra_boxes_object_list): ', len(filtered_infra_boxes_object_list))
    print('len(filtered_vehicle_boxes_object_list): ', len(filtered_vehicle_boxes_object_list))

    BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list], [converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)])

if '__main__' == __name__:
    # cooperative_reader = CooperativeReader('005298', '001374')
    # converted_infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    # converted_infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()

    # boxes_color_list = [[1, 0, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]]
    # BBoxVisualizer_open3d().visualize_specific_type_boxes_object_within_infra_vehicle_boxes_object_list([converted_infra_boxes_object_list, vehicle_boxes_object_list], [converted_infra_pointcloud, vehicle_pointcloud], specific_type='car', colors_list=boxes_color_list)

    # test_alpha_property()

    # test_original_dataset('007038', '000546', 100)
    # test_predicted_dataset('007038', '000546', 100)

    # reader = CooperativeReader('006782', '000102')

    # infra_pointcloud, vehicle_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()
    # T_i2v = reader.get_cooperative_T_i2v()

    # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([],[infra_pointcloud, vehicle_pointcloud], [])

    
    # bbox3d_list = Reader().get_3dbbox_object_list("000010.json")
    # pointcloud = Reader().get_pointcloud("000010.pcd")
    # BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([bbox3d_list], [pointcloud], [(1, 0, 0)])

    scene = [0]

    reader = V2XSim_Reader() 

    core_similarity_component = ['centerpoint_distance','vertex_distance']
    matches_filter_strategy = 'threshold'

    for frame_idx, cav_id, bbox3d_object_list_lidar1, bbox3d_object_list_lidar2, pointcloud1, pointcloud2, T_lidar2_lidar1 in reader.generate_vehicle_vehicle_bboxes_object_list_pointcloud(noise={'pos_std':1, 'rot_std':0, 'pos_mean':0, 'rot_mean':0}):
        
        if frame_idx not in scene:
            continue

        print(f"frame_idx: {frame_idx}, cav_id: {cav_id}")

        matches_with_score_list = BoxesMatch(bbox3d_object_list_lidar1, bbox3d_object_list_lidar2, core_similarity_component=core_similarity_component).get_matches_with_score()
        T_calculated = Matches2Extrinsics(bbox3d_object_list_lidar1, bbox3d_object_list_lidar2, matches_score_list = matches_with_score_list).get_combined_extrinsic(matches_filter_strategy = matches_filter_strategy)

        RE, TE = get_RE_TE_by_compare_T_6DOF_result_true(convert_T_to_6DOF(T_lidar2_lidar1), T_calculated)

        print(f'RE: {RE}, TE: {TE}')

        bbox3d_object_list_lidar1_lidar2 = implement_T_3dbox_object_list(convert_6DOF_to_T(T_calculated), bbox3d_object_list_lidar1)
        pointcloud1_lidar2 = implement_T_points_n_3(convert_6DOF_to_T(T_calculated), pointcloud1[:,:3])

        bbox3d_object_list_lidar1_true = implement_T_3dbox_object_list(T_lidar2_lidar1, bbox3d_object_list_lidar1)
        pointcloud1_true = implement_T_points_n_3(T_lidar2_lidar1, pointcloud1[:,:3])

        BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([bbox3d_object_list_lidar1_lidar2, bbox3d_object_list_lidar2], [pointcloud1_lidar2, pointcloud2[:,:3]], [(1, 0, 0), (0, 1, 0)])
        BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([bbox3d_object_list_lidar1_true, bbox3d_object_list_lidar2], [pointcloud1_true, pointcloud2[:,:3]], [(1, 0, 0), (0, 1, 0)])
        break

