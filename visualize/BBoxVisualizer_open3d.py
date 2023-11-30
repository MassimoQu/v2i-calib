import numpy as np
import open3d as o3d
import sys
sys.path.append('./reader')
sys.path.append('./process')
from CooperativeReader import CooperativeReader
from BBoxVisualizer import BBoxVisualizer
from Filter3dBoxes import Filter3dBoxes



class BBoxVisualizer_open3d(BBoxVisualizer):

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
        self.plot_boxes3d_lists_pointcloud_lists([boxes3d_object_list], [pointcloud], [(1, 0, 0)], [(0, 1, 0)])

    def plot_boxes3d_lists_pointcloud_lists(self, boxes_lists, pointclouds_list, boxes_color_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # 绘制点云
        for pointcloud in pointclouds_list:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            num_points = len(pcd.points)
            colors = np.random.rand(num_points, 3)
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

    def plot_boxes3d_lists(self, boxes_lists, color_list):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for color_, boxes_list in zip(color_list, boxes_lists):
            for box3d_object in boxes_list:
                box3d = self.draw_box3d_open3d(box3d_object.get_bbox3d_8_3(), color=color_)
                vis.add_geometry(box3d)
        vis.run()  # 开始事件循环
        vis.destroy_window()


if '__main__' == __name__:
    cooperative_reader = CooperativeReader('config.yml')
    converted_infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes3d_object_lists_vehicle_coordinate()
    # converted_infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()

    # filetr_3dboxes = Filter3dBoxes()
    # degree = 2
    # converted_infra_boxes_object_list = filetr_3dboxes.filter_according_to_occlusion_truncation(converted_infra_boxes_object_list, degree, degree)
    # vehicle_boxes_object_list = filetr_3dboxes.filter_according_to_occlusion_truncation(vehicle_boxes_object_list, degree, degree)

    # filter3dBoxes = Filter3dBoxes()
    # converted_infra_boxes_object_list, vehicle_boxes_object_list = filter3dBoxes.get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(topk=20)

    boxes_color_list = [[1, 0, 0], [0, 1, 0]]
    BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([converted_infra_boxes_object_list, vehicle_boxes_object_list], [], boxes_color_list)

