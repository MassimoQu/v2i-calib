import open3d as o3d
import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./task/module')
from CooperativeReader import CooperativeReader
from utils import convert_6DOF_to_T, implement_T_points_n_3



class PoincloudVisualizer_open3d():
    def __init__(self):
        pass

    def plot_pointclouds(self, pointclouds, colors):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        # 绘制点云
        for pointcloud, color in zip(pointclouds, colors):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pointcloud)
            num_points = len(pcd.points)
            colors = np.repeat([color], num_points, axis=0)  # 创建一个红色的颜色数组
            pcd.colors = o3d.utility.Vector3dVector(colors)  # 将颜色数组分配给点云
            vis.add_geometry(pcd)
            # 获取渲染选项并设置点大小
            opt = vis.get_render_option()
            opt.point_size = 2.0

        vis.run()
        vis.destroy_window()
        

if __name__ == '__main__':
    cooperative_reader = CooperativeReader('config.yml')
    # infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()
    infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud()

    
    # [-48.23108959  -2.16604824  -1.1209898   -0.24683178  -0.7557314   3.14159265]
    T = convert_6DOF_to_T([-48.23108959, -2.16604824, -1.1209898, -0.24683178, -0.7557314, 3.14159265])
    infra_pointcloud = implement_T_points_n_3(T, infra_pointcloud)  

    pointclouds = [infra_pointcloud, vehicle_pointcloud]
    colors = [(1, 0, 0), (0, 1, 0)]
    PoincloudVisualizer_open3d().plot_pointclouds(pointclouds, colors)

