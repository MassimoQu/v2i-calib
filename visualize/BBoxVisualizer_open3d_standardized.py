import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui # type: ignore
import matplotlib.cm as cm
import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
from CooperativeReader import CooperativeReader
from VehicleReader import VehicleReader
from Reader import Reader
from CooperativeBatchingReader import CooperativeBatchingReader
from Filter3dBoxes import Filter3dBoxes
from extrinsic_utils import implement_T_points_n_3, implement_T_3dbox_object_list, get_reverse_T
from extrinsic_utils import convert_6DOF_to_T


class BBoxVisualizer_open3d_standardized():
    def __init__(self, window_name = 'visualize', vis_names = [], vis_num = 1) -> None:
        self.app = gui.Application.instance
        self.app.initialize()
        
        # self.window = self.app.create_window("Open3D - Two Pointclouds", 2048, 768)
        self.vis = []

        for i in range(vis_num):
            vname = f'vis{i}'
            if i < len(vis_names):
                vname = vis_names[i]
            self.vis.append(o3d.visualization.O3DVisualizer(vname, 1024, 768))

    def draw_box3d_with_text(self, box3d, color=(1, 0, 0), text='', text_position='center', vis_id=0):
        """
        使用 Open3D 绘制 3D 框，并在框的中心添加一个文本标签。
        """
        lines = []
        colors = [color for _ in range(12)]  
        bottom_points = [0, 1, 2, 3, 0]
        for i in range(4):
            lines.append([bottom_points[i], bottom_points[i + 1]])

        top_points = [4, 5, 6, 7, 4]
        for i in range(4):
            lines.append([top_points[i], top_points[i + 1]])

        for i in range(4):
            lines.append([bottom_points[i], top_points[i]])
        lines = np.array(lines, dtype=np.int32)
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.vstack(box3d)),
            lines=o3d.utility.Vector2iVector(np.array(lines)),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)

        if text_position == 'center': 
            position = np.mean(box3d, axis=0)
        elif text_position == 'left_bottom':
            position = box3d[0]

        if vis_id >= len(self.vis):
            print('Error: vis_id out of range during boxes drawing!')
            return
        
        self.vis[vis_id].add_geometry(text, line_set)
        self.vis[vis_id].add_3d_label(position, text)
        # label.color = gui.Color(color[0], color[1], color[2])


    def draw_pointcloud(self, pointcloud, color=(1, 0, 0), name='PointCloud', vis_id=0):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd.paint_uniform_color(color)
        if vis_id >= len(self.vis):
            print('Error: vis_id out of range during pointcloud drawing!')
            return
        self.vis[vis_id].add_geometry(name, pcd)

    def visualize_matches_under_certain_scene(self, boxes_object_lists, pointclouds, matches_with_score, vis_id=0, run_flag=True):
        infra_boxes_object_list, vehicle_boxes_object_list = boxes_object_lists[0], boxes_object_lists[1]
        
        if len(pointclouds) >= 2:
            infra_pointcloud, vehicle_pointcloud = pointclouds[0], pointclouds[1]
            self.draw_pointcloud(infra_pointcloud, color=(0.5, 0, 0), name='infra_pointcloud', vis_id=vis_id)
            self.draw_pointcloud(vehicle_pointcloud, color=(0, 0.5, 0), name='vehicle_pointcloud', vis_id=vis_id)

        icnt, vcnt = 0, 0

        infra_matched_indices = [match[0] for match in matches_with_score.keys()]
        vehicle_matched_indices = [match[1] for match in matches_with_score.keys()]

        for infra_box_object in infra_boxes_object_list:
            if icnt not in infra_matched_indices:
                infra_box = infra_box_object.get_bbox3d_8_3()
                self.draw_box3d_with_text(infra_box, color=(1, 0, 0), text=f'i{icnt}', vis_id=vis_id)
            icnt+=1

        for vehicle_box_object in vehicle_boxes_object_list:
            if vcnt not in vehicle_matched_indices:
                vehicle_box = vehicle_box_object.get_bbox3d_8_3()
                self.draw_box3d_with_text(vehicle_box, color=(0, 1, 0), text=f'v{vcnt}', vis_id=vis_id)
            vcnt+=1

        cnt = 0
        for match, score in matches_with_score.items():
            
            infra_box = infra_boxes_object_list[match[0]].get_bbox3d_8_3()
            vehicle_box = vehicle_boxes_object_list[match[1]].get_bbox3d_8_3()

            self.draw_box3d_with_text(infra_box, color=(1, 0, 0), text=f'im{cnt} :{"{:.3f}".format(score)}', text_position='left_bottom', vis_id=vis_id)
            self.draw_box3d_with_text(vehicle_box, color=(0, 1, 0), text=f'vm{cnt} :{"{:.3f}".format(score)}', text_position='left_bottom', vis_id=vis_id)
            cnt += 1

        self.vis[vis_id].reset_camera_to_default()
        self.app.add_window(self.vis[vis_id])

        if run_flag:
            self.app.run()

 
    def visualize_matches_under_dual_true_predicted_scene(self, boxes_object_lists_true, boxes_object_list_predicted, pointclouds_true, pointclouds_predicted, matches_with_score_true, matches_with_score_predicted):
        self.visualize_matches_under_certain_scene(boxes_object_lists_true, pointclouds_true, matches_with_score_true, vis_id=0, run_flag=False)
        self.visualize_matches_under_certain_scene(boxes_object_list_predicted, pointclouds_predicted, matches_with_score_predicted, vis_id=1, run_flag=False)
        
        # Set up layouts
        # layout = gui.Horiz(0.5)  # Use a horizontal layout with a spacing of 0.5
        
        # layout.add_child(self.vis[0])
        # layout.add_child(self.vis[1])

        # self.window.add_child(layout)

        self.app.run()
        

if __name__ == '__main__':
    
    # LIBGL_ALWAYS_SOFTWARE=1

    # reader = VehicleReader('000001')
    # vehicle_object_list = reader.get_vehicle_boxes_object_list()
    # vehicle_pointcloud = reader.get_vehicle_pointcloud()

    # BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([[], vehicle_object_list], [vehicle_pointcloud, []], {}, vis_id=0)

    bbox3d_list = Reader().get_3dbbox_object_list("000010.json")
    pointcloud = Reader().get_pointcloud("000010.bin")
    BBoxVisualizer_open3d_standardized().visualize_matches_under_certain_scene([[], bbox3d_list], [pointcloud, []], {}, vis_id=0)
