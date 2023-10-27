import mayavi.mlab as mlab
import numpy as np
import cv2

import Visualizer
import sys
sys.path.append(r'E:\WorkSpace\vehicle-infrastructure-cooperation\vehicle_infrastructure_cooperation_normalized_code\task')
# print(sys.path)
from Reader import Reader
from Visualizer import Visualizer
from vis_utils import get_cam_8_points, vis_label_in_img



class BBoxVisualizer(Visualizer):
    
    def __init__(self) -> None:
        self.reader = Reader('config.yml')
        self.infra_boxes_3d_dict = self.reader.get_infra_boxes_dict()
        self.vehicle_boxes_3d_dict = self.reader.get_vehicle_boxes_dict()
        # self.infra_boxes_2d = self.reader.get_infra_boxes_2d_dict()
        # self.vehicle_boxes_2d = self.reader.get_vehicle_boxes_2d_dict()

    def draw_boxes3d(
        self, boxes3d, fig, arrows=None, color=(1, 0, 0), line_width=2, draw_text=False, text_scale=(1, 1, 1), color_list=None
    ):
        """
        boxes3d: numpy array (n,8,3) for XYZs of the box corners
        fig: mayavi figure handler
        color: RGB value tuple in range (0,1), box line color
        line_width: box line width
        draw_text: boolean, if true, write box indices beside boxes
        text_scale: three number tuple
        color_list: RGB tuple
        """
        num = len(boxes3d)
        for n in range(num):
            if arrows is not None:
                mlab.plot3d(
                    arrows[n, :, 0],
                    arrows[n, :, 1],
                    arrows[n, :, 2],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig,
                )
            b = boxes3d[n]
            if color_list is not None:
                color = color_list[n]
            if draw_text:
                mlab.text3d(b[4, 0], b[4, 1], b[4, 2], "%d" % n, scale=text_scale, color=color, figure=fig)
            for k in range(0, 4):
                i, j = k, (k + 1) % 4
                mlab.plot3d(
                    [b[i, 0], b[j, 0]],
                    [b[i, 1], b[j, 1]],
                    [b[i, 2], b[j, 2]],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig,
                )

                i, j = k + 4, (k + 1) % 4 + 4
                mlab.plot3d(
                    [b[i, 0], b[j, 0]],
                    [b[i, 1], b[j, 1]],
                    [b[i, 2], b[j, 2]],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig,
                )

                i, j = k, k + 4
                mlab.plot3d(
                    [b[i, 0], b[j, 0]],
                    [b[i, 1], b[j, 1]],
                    [b[i, 2], b[j, 2]],
                    color=color,
                    tube_radius=None,
                    line_width=line_width,
                    figure=fig,
                )
        return fig

    def plot_boxes3d_lists(self, boxes_lists):
        color_list = [(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        for color_, boxes_list in zip(color_list, boxes_lists):
            for boxes in boxes_list.values():
                self.draw_boxes3d(np.array(boxes), fig, color=color_)
        mlab.show()

    def plot_boxes3d_infra(self):
        self.plot_boxes3d_lists([self.infra_boxes_3d_dict])

    def plot_boxes3d_vehicle(self):
        self.plot_boxes3d_lists([self.vehicle_boxes_3d_dict])

    def plot_boxes3d_pointcloud(self, boxes3d, pointcloud):
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        mlab.points3d(
            pointcloud[:, 0],
            pointcloud[:, 1],
            pointcloud[:, 2],
            pointcloud[:, 2],
            mode="point",
            colormap="spectral",
            figure=fig,
        )
        for boxes in boxes3d.values():
            self.draw_boxes3d(np.array(boxes), fig)
        mlab.show()


    def plot_boxes3d_pointcloud_infra(self):
        self.plot_boxes3d_pointcloud(self.infra_boxes_3d_dict, self.reader.get_infra_pointcloud())

    def plot_boxes3d_pointcloud_vehicle(self):
        self.plot_boxes3d_pointcloud(self.vehicle_boxes_3d_dict, self.reader.get_vehicle_pointcloud())
        

    def plot_3dboxes_pointcloud(self, path_3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic):
        camera_8_points_list = get_cam_8_points(path_3dboxes_list_n_7, path_lidar2camera)
        vis_label_in_img(camera_8_points_list, path_image, path_camera_intrinsic)

    def plot_3dboxes_vehicle_image(self):
        path_lidar2camera = self.reader.parse_vehicle_lidar2camera_path() 
        path_camera_intrinsic = self.reader.parse_vehicle_intrinsic_path()
        path_image = self.reader.parse_vehicle_image_path()
        path_3dboxes_list_n_7 = self.reader.get_vehicle_boxes_list_n_7()
        self.plot_3dboxes_pointcloud(path_3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic)

    def plot_3dboxes_infra_image(self):
        path_lidar2camera = self.reader.parse_infra_virtuallidar2camera_path()
        path_camera_intrinsic = self.reader.parse_infra_intrinsic_path()
        path_image = self.reader.parse_infra_image_path()
        path_3dboxes_list_n_7 = self.reader.get_infra_boxes_list_n_7()
        self.plot_3dboxes_pointcloud(path_3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic)


    def plot_2dboxes_image(self, image, boxes2d, color = (255, 255, 0)):
        for box in boxes2d:
            box = box[0]
            cv2.rectangle(image, ((box[0]).astype(np.int32), (box[1].astype(np.int32))), ((box[2]).astype(np.int32), (box[3]).astype(np.int32)), color, 2)
        cv2.imshow("2dboxes_image", image)
        cv2.waitKey(0)

    def plot_2dboxes_infra_image(self):
        self.plot_2dboxes_image(self.reader.get_infra_image(), self.reader.get_infra_boxes_2d_dict().values(), color=(0, 255, 0))

    def plot_2dboxes_vehicle_image(self):
        self.plot_2dboxes_image(self.reader.get_vehicle_image(), self.reader.get_vehicle_boxes_2d_dict().values(), color=(255, 0, 0))

    # def plot_boxes_2d3d_image(self, image, boxes2d, boxes3d, color_list = [(0, 255, 0), (255, 0, 0)]):
    #     self.plot_2dboxes_image(image, boxes2d, color_list[0])
    #     self.plot_3dboxes_image(image, boxes3d, color_list[1])

    def plot_boxes_2d3d_infra_image(self):
        pass
    

    def visualize(self):
        pass


if __name__ == "__main__":
    bbox_visualizer = BBoxVisualizer()
    # bbox_visualizer.plot_boxes3d_lists([bbox_visualizer.infra_boxes_3d_dict, bbox_visualizer.vehicle_boxes_3d_dict])#1
    # bbox_visualizer.plot_3dboxes_vehicle_image()#2
    # bbox_visualizer.plot_3dboxes_infra_image()#3
    # bbox_visualizer.plot_boxes3d_pointcloud_infra()#4
    # bbox_visualizer.plot_boxes3d_pointcloud_vehicle()#5
    bbox_visualizer.plot_2dboxes_infra_image()#6
    # bbox_visualizer.plot_2dboxes_vehicle_image()#7