import mayavi.mlab as mlab
import numpy as np
import cv2

import sys
sys.path.append('./task')
sys.path.append('./reader')
# print(sys.path)
from Reader import Reader
from PointCloudVisualizer import PointCloudVisualizer
from vis_utils import get_cam_8_points, vis_label_in_img



class BBoxVisualizer():
    
    def __init__(self) -> None:
        self.reader = Reader('config.yml')

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

    def draw_2dboxes_image(self, image, boxes2d, color = (255, 255, 0)):
        for box in boxes2d:
            box = box[0]
            cv2.rectangle(image, ((box[0]).astype(np.int32), (box[1].astype(np.int32))), ((box[2]).astype(np.int32), (box[3]).astype(np.int32)), color, 2)
        return image

    def plot_boxes3d_lists(self, boxes_lists, color_list):
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        for color_, boxes_list in zip(color_list, boxes_lists):
            for boxes in boxes_list.values():
                self.draw_boxes3d(np.array(boxes), fig, color=color_)
        mlab.show()

    def plot_boxes3d_lists_according_to_precision(self, boxes_lists, color_lists, precision_list):
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        for color_, boxes_list, precision in zip(color_lists, boxes_lists, precision_list):
            for boxes, precision_level in zip(boxes_list.values(), precision):
                denomitor = 1
                if 2 in precision_level:
                    denomitor = 10
                elif 1 in precision_level:
                    denomitor = 5
                self.draw_boxes3d(np.array(boxes), fig, color=tuple(val / denomitor for val in color_))
        mlab.show()

    def plot_boxes3d_pointcloud(self, boxes3d, pointcloud):
        fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 500))
        pointcloud_visualizer = PointCloudVisualizer()
        pointcloud_visualizer.draw_pointclouds(fig, [pointcloud])
        for boxes in boxes3d.values():
            self.draw_boxes3d(np.array(boxes), fig)
        mlab.show()

    def plot_3dboxes_image(self, _3dboxes_list_n_7, path_lidar2camera, path_image, path_camera_intrinsic, color=(0, 255, 0)):
        img = vis_label_in_img(get_cam_8_points(_3dboxes_list_n_7, path_lidar2camera), path_image, path_camera_intrinsic, color=color)
        cv2.imshow('3dboxes_image', img)
        cv2.waitKey(0)

    

    def plot_boxes_2d3d_image(self, _3dboxes_list_n_7, boxes_2d_dict, path_lidar2camera, path_image, path_camera_intrinsic, color_list = [(0, 255, 0), (255, 0, 0)]):
        img = vis_label_in_img(get_cam_8_points(_3dboxes_list_n_7, path_lidar2camera), path_image, path_camera_intrinsic, color=color_list[0])

        self.draw_2dboxes_image(img, boxes_2d_dict.values(), color=color_list[1])

        cv2.imshow('3dboxes_image', img)
        cv2.waitKey(0)



if __name__ == "__main__":
    bbox_visualizer = BBoxVisualizer()
