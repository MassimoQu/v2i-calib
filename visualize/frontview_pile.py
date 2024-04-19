
# 画出车端点云投射到车端图像上的可视化
# 路端点云转化到车端坐标系再投影到车端图像上

import numpy as np
import cv2
import sys 
sys.path.append('./reader')
sys.path.append('./process/utils')
from CooperativeReader import CooperativeReader
from CooperativeBatchingReader import CooperativeBatchingReader
from extrinsic_utils import implement_T_points_n_3


def project_points_to_image(points, intrinsics, image, boxes_object_list = None):
    """
    :param points: 3D points in the world coordinate system
    :param intrinsics: camera calibration matrix
    :param image: cv2 image
    :return: 2D points in the image coordinate system
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image passed to heterogeneious projection function.")

    # Homogeneous coordinates
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    # Project points
    points_projected = intrinsics @ points_homogeneous.T
    # Convert to non-homogeneous coordinates (x, y, z) -> (x/z, y/z)
    points_2d = points_projected[:2, :] / points_projected[2, :]
    # Convert points to integers for image coordinates
    points_2d = points_2d.T[:, :2].astype(np.int32)

    result_img = image.copy()

    # Draw points on the image
    for p in points_2d:
        if 0 <= p[0] < result_img.shape[1] and 0 <= p[1] < result_img.shape[0]:
            if boxes_object_list == None:
                cv2.circle(result_img, tuple(p), 2, (0, 255, 0), -1)
                continue
            if len(boxes_object_list) > 0:
                for box_object in boxes_object_list:
                    box2d = box_object.get_bbox2d_4()
                    if box2d[0] <= p[0] <= box2d[2] and \
                            box2d[1] <= p[1] <= box2d[3]:
                        cv2.circle(result_img, tuple(p), 2, (0, 255, 0), -1)

    return result_img


def filter_points_outside_3dboxes(points, boxes_object_list):
    """
    Filter points that are outside of 3D boxes
    :param points: 3D points
    :param boxes_object_list: list of 3D BBox objects
    :return: 3D points that are inside of the 3D boxes
    """
    points_inside = []
    for box_object in boxes_object_list:
        box3d = box_object.get_bbox3d_8_3()
        for point in points:
            if box3d[2][0] <= point[0] <= box3d[4][0] and \
                    box3d[2][1] <= point[1] <= box3d[4][1] and \
                    box3d[2][2] <= point[2] <= box3d[4][2]:
                points_inside.append(point)
        box2d = box_object.get_bbox2d_4()
        for point in points:
            if box2d[0] <= point[0] <= box2d[2] and \
                    box2d[1] <= point[1] <= box2d[3]:
                points_inside.append(point)
    return np.array(points_inside)


def separate_front_view(infra_id, veh_id):
    reader = CooperativeReader(infra_id, veh_id)
    
    inf_image, veh_image = reader.get_cooperative_infra_vehicle_image()
    inf_pointcloud, veh_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()
    
    infra_boxes_object_list, veh_boxes_object_list = reader.get_cooperative_infra_vehicle_boxes_object_list()
    filtered_infra_3d_objects_pointcloud = filter_points_outside_3dboxes(inf_pointcloud, infra_boxes_object_list)
    filtered_veh_3d_objects_pointcloud = filter_points_outside_3dboxes(veh_pointcloud, veh_boxes_object_list)

    inf_instrinsics, veh_instrinsics = reader.get_infra_vehicle_camera_instrinsics()
    T_inf_lidar2cam, T_veh_lidar2cam = reader.get_infra_vehicle_lidar2camera()

    filtered_infra_3d_objects_pointcloud_inf_cam_coord = implement_T_points_n_3(T_inf_lidar2cam, filtered_infra_3d_objects_pointcloud)
    filtered_vehi_3d_objects_pointcloud_veh_cam_coord = implement_T_points_n_3(T_veh_lidar2cam, filtered_veh_3d_objects_pointcloud)
    projected_sameside_inf_image = project_points_to_image(filtered_infra_3d_objects_pointcloud_inf_cam_coord, inf_instrinsics, inf_image, infra_boxes_object_list)
    projected_sameside_veh_image = project_points_to_image(filtered_vehi_3d_objects_pointcloud_veh_cam_coord, veh_instrinsics, veh_image, veh_boxes_object_list)

    cv2.imshow('Projected Same Side Inf Image', projected_sameside_inf_image)
    cv2.imshow('Projected Same Side Veh Image', projected_sameside_veh_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_text_to_image(image, text, position=(50, 50)):
    """ Helper function to add text to an image """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # White
    line_type = 2
    cv2.putText(image, text, position, font, font_scale, font_color, line_type)
    return image

def combine_inf_veh_pointcloud_to_veh_image(infra_id, veh_id):
    reader = CooperativeReader(infra_id, veh_id)
    
    inf_image, veh_image = reader.get_cooperative_infra_vehicle_image()
    inf_pointcloud, veh_pointcloud = reader.get_cooperative_infra_vehicle_pointcloud()
    
    infra_boxes_object_list, veh_boxes_object_list = reader.get_cooperative_infra_vehicle_boxes_object_list()
    filtered_infra_3d_objects_pointcloud = filter_points_outside_3dboxes(inf_pointcloud, infra_boxes_object_list)
    filtered_veh_3d_objects_pointcloud = filter_points_outside_3dboxes(veh_pointcloud, veh_boxes_object_list)

    T_i2v = reader.get_cooperative_T_i2v()
    inf_instrinsics, veh_instrinsics = reader.get_infra_vehicle_camera_instrinsics()
    T_inf_lidar2cam, T_veh_lidar2cam = reader.get_infra_vehicle_lidar2camera()

    filtered_infra_3d_objects_pointcloud_veh_cam_coord = implement_T_points_n_3(T_veh_lidar2cam @ T_i2v, filtered_infra_3d_objects_pointcloud)
    filtered_veh_3d_objects_pointcloud_veh_cam_coord = implement_T_points_n_3(T_veh_lidar2cam, filtered_veh_3d_objects_pointcloud)
    combined_inf_veh_3d_objects_pointcloud_veh_cam_coord = np.concatenate((filtered_infra_3d_objects_pointcloud_veh_cam_coord, filtered_veh_3d_objects_pointcloud_veh_cam_coord))
    
    # print(filtered_infra_3d_objects_pointcloud_veh_cam_coord.shape)
    # print(filtered_veh_3d_objects_pointcloud_veh_cam_coord.shape)
    # print(combined_inf_veh_3d_objects_pointcloud_veh_cam_coord.shape)

    projected_inf_pointcloud_to_veh_image = project_points_to_image(filtered_infra_3d_objects_pointcloud_veh_cam_coord, veh_instrinsics, veh_image, veh_boxes_object_list)
    projected_veh_pointlcloud_veh_image = project_points_to_image(filtered_veh_3d_objects_pointcloud_veh_cam_coord, veh_instrinsics, veh_image, veh_boxes_object_list)
    combined_inf_veh_pointcloud_veh_image = project_points_to_image(combined_inf_veh_3d_objects_pointcloud_veh_cam_coord, veh_instrinsics, veh_image, veh_boxes_object_list)

    add_text_to_image(projected_inf_pointcloud_to_veh_image, 'Infra PC to Vehicle Image')
    add_text_to_image(projected_veh_pointlcloud_veh_image, 'Vehicle PC to Vehicle Image')
    add_text_to_image(combined_inf_veh_pointcloud_veh_image, 'Combined PC to Vehicle Image')

    combined_image = np.vstack((projected_inf_pointcloud_to_veh_image, projected_veh_pointlcloud_veh_image, combined_inf_veh_pointcloud_veh_image))
    window_name = 'Combined Pointclouds to Vehicle Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Use WINDOW_NORMAL to allow resizing
    cv2.imshow(window_name, combined_image)
    cv2.resizeWindow(window_name, 800, 1200)  # Adjust the window size

    # cv2.imshow('Projected Infra Pointcloud to Vehicle Image', projected_inf_pointcloud_to_veh_image)
    # cv2.imshow('Projected Vehicle Pointcloud to Vehicle Image', projected_veh_pointlcloud_veh_image)
    # cv2.imshow('Combined Infra and Vehicle Pointcloud to Vehicle Image', combined_inf_veh_pointcloud_veh_image)

    if cv2.waitKey(0) & 0xFF == ord('q'):  # Wait for 'q' key to quit
        cv2.destroyAllWindows()
        return False  # Signal to stop
    cv2.destroyAllWindows()
    return True


if __name__ == '__main__':
    
    # "007489", "000289"
    #  "005298", "001374"
    #  "006782", "000102"
    # "005295", "001372"
    # "016844", "010839"
    # "014442", "011617"
    # "005633", "001953"

    inf_name_list, veh_name_list = CooperativeBatchingReader(path_data_info='/home/massimo/vehicle_infrastructure_calibration/dataset_division/easy_data_info.json').get_infra_vehicle_file_names()

    for inf_id, veh_id in zip(inf_name_list, veh_name_list):
        print(inf_id, veh_id)
        if not combine_inf_veh_pointcloud_to_veh_image(inf_id, veh_id):
            print("Exiting...")
            break  # Exit if combine_inf_veh_pointcloud_to_veh_image returns False
        print("---------------------------------------------------------")
