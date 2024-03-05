import sys
sys.path.append('./visualize')
from CooperativeBatchingReader import CooperativeBatchingReader
from BBoxVisualizer_open3d import BBoxVisualizer_open3d
from extrinsic_utils import implement_T_3dbox_object_list, implement_T_points_n_3
import cv2

def put_text_on_image(image, text, position=(10, 30), font_scale=0.7, color=(255, 255, 255)):
    """在图像上添加文本"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    cv2.putText(image, text, position, font, font_scale, color, thickness, cv2.LINE_AA)


def show_images_side_by_side():
    # 创建一个可调整大小的窗口
    cv2.namedWindow('Combined Image', cv2.WINDOW_NORMAL)
    # 设置窗口的初始大小，例如宽1200像素，高600像素
    cv2.resizeWindow('Combined Image', 2400, 1200)

    for infra_file_name, vehicle_file_name, infra_image, vehicle_image in CooperativeBatchingReader().generate_infra_vehicle_image():
        
        put_text_on_image(infra_image, f'Infra: {infra_file_name}')
        put_text_on_image(vehicle_image, f'Vehicle: {vehicle_file_name}')

        
        # 调整图像高度以匹配
        if infra_image.shape[0] != vehicle_image.shape[0]:
            target_height = max(infra_image.shape[0], vehicle_image.shape[0])
            infra_image = cv2.resize(infra_image, (int(infra_image.shape[1] * target_height / infra_image.shape[0]), target_height), interpolation=cv2.INTER_AREA)
            vehicle_image = cv2.resize(vehicle_image, (int(vehicle_image.shape[1] * target_height / vehicle_image.shape[0]), target_height), interpolation=cv2.INTER_AREA)

        # 水平拼接图像
        combined_image = cv2.hconcat([infra_image, vehicle_image])

        # 如果拼接后的图像宽度大于窗口宽度，则等比例缩放图像以适应窗口
        if combined_image.shape[1] > 2400:
            scale_factor = 2400 / combined_image.shape[1]
            combined_image = cv2.resize(combined_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

        # 显示拼接后的图像
        cv2.imshow('Combined Image', combined_image)

        # 等待用户按键，如果用户按任意键，继续显示下一组图像
        key = cv2.waitKey(0)  # 0 表示无限等待直到用户按下一个键
        if key == 27:  # 如果按下ESC，则退出循环
            break

        # 关闭当前显示的窗口
        cv2.destroyAllWindows()

def find_cooperation_example():
    reader = CooperativeBatchingReader(path_data_info='/home/massimo/vehicle_infrastructure_calibration/dataset_division/hard_data_info.json')
    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, infra_pointcloud, vehicle_pointcloud, T_true in reader.generate_infra_vehicle_bboxes_object_list_pointcloud():
        
        print(f'Infra: {infra_file_name}, Vehicle: {vehicle_file_name}')

        converted_infra_boxes_object_list = implement_T_3dbox_object_list(T_true, infra_boxes_object_list)
        converted_infra_pointcloud = implement_T_points_n_3(T_true, infra_pointcloud)

        BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([converted_infra_boxes_object_list, vehicle_boxes_object_list], [converted_infra_pointcloud, vehicle_pointcloud], [(1, 0, 0), (0, 1, 0)])
        
        key = cv2.waitKey(0)  # 0 表示无限等待直到用户按下一个键
        if key == 27:
            break

        cv2.destroyAllWindows()


# 调用函数显示图像
# show_images_side_by_side()
        
find_cooperation_example()