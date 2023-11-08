import os.path as osp
import cv2
from Reader import Reader


class InfraReader():
    def __init__(self, yaml_filename):
        self.reader = Reader(yaml_filename)


    def parse_infra_intrinsic_path(self):
        return osp.join(self.reader.folder_root, 'infrastructure-side', 'calib', 'camera_intrinsic', self.reader.infra_file_name + '.json')

    def parse_infra_virtuallidar2camera_path(self):
        return osp.join(self.reader.folder_root, 'infrastructure-side', 'calib', 'virtuallidar_to_camera', self.reader.infra_file_name + '.json')
    
    def parse_infra_virtuallidar2world_path(self):
        return osp.join(self.reader.folder_root, 'infrastructure-side', 'calib', 'virtuallidar_to_world', self.reader.infra_file_name + '.json')
    
    def parse_infra_image_path(self):
        folder_infra_image = self.reader.para_yaml['boxes']['folder_infra_images']
        return osp.join(folder_infra_image, self.reader.infra_file_name + '.jpg')

    def parse_infra_pointcloud_path(self):
        folder_infra_pointcloud = self.reader.para_yaml['boxes']['folder_infra_pointcloud']
        return osp.join(folder_infra_pointcloud, self.reader.infra_file_name + '.pcd')

    def parse_infra_label_path(self):
        return osp.join(self.reader.folder_root, 'infrastructure-side', 'label', 'virtuallidar', self.reader.infra_file_name + '.json')
    

    def get_infra_image(self):
        return cv2.imread(self.parse_infra_image_path())

    def get_infra_pointcloud(self):
        return self.reader.get_pointcloud(self.parse_infra_pointcloud_path())    

    def get_infra_boxes_dict(self, high_precision_constraint_flag=False):
        return self.reader.get_3dboxes_dict_n_8_3(self.parse_infra_label_path(), high_precision_constraint_flag=high_precision_constraint_flag)
    
    def get_infra_boxes_2d_dict(self):
        return self.reader.get_2dbbox_dict_n_4(self.parse_infra_label_path())

    def get_infra_boxes_list_n_7(self):
        return self.reader.get_3dboxes_list_n_7(self.parse_infra_label_path())

    def get_infra_occluded_truncated_state_list(self):
        return self.reader.get_occluded_truncated_state_list(self.parse_infra_label_path())

    def get_infra_intrinsic(self):
        return self.reader.get_intrinsic(self.parse_infra_intrinsic_path())
    
    def get_infra_virtuallidar2world(self):
        virtuallidar2world = self.reader.read_json(self.parse_infra_virtuallidar2world_path())
        rotation = virtuallidar2world["rotation"]
        translation = virtuallidar2world["translation"]
        translation[0][0] += virtuallidar2world["relative_error"]["delta_x"]
        translation[1][0] += virtuallidar2world["relative_error"]["delta_y"]
        return rotation, translation

    
    
if __name__ == "__main__":
    reader = InfraReader('config.yml')
    print(reader.get_infra_virtuallidar2world())