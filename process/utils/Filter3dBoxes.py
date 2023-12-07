import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./visualize')
from CooperativeReader import CooperativeReader

class Filter3dBoxes():
    def __init__(self, boxes_object_list=None):
        self.boxes_object_list = boxes_object_list


    def filter_according_to_occlusion(self, degree = 0):
        filtered_boxes_object_list = []
        for box_object in self.boxes_object_list:
            if box_object.occluded_state <= degree:
                filtered_boxes_object_list.append(box_object)
        return filtered_boxes_object_list
    
    def filter_according_to_truncation(self, degree = 0):
        filtered_boxes_object_list = []
        for box_object in self.boxes_object_list:
            if box_object.truncated_state <= degree:
                filtered_boxes_object_list.append(box_object)
        return filtered_boxes_object_list
    
    def filter_according_to_occlusion_truncation(self, occlusion_degree = 0, truncation_degree = 0):
        occulussion_filtered_boxes_object_list = self.filter_according_to_occlusion(self.boxes_object_list, occlusion_degree)
        truncation_filtered_boxes_object_list = self.filter_according_to_truncation(occulussion_filtered_boxes_object_list, truncation_degree)
        return truncation_filtered_boxes_object_list
    
    def filter_according_to_distance(self, distance = 80):
        filtered_boxes_object_list = []
        for box_object in self.boxes_object_list:
            centroid = box_object.get_bbox3d_8_3().mean(axis=0)
            centroid_xy = centroid[:2]
            centroid_dist = np.linalg.norm(centroid_xy)
            if centroid_dist <= distance:
                filtered_boxes_object_list.append(box_object)

        return filtered_boxes_object_list
    
    def filter_according_to_size_topK(self, k = 10):
        volume_list = []
        for box_object in self.boxes_object_list:
            box3d = box_object.get_bbox3d_8_3()
            box_size = np.abs(box3d[4] - box3d[2])
            volume = box_size[0] * box_size[1] * box_size[2]
            volume_list.append(volume)
        k = min(k, len(volume_list))
        top_k_volumes = sorted(volume_list, reverse=True)[:k]
        filtered_boxes_object_list = [box_object for box_object, volume in zip(self.boxes_object_list, volume_list) if volume in top_k_volumes]
        return filtered_boxes_object_list
    
    def filter_according_to_size_percentile(self, percentile = 75):
        volume_list = []
        for box_object in self.boxes_object_list:
            box3d = box_object.get_bbox3d_8_3()
            box_size = np.abs(box3d[4] - box3d[2])
            volume = box_size[0, 0] * box_size[0, 1] * box_size[0, 2]
            volume_list.append(volume)
        percentile_volume = np.percentile(volume_list, percentile)
        filtered_boxes_object_list = [boxes_object for boxes_object, volume in zip(self.boxes_object_list, volume_list) if volume >= percentile_volume]
        return filtered_boxes_object_list

    def filter_according_to_category(self, category):
        filtered_boxes_object_list = []
        for box_object in self.boxes_object_list:
            if box_object.get_bbox_type().lower() == category.lower():
                filtered_boxes_object_list.append(box_object)
        return filtered_boxes_object_list

    # 参数局限，谨慎使用
    def filter_according_to_size_distance_occlusion_truncation(self, distance = 80, occlusion_degree = 1, truncation_degree = 1, topk = 10):
        distance_filted_boxes_object_list = self.filter_according_to_distance(self.boxes_object_list, distance)
        occlusion_truncation_filted_boxes_object_list = self.filter_according_to_occlusion_truncation(distance_filted_boxes_object_list, occlusion_degree, truncation_degree)
        size_filted_boxes_object_list = self.filter_according_to_size_topK(occlusion_truncation_filted_boxes_object_list, topk)
        return size_filted_boxes_object_list

    def get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(self, infra_filtered_distance = 80, vehicle_filterd_distance = 45, occlusion_degree = 1, truncation_degree = 1, topk = 10):
        cooperative_reader = CooperativeReader()
        converted_infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
        filtered_infra_3dboxes = self.filter_according_to_size_distance_occlusion_truncation(converted_infra_boxes_object_list, infra_filtered_distance, occlusion_degree, truncation_degree, topk)
        filered_vehicle_3dboxes = self.filter_according_to_size_distance_occlusion_truncation(vehicle_boxes_object_list, vehicle_filterd_distance, occlusion_degree, truncation_degree, topk)
        return filtered_infra_3dboxes, filered_vehicle_3dboxes


    def test_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(self):
        cooperative_reader = CooperativeReader()
        converted_infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
        
        distance_filted_infra_boxes_object_list = Filter3dBoxes(converted_infra_boxes_object_list).filter_according_to_distance(80)
        distance_filted_vehicle_boxes_object_list =  Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_distance(45)
        print('distance filter:')
        print('infra_boxes: {} -> {} ', len(converted_infra_boxes_object_list), len(distance_filted_infra_boxes_object_list))
        print('vehicle_boxes: {} -> {} ', len(vehicle_boxes_object_list), len(distance_filted_vehicle_boxes_object_list))

        degree = 1
        occlusion_filted_infra_boxes_object_list = Filter3dBoxes(distance_filted_infra_boxes_object_list).filter_according_to_occlusion_truncation(degree, degree)
        occlusion_filted_vehicle_boxes_object_list = Filter3dBoxes(distance_filted_vehicle_boxes_object_list).filter_according_to_occlusion_truncation(degree, degree)
        print('occlusion filter:')
        print('infra_boxes: {} -> {} ', len(distance_filted_infra_boxes_object_list), len(occlusion_filted_infra_boxes_object_list))
        print('vehicle_boxes: {} -> {} ', len(distance_filted_vehicle_boxes_object_list), len(occlusion_filted_vehicle_boxes_object_list))

        size_filted_infra_boxes_object_list = Filter3dBoxes(occlusion_filted_infra_boxes_object_list).filter_according_to_size_percentile()
        size_filted_vehicle_boxes_object_list = Filter3dBoxes(occlusion_filted_vehicle_boxes_object_list).filter_according_to_size_percentile()
        print('size filter:')
        print('infra_boxes: {} -> {} ', len(occlusion_filted_infra_boxes_object_list), len(size_filted_infra_boxes_object_list))
        print('vehicle_boxes: {} -> {} ', len(occlusion_filted_vehicle_boxes_object_list), len(size_filted_vehicle_boxes_object_list))

        return size_filted_infra_boxes_object_list, size_filted_vehicle_boxes_object_list



if __name__ == "__main__":
    cooperative_reader = CooperativeReader()
    converted_infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()

    filtered_infra_3dboxes, filered_vehicle_3dboxes = Filter3dBoxes().test_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation()

    boxes_color_list = [[1, 0, 0], [0, 1, 0]]
    from BBoxVisualizer_open3d import BBoxVisualizer_open3d
    BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([filtered_infra_3dboxes, filered_vehicle_3dboxes], [converted_infra_pointcloud, vehicle_pointcloud], boxes_color_list)
