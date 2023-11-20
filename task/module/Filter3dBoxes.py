import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./visualize')
sys.path.append('./task')
from CooperativeReader import CooperativeReader

class Filter3dBoxes():
    def __init__(self) -> None:
        pass

    def filter_according_to_occlusion(self, boxes_object_list, degree = 0):
        filtered_boxes_object_list = []
        for box_object in boxes_object_list:
            if box_object.occluded_state <= degree:
                filtered_boxes_object_list.append(box_object)
        return filtered_boxes_object_list
    
    def filter_according_to_truncation(self, boxes_object_list, degree = 0):
        filtered_boxes_object_list = []
        for box_object in boxes_object_list:
            if box_object.truncated_state <= degree:
                filtered_boxes_object_list.append(box_object)
        return filtered_boxes_object_list
    
    def filter_according_to_occlusion_truncation(self, boxes_object_list, occlusion_degree = 0, truncation_degree = 0):
        occulussion_filtered_boxes_object_list = self.filter_according_to_occlusion(boxes_object_list, occlusion_degree)
        truncation_filtered_boxes_object_list = self.filter_according_to_truncation(occulussion_filtered_boxes_object_list, truncation_degree)
        return truncation_filtered_boxes_object_list
         
    def filter_according_to_distance(self, boxes_object_list, distance = 80):
        filtered_boxes_object_list = []
        for box_object in boxes_object_list:
            centroid = box_object.get_bbox3d_8_3().mean(axis=0)
            centroid_xy = centroid[:2]
            centroid_dist = np.linalg.norm(centroid_xy)
            if centroid_dist <= distance:
                filtered_boxes_object_list.append(box_object)

        return filtered_boxes_object_list

    def filter_according_to_size(self, bboxes_3d_object_list):
        volume_list = []
        for box_object in bboxes_3d_object_list:
            box3d = box_object.get_bbox3d_8_3()
            box_size = np.abs(box3d[4] - box3d[2])
            volume = box_size[0, 0] * box_size[0, 1] * box_size[0, 2]
            volume_list.append(volume)
        median_volume = np.median(volume_list)
        upper_quartile_volume = np.percentile(volume_list, 75)
        filtered_boxes_object_list = [boxes_object for boxes_object, volume in zip(bboxes_3d_object_list, volume_list) if volume >= upper_quartile_volume]
        return filtered_boxes_object_list
    
    def filter_according_to_size_topK(self, bboxes_3d_object_list, k = 10):
        volume_list = []
        for box_object in bboxes_3d_object_list:
            box3d = box_object.get_bbox3d_8_3()
            box_size = np.abs(box3d[4] - box3d[2])
            volume = box_size[0, 0] * box_size[0, 1] * box_size[0, 2]
            volume_list.append(volume)
        k = min(k, len(volume_list))
        top_k_volumes = sorted(volume_list, reverse=True)[:k]
        filtered_boxes_object_list = [box_object for box_object, volume in zip(bboxes_3d_object_list, volume_list) if volume in top_k_volumes]
        return filtered_boxes_object_list
    


    def filter_according_to_size_distance_occlusion_truncation(self, boxes_object_list, distance = 80, occlusion_degree = 1, truncation_degree = 1, topk = 10):#转换前和转化后
        distance_filted_boxes_object_list = self.filter_according_to_distance(boxes_object_list, distance)
        occlusion_truncation_filted_boxes_object_list = self.filter_according_to_occlusion_truncation(distance_filted_boxes_object_list, occlusion_degree, truncation_degree)
        size_filted_boxes_object_list = self.filter_according_to_size_topK(occlusion_truncation_filted_boxes_object_list, topk)
        return size_filted_boxes_object_list

    def get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(self, infra_filtered_distance = 80, vehicle_filterd_distance = 45, occlusion_degree = 1, truncation_degree = 1, topk = 10):
        cooperative_reader = CooperativeReader('config.yml')
        converted_infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes3d_object_lists_vehicle_coordinate()
        filtered_infra_3dboxes = self.filter_according_to_size_distance_occlusion_truncation(converted_infra_boxes_object_list, infra_filtered_distance, occlusion_degree, truncation_degree, topk)
        filered_vehicle_3dboxes = self.filter_according_to_size_distance_occlusion_truncation(vehicle_boxes_object_list, vehicle_filterd_distance, occlusion_degree, truncation_degree, topk)
        return filtered_infra_3dboxes, filered_vehicle_3dboxes


    def test_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(self):
        cooperative_reader = CooperativeReader('config.yml')
        converted_infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes3d_object_lists_vehicle_coordinate()
        
        distance_filted_infra_boxes_object_list = Filter3dBoxes().filter_according_to_distance(converted_infra_boxes_object_list, 80)
        distance_filted_vehicle_boxes_object_list =  Filter3dBoxes().filter_according_to_distance(vehicle_boxes_object_list, 45)
        print('distance filter:')
        print('infra_boxes: {} -> {} ', len(converted_infra_boxes_object_list), len(distance_filted_infra_boxes_object_list))
        print('vehicle_boxes: {} -> {} ', len(vehicle_boxes_object_list), len(distance_filted_vehicle_boxes_object_list))

        degree = 1
        occlusion_filted_infra_boxes_object_list = Filter3dBoxes().filter_according_to_occlusion_truncation(distance_filted_infra_boxes_object_list, degree, degree)
        occlusion_filted_vehicle_boxes_object_list = Filter3dBoxes().filter_according_to_occlusion_truncation(distance_filted_vehicle_boxes_object_list, degree, degree)
        print('occlusion filter:')
        print('infra_boxes: {} -> {} ', len(distance_filted_infra_boxes_object_list), len(occlusion_filted_infra_boxes_object_list))
        print('vehicle_boxes: {} -> {} ', len(distance_filted_vehicle_boxes_object_list), len(occlusion_filted_vehicle_boxes_object_list))

        size_filted_infra_boxes_object_list = Filter3dBoxes().filter_according_to_size(occlusion_filted_infra_boxes_object_list)
        size_filted_vehicle_boxes_object_list = Filter3dBoxes().filter_according_to_size(occlusion_filted_vehicle_boxes_object_list)
        print('size filter:')
        print('infra_boxes: {} -> {} ', len(occlusion_filted_infra_boxes_object_list), len(size_filted_infra_boxes_object_list))
        print('vehicle_boxes: {} -> {} ', len(occlusion_filted_vehicle_boxes_object_list), len(size_filted_vehicle_boxes_object_list))

        return size_filted_infra_boxes_object_list, size_filted_vehicle_boxes_object_list

    def plot_boxes3d_distance_to_0(self, boxes3d_object_list):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        distance_list = []
        for box3d_object in boxes3d_object_list:
            centroid = box3d_object.get_bbox3d_8_3().mean(axis=0)
            centroid_xy = centroid[:2]
            distance = np.linalg.norm(centroid_xy)
            distance_list.append(distance)
        ax.boxplot(distance_list)
        plt.show()



if __name__ == "__main__":
    cooperative_reader = CooperativeReader('config.yml')
    converted_infra_pointcloud, vehicle_pointcloud = cooperative_reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()

    filtered_infra_3dboxes, filered_vehicle_3dboxes = Filter3dBoxes().test_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation()
    
    from GenerateCorrespondingListTask import GenerateCorrespondingListTask
    generateCorrespondingListTask = GenerateCorrespondingListTask('config.yml')
    infra_vehicle_corresponding_list, IoU_list, _ = generateCorrespondingListTask.generate_corresponding_list(filtered_infra_3dboxes, filered_vehicle_3dboxes)
    print(infra_vehicle_corresponding_list)
    print(IoU_list)

    boxes_color_list = [[1, 0, 0], [0, 1, 0]]
    from BBoxVisualizer_open3d import BBoxVisualizer_open3d
    BBoxVisualizer_open3d().plot_boxes3d_lists_pointcloud_lists([filtered_infra_3dboxes, filered_vehicle_3dboxes], [converted_infra_pointcloud, vehicle_pointcloud], boxes_color_list)
