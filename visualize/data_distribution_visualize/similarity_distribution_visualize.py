import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
from CooperativeReader import CooperativeReader
from Filter3dBoxes import Filter3dBoxes
from similarity_utils import cal_similarity_size, cal_similarity_angle, cal_similarity_length
from graph_utils import get_full_connected_edge
from bbox_utils import get_bbox3d_n_8_3_from_bbox_object_list
from statistic_utils import normalize_to_0_1


def visualize_size_similarity_between_same_category(boxes_object_list):
    category_box_object_dict = {}
    for box_object in boxes_object_list:
        category = box_object.get_bbox_type()

        if category not in category_box_object_dict:
            category_box_object_dict[category] = []
        category_box_object_dict[category].append(box_object)
    

    similarity_list_dict = {}
    for category, boxes_object_list in category_box_object_dict.items():
        similarity_list_dict[category] = []
        previous_box_object = None
        for box_object in boxes_object_list:
            if previous_box_object is None:
                previous_box_object = box_object
                continue
            similarity_list_dict[category].append(cal_similarity_size(previous_box_object.get_bbox3d_8_3(), box_object.get_bbox3d_8_3()))
            previous_box_object = box_object

    sorted_keys = sorted(similarity_list_dict.keys())

    # 准备数据和标签用于绘制箱型图
    data_to_plot = []
    labels = []
    all_similarity_list = []
    for category in sorted_keys:
        similarity_list = similarity_list_dict[category]
        all_similarity_list += similarity_list
        data_to_plot.append(similarity_list)
        labels.append(category + ' ( ' + str(len(similarity_list)) + ')' )
    
    labels.append('all ( ' + str(len(all_similarity_list)) + ')' )
    data_to_plot.append(all_similarity_list)

    # 创建箱型图
    plt.figure(figsize=(10, 9))
    plt.boxplot(data_to_plot, labels=labels)
    plt.title('Size Distribution within Category')
    plt.ylabel('Size Similarity within Category')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Category')
    plt.grid(True)
    plt.show()


def visualize_edge_property_similarity_between_category(boxes_object_list1, boxes_object_list2, edge_property_calculator = cal_similarity_length):
    edges_list1 = get_full_connected_edge(boxes_object_list1)
    edges_list2 = get_full_connected_edge(boxes_object_list2)

    keys = ['same', 'different', 'all']
    similarity_list_dict = {}
    for key in keys:
        similarity_list_dict[key] = []

    for edge1 in edges_list1:
        for edge2 in edges_list2:
            if edge1[0].get_bbox_type() != edge2[0].get_bbox_type() or edge1[1].get_bbox_type() != edge2[1].get_bbox_type():
                continue
            
            bbox3d_2_8_3_one = get_bbox3d_n_8_3_from_bbox_object_list(edge1)
            bbox3d_2_8_3_two = get_bbox3d_n_8_3_from_bbox_object_list(edge2)
            property_similarity = edge_property_calculator(bbox3d_2_8_3_one, bbox3d_2_8_3_two)

            if edge1[0].get_bbox_type() == edge1[1].get_bbox_type():
                similarity_list_dict['same'].append(property_similarity)
            else:
                similarity_list_dict['different'].append(property_similarity)
            similarity_list_dict['all'].append(property_similarity)

    data_to_plot = []
    labels = []
    for key in keys:
        data = normalize_to_0_1(similarity_list_dict[key])
        data_to_plot.append(data)
        labels.append(key + ' ( ' + str(len(similarity_list_dict[key])) + ')' )

    plt.figure(figsize=(10, 9))
    plt.boxplot(data_to_plot, labels=labels)
    plt.title(edge_property_calculator.__name__ + ' Distribution')
    plt.ylabel(edge_property_calculator.__name__ + ' Similarity within Category')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
    

    

if __name__ == '__main__':
    cooperative_reader = CooperativeReader()
    infra_boxes_object_list, vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_bboxes_object_list()
    
    filter3dBoxes = Filter3dBoxes()
    infra_boxes_object_list, vehicle_boxes_object_list = filter3dBoxes.get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(topk=20)


    
    visualize_edge_property_similarity_between_category(infra_boxes_object_list, vehicle_boxes_object_list, edge_property_calculator=cal_similarity_angle)
    visualize_edge_property_similarity_between_category(infra_boxes_object_list, vehicle_boxes_object_list, edge_property_calculator=cal_similarity_length)

