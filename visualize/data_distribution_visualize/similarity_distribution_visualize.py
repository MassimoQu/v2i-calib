import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
from CooperativeBatchingReader import CooperativeBatchingReader
from CooperativeReader import CooperativeReader
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
from similarity_utils import cal_similarity_size, cal_similarity_angle, cal_similarity_length
from graph_utils import get_full_connected_edge
from bbox_utils import get_bbox3d_n_8_3_from_bbox_object_list
from statistic_utils import normalize_to_0_1

#单端
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
    

def test_length_angle_similarity_within_coupled_scene(infra_file_name, vehicle_file_name):
    infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    matches = CorrespondingDetector(infra_boxes_object_list, vehicle_boxes_object_list).corresponding_IoU_dict.keys()
    
    infra_boxes_object_list = []
    vehicle_boxes_object_list = []
    for match in matches:
        infra_boxes_object_list.append(infra_boxes_object_list[match[0]])
        vehicle_boxes_object_list.append(vehicle_boxes_object_list[match[1]])

    similarity_list = {}
    similarity_list['length'] = []
    similarity_list['angle'] = []

    for i in range(len(infra_boxes_object_list)):
        for j in range(len(infra_boxes_object_list)):
            if i != j:                 
                length_similarity = cal_similarity_length((infra_boxes_object_list[i].get_bbox3d_8_3(), infra_boxes_object_list[j].get_bbox3d_8_3()), (vehicle_boxes_object_list[i].get_bbox3d_8_3(), vehicle_boxes_object_list[j].get_bbox3d_8_3()))
                angle_similarity = cal_similarity_angle((infra_boxes_object_list[i].get_bbox3d_8_3(), infra_boxes_object_list[j].get_bbox3d_8_3()), (vehicle_boxes_object_list[i].get_bbox3d_8_3(), vehicle_boxes_object_list[j].get_bbox3d_8_3()))

                similarity_list['length'].append(length_similarity)
                similarity_list['angle'].append(angle_similarity)

    plt.figure(figsize=(10, 10))
    plt.boxplot([similarity_list['length'], similarity_list['angle']], labels=['length(' + str(len(similarity_list['length'])), 'angle(' + str(len(similarity_list['angle']))])
    plt.title('Length and Angle Distribution')
    plt.ylabel(f'Similarity of {infra_file_name} and {vehicle_file_name}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()
    

def test_length_angle_similarity_within_wide_scene(start_index, end_index):
    reader = CooperativeBatchingReader()

    similarity_list = {}
    similarity_list['length'] = []
    similarity_list['angle'] = []

    end_index -= start_index

    for infra_file_name, vehicle_file_name in zip(*reader.get_infra_vehicle_file_names()):
        if start_index > 0:
            start_index -= 1
            continue
        elif end_index > 0:
            end_index -= 1
        else :
            break

        infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
        matches = CorrespondingDetector(infra_boxes_object_list, vehicle_boxes_object_list).corresponding_IoU_dict.keys()
        
        infra_boxes_object_list = []
        vehicle_boxes_object_list = []
        for match in matches:
            infra_boxes_object_list.append(infra_boxes_object_list[match[0]])
            vehicle_boxes_object_list.append(vehicle_boxes_object_list[match[1]])

        for i in range(len(infra_boxes_object_list)):
            for j in range(len(infra_boxes_object_list)):
                if vehicle_boxes_object_list[i] == vehicle_boxes_object_list[j] or infra_boxes_object_list[i] == infra_boxes_object_list[j]:
                    continue
                length_similarity = cal_similarity_length((infra_boxes_object_list[i].get_bbox3d_8_3(), infra_boxes_object_list[j].get_bbox3d_8_3()), (vehicle_boxes_object_list[i].get_bbox3d_8_3(), vehicle_boxes_object_list[j].get_bbox3d_8_3()))
                angle_similarity = cal_similarity_angle((infra_boxes_object_list[i].get_bbox3d_8_3(), infra_boxes_object_list[j].get_bbox3d_8_3()), (vehicle_boxes_object_list[i].get_bbox3d_8_3(), vehicle_boxes_object_list[j].get_bbox3d_8_3()))

                similarity_list['length'].append(length_similarity)
                similarity_list['angle'].append(angle_similarity)

    plt.figure(figsize=(10, 10))
    plt.boxplot([similarity_list['length'], similarity_list['angle']], labels=['length(' + str(len(similarity_list['length'])), 'angle(' + str(len(similarity_list['angle']))])
    plt.title('Length and Angle Distribution')
    plt.ylabel(f'Similarity of scene No.{start_index} to No.{end_index}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()


def test_length_angle_similarity_within_wide_scene(start_index, end_index):
    reader = CooperativeBatchingReader()

    similarity_list = {}
    similarity_list['length'] = []
    similarity_list['angle'] = []

    end_index -= start_index

    for infra_file_name, vehicle_file_name in zip(*reader.get_infra_vehicle_file_names()):
        if start_index > 0:
            start_index -= 1
            continue
        elif end_index > 0:
            end_index -= 1
        else :
            break

        infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_boxes_object_list()
        
        for ii in range(len(infra_boxes_object_list)):
            for ij in range(len(infra_boxes_object_list)):
                if ii == ij:
                    continue
                for vi in range(len(vehicle_boxes_object_list)):
                    for vj in range(len(vehicle_boxes_object_list)):
                        if vi == vj:
                            continue
                        length_similarity = cal_similarity_length((infra_boxes_object_list[ii].get_bbox3d_8_3(), infra_boxes_object_list[ij].get_bbox3d_8_3()), (vehicle_boxes_object_list[vi].get_bbox3d_8_3(), vehicle_boxes_object_list[vj].get_bbox3d_8_3()))
                        angle_similarity = cal_similarity_angle((infra_boxes_object_list[ii].get_bbox3d_8_3(), infra_boxes_object_list[ij].get_bbox3d_8_3()), (vehicle_boxes_object_list[vi].get_bbox3d_8_3(), vehicle_boxes_object_list[vj].get_bbox3d_8_3()))

                        similarity_list['length'].append(length_similarity)
                        similarity_list['angle'].append(angle_similarity)
                
    plt.figure(figsize=(10, 10))
    plt.boxplot([similarity_list['length'], similarity_list['angle']], labels=['length(' + str(len(similarity_list['length'])), 'angle(' + str(len(similarity_list['angle']))])
    plt.title('Length and Angle Distribution')
    plt.ylabel(f'Similarity of scene No.{start_index} to No.{end_index}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()



if __name__ == '__main__':
    # infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader('003920', '020092').get_cooperative_infra_vehicle_boxes_object_list()
    
    # infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k = 20)
    # vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k = 20)
    
    # # visualize_edge_property_similarity_between_category(infra_boxes_object_list, vehicle_boxes_object_list, edge_property_calculator=cal_similarity_angle)
    # # visualize_edge_property_similarity_between_category(infra_boxes_object_list, vehicle_boxes_object_list, edge_property_calculator=cal_similarity_length)

    # visualize_size_similarity_between_same_category(infra_boxes_object_list)

    # test_length_angle_similarity_within_coupled_scene('015630', '006742')

    test_length_angle_similarity_within_wide_scene(0, 1)