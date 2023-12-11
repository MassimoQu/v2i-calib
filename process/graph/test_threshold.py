import sys
sys.path.append('./reader')
sys.path.append('./process')
sys.path.append('./process/graph')
sys.path.append('./process/corresponding')
sys.path.append('./process/utils')
sys.path.append('./process/search')
sys.path.append('./visualize')

from CooperativeReader import CooperativeReader
from CorrespondingDetector import CorrespondingDetector
from similarity_utils import cal_similarity_length, cal_similarity_angle


def test_length_similarity_within_coupled_scene(infra_file_name, vehicle_file_name):
    infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader(infra_file_name, vehicle_file_name).get_cooperative_infra_vehicle_boxes_object_lists_vehicle_coordinate()
    matches = CorrespondingDetector(infra_boxes_object_list, vehicle_boxes_object_list).corresponding_IoU_dict.keys()
    
    matched_infra_boxes_object_list = []
    matched_vehicle_boxes_object_list = []
    for match in matches:
        matched_infra_boxes_object_list.append(infra_boxes_object_list[match[0]])
        matched_vehicle_boxes_object_list.append(vehicle_boxes_object_list[match[1]])

    similarity_list = {}
    similarity_list['length'] = []
    similarity_list['angle'] = []

    for i in range(len(matched_infra_boxes_object_list)):
        for j in range(len(matched_infra_boxes_object_list)):
            if i != j:
                if matched_infra_boxes_object_list[i].get_bbox_type() != matched_vehicle_boxes_object_list[j].get_bbox_type():
                    print(f'infra_file_name: {infra_file_name}, vehicle_file_name: {vehicle_file_name} match type not equal')
                    print(f'matched_infra_boxes_object_list[i].get_bbox_type(): {matched_infra_boxes_object_list[i].get_bbox_type()}, matched_vehicle_boxes_object_list[j].get_bbox_type(): {matched_vehicle_boxes_object_list[j].get_bbox_type()}')
                    
                length_similarity = cal_similarity_length((matched_infra_boxes_object_list[i].get_bbox3d_8_3(), matched_infra_boxes_object_list[j].get_bbox3d_8_3()), (matched_vehicle_boxes_object_list[i].get_bbox3d_8_3(), matched_vehicle_boxes_object_list[j].get_bbox3d_8_3()))
                angle_similarity = cal_similarity_angle((matched_infra_boxes_object_list[i].get_bbox3d_8_3(), matched_infra_boxes_object_list[j].get_bbox3d_8_3()), (matched_vehicle_boxes_object_list[i].get_bbox3d_8_3(), matched_vehicle_boxes_object_list[j].get_bbox3d_8_3()))

                similarity_list['length'].append(length_similarity)
                similarity_list['angle'].append(angle_similarity)




if __name__ == '__main__':
    test_length_similarity_within_coupled_scene('015630', '006742')
