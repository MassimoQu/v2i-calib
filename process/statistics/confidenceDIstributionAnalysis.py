import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
from CooperativeBatchingReader import CooperativeBatchingReader

def batching_test_bbox3d_confidence(k = 100, verbose = False):
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    infra_boxes_confidence_total_list = []
    vehicle_boxes_confidence_total_list = []

    infra_boxes_confidence_categoried_dict = {}
    vehicle_boxes_confidence_categoried_dict = {}

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list_predicted():
        
        if cnt >= k:
            break

        print('cnt: ', cnt)
        
        if verbose:
            print('infra_file_name: ', infra_file_name)
            print('vehicle_file_name: ', vehicle_file_name)
            print('len(infra_boxes_object_list): ', len(infra_boxes_object_list))
            print('len(vehicle_boxes_object_list): ', len(vehicle_boxes_object_list))
        
        for infra_box_object in infra_boxes_object_list:
            infra_boxes_confidence_total_list.append(infra_box_object.confidence)
            if infra_box_object.get_bbox_type() not in infra_boxes_confidence_categoried_dict:
                infra_boxes_confidence_categoried_dict[infra_box_object.get_bbox_type()] = []
            infra_boxes_confidence_categoried_dict[infra_box_object.get_bbox_type()].append(infra_box_object.confidence)

        for vehicle_box_object in vehicle_boxes_object_list:
            vehicle_boxes_confidence_total_list.append(vehicle_box_object.confidence)
            if vehicle_box_object.get_bbox_type() not in vehicle_boxes_confidence_categoried_dict:
                vehicle_boxes_confidence_categoried_dict[vehicle_box_object.get_bbox_type()] = []
            vehicle_boxes_confidence_categoried_dict[vehicle_box_object.get_bbox_type()].append(vehicle_box_object.confidence)

        cnt += 1

    print(f'batching length: {k}')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 6))
    # plt.boxplot([infra_boxes_volume_list, vehicle_boxes_volume_list], labels=['infra(' + str(len(infra_boxes_volume_list)), 'vehicle(' + str(len(vehicle_boxes_volume_list)) + ')'])
    plt.boxplot([infra_boxes_confidence_total_list, vehicle_boxes_confidence_total_list], labels=['infra(' + str(len(infra_boxes_confidence_total_list)), 'vehicle(' + str(len(vehicle_boxes_confidence_total_list)) + ')'])
    plt.title(f'infra & vehicle box confidence distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.figure(figsize=(4, 6))
    plt.boxplot([infra_boxes_confidence_categoried_dict[key] for key in infra_boxes_confidence_categoried_dict], labels=[key + '(' + str(len(infra_boxes_confidence_categoried_dict[key])) + ')' for key in infra_boxes_confidence_categoried_dict])
    plt.title(f'infra box confidence distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.figure(figsize=(4, 6))
    plt.boxplot([vehicle_boxes_confidence_categoried_dict[key] for key in vehicle_boxes_confidence_categoried_dict], labels=[key + '(' + str(len(vehicle_boxes_confidence_categoried_dict[key])) + ')' for key in vehicle_boxes_confidence_categoried_dict])
    plt.title(f'vehicle box confidence distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)

    plt.show()

def test_reader():

    reader = CooperativeBatchingReader('config.yml')
    
    cnt = 0

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list_predicted():
        
        if cnt > 1:
            break

        print('cnt: ', cnt)

        print('infra_file_name: ', infra_file_name)
        print('vehicle_file_name: ', vehicle_file_name)

        # print infra_boxes_object_list
        print('len(infra_boxes_object_list): ', len(infra_boxes_object_list))

        for infra_box_object in infra_boxes_object_list:
            print('infra_box_object.get_bbox_type(): ', infra_box_object.get_bbox_type())
            print('infra_box_object.get_bbox3d_8_3(): ', infra_box_object.get_bbox3d_8_3())
            print('infra_box_object.get_confidence(): ', infra_box_object.get_confidence())
            print('--------------------------------------------------------')

        cnt += 1

if __name__ == "__main__":
    batching_test_bbox3d_confidence(k=100, verbose=False)
    # test_reader()