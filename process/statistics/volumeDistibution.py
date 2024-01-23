import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
from CooperativeBatchingReader import CooperativeBatchingReader
from bbox_utils import get_volume_from_bbox3d_8_3

def batching_test_bbox3d_volume(k = 100, verbose = False):
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    infra_boxes_volume_list = []
    vehicle_boxes_volume_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list():
        
        if cnt >= k:
            break

        print('cnt: ', cnt)
        
        if verbose:
            print('infra_file_name: ', infra_file_name)
            print('vehicle_file_name: ', vehicle_file_name)
            print('len(infra_boxes_object_list): ', len(infra_boxes_object_list))
            print('len(vehicle_boxes_object_list): ', len(vehicle_boxes_object_list))
        
        for infra_box_object in infra_boxes_object_list:
            infra_boxes_volume_list.append(np.log(get_volume_from_bbox3d_8_3(infra_box_object.get_bbox3d_8_3())))

        for vehicle_box_object in vehicle_boxes_object_list:
            vehicle_boxes_volume_list.append(np.log(get_volume_from_bbox3d_8_3(vehicle_box_object.get_bbox3d_8_3())))

        cnt += 1

    print(f'batching length: {k}')

    import matplotlib.pyplot as plt
    plt.figure(figsize=(4, 6))
    plt.boxplot([infra_boxes_volume_list, vehicle_boxes_volume_list], labels=['infra(' + str(len(infra_boxes_volume_list)), 'vehicle(' + str(len(vehicle_boxes_volume_list)) + ')'])
    plt.title(f'infra & vehicle box volume distribution')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    batching_test_bbox3d_volume(k=100, verbose=False)