import time
import json
import numpy as np
import sys 
sys.path.append('./reader')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
sys.path.append('./visualize')

from CooperativeBatchingReader import CooperativeBatchingReader
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
from scipy.optimize import linear_sum_assignment
from extrinsic_utils import get_time_judge
import similarity_utils


class BoxesMatch():
    def __init__(self,infra_boxes_object_list, vehicle_boxes_object_list, verbose=False):

        self.infra_boxes_object_list, self.vehicle_boxes_object_list = infra_boxes_object_list, vehicle_boxes_object_list
        
        infra_node_num, vehicle_node_num = len(self.infra_boxes_object_list), len(self.vehicle_boxes_object_list)
        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)

        @get_time_judge(verbose)
        def cal_KP():
            for i, infra_bbox_object in enumerate(self.infra_boxes_object_list):
                # print('i == ', i)
                # print(infra_bbox_object.get_bbox_type())
                for j, vehicle_bbox_object in enumerate(self.vehicle_boxes_object_list):
                    if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                        self.KP[i, j] = 0
                        continue         
                    # 检测框大小
                    similarity_size = similarity_utils.cal_similarity_size(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    # 邻近k个点的相似度
                    similarity_knn = similarity_utils.cal_similarity_knn(self.infra_boxes_object_list, i, self.vehicle_boxes_object_list, j)
                    # self.KP[i, j] = int(similarity_size * 10) + int(similarity_knn)
                    self.KP[i, j] =  int(similarity_knn)

                    # if self.KP[i, j] > 0:
                    #     print('j ==', j)
                    #     print(vehicle_bbox_object.get_bbox_type())

        cal_KP()

        self.matches = self.get_matched_boxes_Hungarian_matching()

    def output_intermediate_KP(self):
        output_dir = './intermediate_output'
        np.savetxt(f"{output_dir}/KP_combined.csv", self.KP, delimiter=",", fmt='%d')


    def get_matched_boxes_Hungarian_matching(self):
        row_ind, col_ind = linear_sum_assignment(self.KP, maximize=True)
        matches = list(zip(row_ind, col_ind))
        return matches


    def view_matches_accuracy(self, matches=None):
        cnt, total_matches_cnt, total_cnt =  self.cal_matches_accuracy(matches)

        print('true result matches / true matches / total: {} / {} / {}'.format(cnt, total_matches_cnt, total_cnt))
        print('Accuracy(true result matches / true matches): ', cnt / total_matches_cnt) if total_matches_cnt > 0 else print('Accuracy(true result matches / true matches): ', 0)

    def cal_matches_accuracy(self, matches=None):
        if matches is None:
            matches = self.matches

        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        for match in matches:
            matched_infra_bboxes_object_list.append(self.infra_boxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(self.vehicle_boxes_object_list[match[1]])

        total_matches = CorrespondingDetector(self.infra_boxes_object_list, self.vehicle_boxes_object_list).corresponding_IoU_dict.keys()
        
        true_result_matches = []
        for match in matches:
            if match in total_matches:
                true_result_matches.append(match)
                
        cnt = len(true_result_matches)

        return cnt, len(total_matches), len(matched_infra_bboxes_object_list)


    def filter_mismatches(self, matches=None):
        if matches is None:
            matches = self.matches

        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        for match in matches:
            matched_infra_bboxes_object_list.append(self.infra_boxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(self.vehicle_boxes_object_list[match[1]])

        total_matches = CorrespondingDetector(self.infra_boxes_object_list, self.vehicle_boxes_object_list).corresponding_IoU_dict.keys()
        
        true_result_matches = []
        for match in matches:
            if match in total_matches:
                true_result_matches.append(match)
                
        cnt = len(true_result_matches)

        print('true result matches / true matches / total: {} / {} / {}'.format(cnt, len(total_matches), len(matched_infra_bboxes_object_list)))
        print('Accuracy(true result matches / true matches): ', cnt / len(total_matches))

def main():
    filter3dBoxes = Filter3dBoxes()
    infra_boxes_object_list, vehicle_boxes_object_list = filter3dBoxes.get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(topk=20)
    
    task = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list, verbose=True)
    task.view_matches_accuracy()
    task.output_intermediate_KP()


def batching_test_boxes_match():
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    valid_matches_list = []
    error_matches_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list in reader.generate_infra_vehicle_bboxes_object_list():
        # if cnt > 10:
        #     break

        # print('infra: {}   vehicle: {}'.format(infra_file_name, vehicle_file_name))

        filtered_infra_boxes_object_list = Filter3dBoxes().filter_according_to_size_topK(infra_boxes_object_list, k=10)
        filtered_vehicle_boxes_object_list = Filter3dBoxes().filter_according_to_size_topK(vehicle_boxes_object_list, k=10)

        try:
            start_time = time.time()  # 开始计时
            matches_cnt, available_matches_cnt, filtered_cnt = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list).cal_matches_accuracy()
            end_time = time.time()  # 结束计时

            if matches_cnt > 0:
                valid_matches = {}
                valid_matches['infra_file_name'] = infra_file_name
                valid_matches['vehicle_file_name'] = vehicle_file_name
                valid_matches['matches_cnt'] = matches_cnt
                valid_matches['available_matches_cnt'] = available_matches_cnt
                valid_matches['filtered_cnt'] = filtered_cnt
                valid_matches['infra_total_box_cnt'] = len(infra_boxes_object_list)
                valid_matches['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                valid_matches['match_module_cost_time'] = end_time - start_time
                valid_matches_list.append(valid_matches)

        except Exception as e:
            # print('Error: ', infra_file_name, vehicle_file_name)
            error_matches = {}
            error_matches['infra_file_name'] = infra_file_name
            error_matches['vehicle_file_name'] = vehicle_file_name
            error_matches["error_message"] = str(e)
            error_matches_list.append(error_matches)
            
        cnt += 1
        print(cnt)
        # print('---------------------------------')

        if cnt % 50 == 0:
            with open(f'intermediate_output/successful_matches_{cnt}.json', 'w') as f:
                json.dump(valid_matches_list, f)

            with open(f'intermediate_output/error_matches_{cnt}.json', 'w') as f:
                json.dump(error_matches_list, f)

            valid_matches_list = []
            error_matches_list = []

            print('----------------write to file---------------------')

            

    if valid_matches_list or error_matches_list:
        with open(f'intermediate_output/successful_matches_{cnt}.json', 'w') as f:
                json.dump(valid_matches_list, f)

        with open(f'intermediate_output/error_matches_{cnt}.json', 'w') as f:
            json.dump(error_matches_list, f)

if __name__ == "__main__":
    # main()
    batching_test_boxes_match()