import time
import json
import numpy as np
import cv2
import sys 
sys.path.append('./reader')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
sys.path.append('./visualize')

from CooperativeBatchingReader import CooperativeBatchingReader
from CooperativeReader import CooperativeReader
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes
from scipy.optimize import linear_sum_assignment
from extrinsic_utils import get_time_judge, implement_T_3dbox_object_list, get_extrinsic_from_two_3dbox_object
import similarity_utils
from appearance_similarity import cal_appearance_similarity

def crop_image_with_box2d(image, box2d):
    x1, y1, x2, y2 = box2d

    # print('x1, y1, x2, y2: ', x1, y1, x2, y2)
    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # to int
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    return image[y1:y2, x1:x2]

class BoxesMatch():

    def __init__(self,infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle = None, verbose=False, image_list = None):

        self.infra_boxes_object_list, self.vehicle_boxes_object_list = infra_boxes_object_list, vehicle_boxes_object_list
        
        self.T_infra2vehicle = T_infra2vehicle
        infra_node_num, vehicle_node_num = len(self.infra_boxes_object_list), len(self.vehicle_boxes_object_list)
        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)

        self.result_matches = []
        self.total_matches = []

        self.matches_num = []
        self.matches_avg_num = 0

        self.verbose = verbose

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
                    # similarity_size = similarity_utils.cal_similarity_size(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    # 邻近k个点的相似度
                    # similarity_knn = similarity_utils.cal_similarity_knn(self.infra_boxes_object_list, i, self.vehicle_boxes_object_list, j)
                    # self.KP[i, j] = int(similarity_size * 10) + int(similarity_knn)
                    # self.KP[i, j] =  int(similarity_knn)

                    T = get_extrinsic_from_two_3dbox_object(infra_bbox_object, vehicle_bbox_object)
                    converted_infra_boxes_object_list = implement_T_3dbox_object_list(T, infra_boxes_object_list)
                    
                    corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list)

                    self.matches_num.append(corresponding_detector.get_matched_num())

                    self.KP[i, j] = int(corresponding_detector.get_Yscore() * 100) 

            self.matches_num = sorted(self.matches_num)
            self.matches_avg_num = np.mean(self.matches_num[-3:])

            max_matches_num = 0
            if len(self.matches_num) > 0:
                max_matches_num = np.max(self.matches_num)

            max_val = np.max(self.KP)
            min_val = np.min(self.KP)
            for i in range(len(self.infra_boxes_object_list)):
                for j in range(len(self.vehicle_boxes_object_list)):
                    if self.KP[i, j] != 0:
                        self.KP[i, j] = int((self.KP[i, j] - min_val) / (max_val - min_val) * 10)

            if 0 < max_matches_num < 2:
                for i in range(len(self.infra_boxes_object_list)):
                    for j in range(len(self.vehicle_boxes_object_list)):
                        if self.KP[i, j] != 0:
                            appearance_similarity = 0
                            if image_list is not None:
                                appearance_similarity = cal_appearance_similarity(crop_image_with_box2d(image_list[0], infra_bbox_object.get_bbox2d_4()), crop_image_with_box2d(image_list[1], vehicle_bbox_object.get_bbox2d_4()))
                            self.KP[i, j] += int(appearance_similarity * 10)

        cal_KP()

        self.matches = self.get_matched_boxes_Hungarian_matching()

    def get_KP(self):
        return self.KP

    def get_max_matches_num(self):
        return self.matches_avg_num

    def get_matches(self):
        return self.matches

    def get_matches_with_score(self):
        matches_score_dict = {}
        for match in self.matches:
            if self.KP[match[0], match[1]] != 0:
                matches_score_dict[match] = self.KP[match[0], match[1]]
        sorted_matches_score_dict = sorted(matches_score_dict.items(), key=lambda x: x[1], reverse=True)
        return sorted_matches_score_dict

    def output_intermediate_KP(self):
        output_dir = './intermediate_output'
        np.savetxt(f"{output_dir}/KP_combined.csv", self.KP, delimiter=",", fmt='%d')


    def get_matched_boxes_Hungarian_matching(self):
        non_zero_rows = np.any(self.KP, axis=1)
        non_zero_columns = np.any(self.KP, axis=0)
        reduced_KP = self.KP[non_zero_rows][:, non_zero_columns]

        row_ind, col_ind = linear_sum_assignment(reduced_KP, maximize=True)
        original_row_ind = np.where(non_zero_rows)[0][row_ind]
        original_col_ind = np.where(non_zero_columns)[0][col_ind]
        matches = list(zip(original_row_ind, original_col_ind))
        return matches
       

    def cal_matches_accuracy(self, matches=None):
        if matches is None:
            matches = self.matches

        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        for match in matches:
            matched_infra_bboxes_object_list.append(self.infra_boxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(self.vehicle_boxes_object_list[match[1]])

        converted_infra_boxes_object_list = implement_T_3dbox_object_list(self.T_infra2vehicle, self.infra_boxes_object_list)
        
        self.total_matches = CorrespondingDetector(converted_infra_boxes_object_list, self.vehicle_boxes_object_list).corresponding_IoU_dict.keys()
        
        for match in matches:
            if match in self.total_matches:
                self.result_matches.append(match)
                
        cnt = len(self.result_matches)

        cnt, total_matches_cnt, total_cnt =  cnt, len(self.total_matches), len(matched_infra_bboxes_object_list)

        if self.verbose:
            print('true result matches / true matches / total: {} / {} / {}'.format(cnt, total_matches_cnt, total_cnt))
            print('Accuracy(true result matches / true matches): ', cnt / total_matches_cnt) if total_matches_cnt > 0 else print('Accuracy(true result matches / true matches): ', 0)

            infra_result_matches_types_list, vehicle_result_matches_types_list = self.get_matches_types(self.result_matches)
            infra_result_matches_types_list = sorted(infra_result_matches_types_list)
            vehicle_result_matches_types_list = sorted(vehicle_result_matches_types_list)
            print('infra_result_matches_types_list: ', infra_result_matches_types_list)
            print('vehicle_result_matches_types_list: ', vehicle_result_matches_types_list)

            infra_total_matches_types_list, vehicle_total_matches_types_list = self.get_matches_types(self.total_matches)
            infra_total_matches_types_list = sorted(infra_total_matches_types_list)
            vehicle_total_matches_types_list = sorted(vehicle_total_matches_types_list)
            infra_missing_matches_types_list = list(set(infra_total_matches_types_list) - set(infra_result_matches_types_list))
            vehicle_missing_matches_types_list = list(set(vehicle_total_matches_types_list) - set(vehicle_result_matches_types_list))
            print('infra_missing_matches_types_list: ', infra_missing_matches_types_list)
            print('vehicle_missing_matches_types_list: ', vehicle_missing_matches_types_list)

        return cnt, total_matches_cnt, total_cnt
    
    def get_matches_types(self, matches):
        infra_result_matches_types = []
        vehicle_result_matches_types = []
        for match in matches:
            infra_result_matches_types.append(self.infra_boxes_object_list[match[0]].get_bbox_type())
            vehicle_result_matches_types.append(self.vehicle_boxes_object_list[match[1]].get_bbox_type())
        return infra_result_matches_types, vehicle_result_matches_types

def specific_test_boxes_match(infra_num = '003920', vehicle_num = '020092', k = 10):
    infra_boxes_object_list, vehicle_boxes_object_list = CooperativeReader(infra_num, vehicle_num).get_cooperative_infra_vehicle_boxes_object_list()
    T_infra2vehicle = CooperativeReader(infra_num, vehicle_num).get_cooperative_T_i2v()
    
    infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
    vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)
    
    task = BoxesMatch(infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle, verbose=True)
    task.cal_matches_accuracy()
    task.output_intermediate_KP()


def batching_test_boxes_match(verbose = False, k = 10):
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    valid_matches_list = []
    non_matches_list = []
    error_matches_list = []

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list():
        # if cnt > 10:
        #     break

        if verbose:
            print('infra: {}   vehicle: {}'.format(infra_file_name, vehicle_file_name))

        filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
        filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)

        try:
            start_time = time.time()  # 开始计时

            boxes_matcher = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, T_infra2vehicle, verbose)
            matches_cnt, available_matches_cnt, filtered_cnt = boxes_matcher.cal_matches_accuracy()
            
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

            else:
                non_matches = {}
                non_matches['infra_file_name'] = infra_file_name
                non_matches['vehicle_file_name'] = vehicle_file_name
                non_matches['matches_cnt'] = matches_cnt
                non_matches['available_matches_cnt'] = available_matches_cnt
                non_matches['filtered_cnt'] = filtered_cnt
                non_matches['infra_total_box_cnt'] = len(infra_boxes_object_list)
                non_matches['vehicle_total_box_cnt'] = len(vehicle_boxes_object_list)
                non_matches['match_module_cost_time'] = end_time - start_time
                non_matches_list.append(non_matches)


        except Exception as e:
            if verbose:
                print('Error: ', infra_file_name, vehicle_file_name)
                print(e)
            error_matches = {}
            error_matches['infra_file_name'] = infra_file_name
            error_matches['vehicle_file_name'] = vehicle_file_name
            error_matches["error_message"] = str(e)
            error_matches_list.append(error_matches)
            
        cnt += 1
        print(cnt)
        if verbose:
            print('---------------------------------')

        if cnt % 100 == 0:
            if len(valid_matches_list):
                with open(f'intermediate_output/successful_matches_k{k}_cnt{cnt}.json', 'w') as f:
                    json.dump(valid_matches_list, f)

            if len(non_matches_list):
                with open(f'intermediate_output/non_matches_k{k}_cnt{cnt}.json', 'w') as f:
                    json.dump(non_matches_list, f)

            if len(error_matches_list):
                with open(f'intermediate_output/error_matches_k{k}_cnt{cnt}.json', 'w') as f:
                    json.dump(error_matches_list, f)

            valid_matches_list = []
            non_matches_list = []
            error_matches_list = []

            print('----------------write to file---------------------')

    if len(valid_matches_list):
        with open(f'intermediate_output/successful_matches_k{k}_cnt{cnt}.json', 'w') as f:
            json.dump(valid_matches_list, f)

    if len(non_matches_list):
        with open(f'intermediate_output/non_matches_k{k}_cnt{cnt}.json', 'w') as f:
            json.dump(non_matches_list, f)

    if len(error_matches_list):
        with open(f'intermediate_output/error_matches_k{k}_cnt{cnt}.json', 'w') as f:
            json.dump(error_matches_list, f)


def batching_test_matches_score(verbose = False, k = 15):
    reader = CooperativeBatchingReader('config.yml')
    cnt = 0 

    KP_score_list = {}
    matches_score_list = {}

    for infra_file_name, vehicle_file_name, infra_boxes_object_list, vehicle_boxes_object_list, T_infra2vehicle in reader.generate_infra_vehicle_bboxes_object_list():
        if cnt >= 100:
            break

        if verbose:
            print('infra: {}   vehicle: {}'.format(infra_file_name, vehicle_file_name))
            print('infra_boxes_object_list: ', len(infra_boxes_object_list))
            print('vehicle_boxes_object_list: ', len(vehicle_boxes_object_list))

        filtered_infra_boxes_object_list = Filter3dBoxes(infra_boxes_object_list).filter_according_to_size_topK(k)
        filtered_vehicle_boxes_object_list = Filter3dBoxes(vehicle_boxes_object_list).filter_according_to_size_topK(k)

        boxes_matcher = BoxesMatch(filtered_infra_boxes_object_list, filtered_vehicle_boxes_object_list, T_infra2vehicle, verbose)

        KP = boxes_matcher.get_KP()
        matches_with_score_list = boxes_matcher.get_matches_with_score()

        KP_score_list_part = {}
        matches_score_list_part = {} 

        for i in range(KP.shape[0]):
            for j in range(KP.shape[1]):
                if KP[i, j] != 0:
                    if KP[i, j] not in KP_score_list_part:
                        KP_score_list_part[KP[i, j]] = 1
                    else:
                        KP_score_list_part[KP[i, j]] += 1

        sorted_KP_score_list_part = sorted(KP_score_list_part.items(), key=lambda x: x[0])
        for KP_score_with_cnt in sorted_KP_score_list_part:
            # if KP_score_with_cnt[0] > 100:
            #     break
            if KP_score_with_cnt[0] not in KP_score_list:
                KP_score_list[KP_score_with_cnt[0]] = KP_score_with_cnt[1]
            else:
                KP_score_list[KP_score_with_cnt[0]] += KP_score_with_cnt[1]

        for matches_with_score in matches_with_score_list:
            if matches_with_score[1] not in matches_score_list_part:
                matches_score_list_part[matches_with_score[1]] = 1
            else:
                matches_score_list_part[matches_with_score[1]] += 1

        sorted_matches_score_list_part = sorted(matches_score_list_part.items(), key=lambda x: x[0])
        for matches_score_with_cnt in sorted_matches_score_list_part:
            # if matches_score_with_cnt[0] > 100:
            #     break
            if matches_score_with_cnt[0] not in matches_score_list:
                matches_score_list[matches_score_with_cnt[0]] = matches_score_with_cnt[1]
            else:
                matches_score_list[matches_score_with_cnt[0]] += matches_score_with_cnt[1]

        cnt += 1
        print(cnt)

        if verbose:
            print('len(KP_score_list_part): ', len(KP_score_list_part))
            print('sorted_KP_score_list_part: ', sorted_KP_score_list_part)
            print('len(matches_score_list_part): ', len(matches_score_list_part))
            print('sorted_matches_score_list_part: ', sorted_matches_score_list_part)
            print('---------------------------------')

    sorted_KP_score_list = sorted(KP_score_list.items(), key=lambda x: x[0])
    sorted_matches_score_list = sorted(matches_score_list.items(), key=lambda x: x[0])

    if verbose:
        print('------final-----------')
        print('KP_score_list: ', sorted_KP_score_list)
        print('matches_score_list: ', sorted_matches_score_list)

    # draw boxplot
    import matplotlib.pyplot as plt
    # Splitting the tuples into two lists
    scores, counts = zip(*sorted_KP_score_list)
    # Creating the plot
    plt.figure()
    plt.scatter(scores, counts, color='blue') # Scatter plot to show the distribution
    plt.plot(scores, counts, color='red', linestyle='dashed') # Line plot to show the trend
    plt.title(f'Distribution of Scores and their Counts of KP score , len:{len(scores)}')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(True)

    # Splitting the tuples into two lists
    scores, counts = zip(*sorted_matches_score_list)
    # Creating the plot
    plt.figure()
    plt.scatter(scores, counts, color='blue') # Scatter plot to show the distribution
    plt.plot(scores, counts, color='red', linestyle='dashed') # Line plot to show the trend
    plt.title(f'Distribution of Scores and their Counts of matches score , len:{len(scores)}')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()
    


if __name__ == "__main__":
    # specific_test_boxes_match('007489', '000289', k=15)
    # batching_test_boxes_match(verbose=True, k=15)
    batching_test_matches_score(verbose=True, k=15)