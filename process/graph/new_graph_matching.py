import numpy as np
import sys 
sys.path.append('./reader')
sys.path.append('./visualize')
from CooperativeReader import CooperativeReader
from process.corresponding.GenerateCorrespondingListTask import GenerateCorrespondingListTask
from BBoxVisualizer_open3d import BBoxVisualizer_open3d as BBoxVisualizer
from Filter3dBoxes import Filter3dBoxes
from scipy.optimize import linear_sum_assignment
from convert_utils import get_time
import similarity_utils


class BoxesObjectMatching():
    def __init__(self):
        # self.reader = CooperativeReader()
        # self.infra_boxes_object_list, self.vehicle_boxes_object_list = self.reader.get_cooperative_infra_vehicle_pointcloud_vehicle_coordinate()

        self.filter3dBoxes = Filter3dBoxes()
        self.infra_boxes_object_list, self.vehicle_boxes_object_list = self.filter3dBoxes.get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(topk=20)
        
        infra_node_num, vehicle_node_num = len(self.infra_boxes_object_list), len(self.vehicle_boxes_object_list)
        
        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float32)


    def cal_KP(self):

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

                # self.KP[i, j] = int(similarity_size) + similarity_knn # 乘的策略好像有点粗暴
                self.KP[i, j] = similarity_knn

                # if self.KP[i, j] > 0:
                #     print('j ==', j)
                #     print(vehicle_bbox_object.get_bbox_type())


    def output_intermediate_result(self):
        self.cal_KP()

        output_dir = './output'

        # Save matrices KP, KQ, Ct, asgTX
        np.savetxt(f"{output_dir}/KP_combined.csv", self.KP, delimiter=",", fmt='%d')


    def get_matched_boxes_topK(self, topK = 5):
        self.cal_KP()
        indices_and_values = [(index, value) for index, value in np.ndenumerate(self.KP)]
        topK_matches = sorted(indices_and_values, key=lambda x: x[1], reverse=True)[:topK]
        return topK_matches


    def get_matched_boxes_Hungarian_matching(self):
        print('start KP')
        self.cal_KP()
        print('KP completed')
        print('start Hungarian matching')
        row_ind, col_ind = linear_sum_assignment(self.KP, maximize=True)
        print('Hungarian matching completed')
        matches = list(zip(row_ind, col_ind))
        return matches
    

    def test_given_matches_visualization_view(self, matches):
        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        # cnt = 0

        for match in matches:
            # if cnt != 0:
            #     continue
            # cnt += 1
            matched_infra_bboxes_object_list.append(self.infra_boxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(self.vehicle_boxes_object_list[match[1]])

        for select_num in range(len(matched_infra_bboxes_object_list)):

            # print(len(matched_infra_bboxes_object_list))
            # print(len(matched_vehicle_bboxes_object_list))

            bbox_visualizer = BBoxVisualizer()
            # converted_matched_infra_bboxes_object_list = CoordinateConversion().convert_bboxes_object_list_infra_lidar_2_vehicle_lidar(matched_infra_bboxes_object_list)
            
            selected_infra_bboxes_object_list = [matched_infra_bboxes_object_list[select_num]]
            selected_vehicle_bboxes_object_list = [matched_vehicle_bboxes_object_list[select_num]]
            
            bbox_visualizer.plot_boxes3d_lists([matched_vehicle_bboxes_object_list, matched_infra_bboxes_object_list, selected_vehicle_bboxes_object_list, selected_infra_bboxes_object_list], 
                                                [(1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1)])


            # converted_infra_bboxes_object_list = CoordinateConversion().convert_bboxes_object_list_infra_lidar_2_vehicle_lidar(task.infra_boxes_object_list)
            # bbox_visualizer.plot_boxes3d_lists([selected_infra_bboxes_object_list, converted_infra_bboxes_object_list, selected_vehicle_bboxes_object_list, task.vehicle_boxes_object_list], 
            #                                 [(0, 0, 0), (0, 1, 0), (0, 0, 0),(1, 0, 0)])


    def count_given_matches_accuracy(self, matches):
        generate_corresponding_list_task = GenerateCorrespondingListTask()

        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        cnt = 0

        for match in matches:
            matched_infra_bboxes_object_list.append(self.infra_boxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(self.vehicle_boxes_object_list[match[1]])

        # for select_num in range(len(matched_infra_bboxes_object_list)):
        #     selected_infra_bbox_object = matched_infra_bboxes_object_list[select_num]
        #     selected_vehicle_bbox_object = matched_vehicle_bboxes_object_list[select_num]
        #     iou = generate_corresponding_list_task.cal_single_3dIoU(selected_infra_bbox_object.get_bbox3d_8_3(), selected_vehicle_bbox_object.get_bbox3d_8_3())
        #     if iou != 0:
        #         cnt += 1

        total_matches = generate_corresponding_list_task.generate_corresponding_list(self.infra_boxes_object_list, self.vehicle_boxes_object_list)[0]
        
        true_result_matches = []
        for match in matches:
            if match in total_matches:
                true_result_matches.append(match)
                
        cnt = len(true_result_matches)

        print('true result matches / true matches / total: {} / {} / {}'.format(cnt, len(total_matches), len(matched_infra_bboxes_object_list)))
        print('Accuracy(true result matches / true matches): ', cnt / len(total_matches))

@get_time
def main():
    task = BoxesObjectMatching()

    matches = task.get_matched_boxes_Hungarian_matching()
    # print(matches)
    # task.test_given_matches_visualization_view(matches)
    task.count_given_matches_accuracy(matches)
    task.output_intermediate_result()


if __name__ == "__main__":
    main()
