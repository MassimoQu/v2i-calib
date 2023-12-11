import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

import sys
sys.path.append('./reader')
sys.path.append('./process/utils')
sys.path.append('./process/serach')
sys.path.append('./process/corresponding')
from CooperativeReader import CooperativeReader
from ExtrinsicCandidateGenerate import ExtrinsicCandidateGenerator
from extrinsic_utils import implement_T_3dbox_object_list, convert_T_to_6DOF, convert_Rt_to_T, get_time
from CorrespondingDetector import CorrespondingDetector
from Filter3dBoxes import Filter3dBoxes


class DataDistributionAnalysis():
    def __init__(self, infra_num, vehicle_num):
        cooperative_reader = CooperativeReader(infra_num, vehicle_num)
        self.infra_boxes_object_list, self.vehicle_boxes_object_list = cooperative_reader.get_cooperative_infra_vehicle_boxes_object_list()
        # self.infra_boxes_object_list = filter_3dboxes.filter_according_to_size_percentile(self.infra_boxes_object_list, 75)
        # self.vehicle_boxes_object_list = filter_3dboxes.filter_according_to_size_percentile(self.vehicle_boxes_object_list, 75)
        self.infra_boxes_object_list = Filter3dBoxes(self.infra_boxes_object_list).filter_according_to_size_topK(k = 30)
        self.vehicle_boxes_object_list = Filter3dBoxes(self.vehicle_boxes_object_list).filter_according_to_size_topK(k = 30)
        self.T_infra2vehicle = convert_Rt_to_T(*cooperative_reader.get_cooperative_Rt_i2v())

        extrinsic_candidate_generator = ExtrinsicCandidateGenerator(self.infra_boxes_object_list, self.vehicle_boxes_object_list)
        self.candidate6DOF_list = extrinsic_candidate_generator.get_whole_candidate6DOF_list()
        self.candidateT_list = extrinsic_candidate_generator.get_whole_candidateT_list()


    def test__DBSCAN_cluster(self):
        neighbors = NearestNeighbors(n_neighbors=2).fit(self.candidate6DOF_list)
        distances, indices = neighbors.kneighbors(self.candidate6DOF_list)
        distances = np.sort(distances, axis=0)
        distances = distances[:,1]
        plt.plot(distances)
        plt.title('K-Nearest Neighbors Distance')
        plt.xlabel('Points sorted by distance')
        plt.ylabel('Epsilon (eps)')
        plt.show()

        # eps = 50

        for eps in np.linspace(0, 1, 10):
            if eps < 10e-6:
                continue
            # 运行 DBSCAN
            dbscan = DBSCAN(eps=eps, min_samples=2)  # 假设 min_samples 为 2
            dbscan.fit(self.candidate6DOF_list)

            # 查看聚类结果
            labels = dbscan.labels_

            # 找出最大的聚类
            labels, counts = np.unique(labels[labels >= 0], return_counts=True)  # 忽略噪声点
            if counts is None or len(counts) == 0:
                continue
            largest_cluster_idx = labels[np.argmax(counts)]
            largest_cluster_points = self.candidate6DOF_list[dbscan.labels_ == largest_cluster_idx]

            # 计算最大聚类的代表值，例如，取均值
            # representative_param = np.mean(largest_cluster_points, axis=0)

            print("eps = {} 时代表外参: {}".format(eps, largest_cluster_points))


    def test_IoU_num_between_all_candidateT_list(self):
        cnt = 0
        for candidateT in self.candidateT_list:
            infra_boxes_object_list, vehicle_boxes_object_list = self.infra_boxes_object_list.copy(), self.vehicle_boxes_object_list.copy()
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(candidateT, infra_boxes_object_list)
            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list)
            print('cnt / candidate : {} / {} '.format(cnt, len(self.candidateT_list)))
            print('candidate6DOF: ', convert_T_to_6DOF(candidateT))
            print('IoU: ', corresponding_detector.get_Yscore())
            print('matched_num / total_num : {} / {} '.format(corresponding_detector.get_matched_num(), corresponding_detector.get_total_num()))
            print('--------------------------------------------------------')
            cnt += 1

    @get_time
    def get_rough6DOF_from_all_candidateT_list(self):
        rough6DOF_list = []
        cnt = 0
        for candidateT in self.candidateT_list:
            infra_boxes_object_list, vehicle_boxes_object_list = self.infra_boxes_object_list.copy(), self.vehicle_boxes_object_list.copy()
            converted_infra_boxes_object_list = implement_T_3dbox_object_list(candidateT, infra_boxes_object_list)
            corresponding_detector = CorrespondingDetector(converted_infra_boxes_object_list, vehicle_boxes_object_list)
            if corresponding_detector.get_matched_num() > 10:
                rough6DOF_list.append(convert_T_to_6DOF(candidateT))
                print('cnt / candidate : {} / {} '.format(cnt, len(self.candidateT_list)))
                print('matched_num / total_num : {} / {} '.format(corresponding_detector.get_matched_num(), corresponding_detector.get_total_num()))
                print('IoU: ', corresponding_detector.get_Yscore())
                print('candidate6DOF: ', convert_T_to_6DOF(candidateT))
                print('--------------------------------------------------------')
            cnt += 1
        print('len(rough6DOF_list): ', len(rough6DOF_list))

        print('true6DOF: ', convert_T_to_6DOF(self.T_infra2vehicle))

        return rough6DOF_list


if __name__ == "__main__":
    dataDistributionAnalysis = DataDistributionAnalysis('005298', '001374')
    # dataDistributionAnalysis.test_IoU_num_between_all_candidateT_list()
    # dataDistributionAnalysis.test__DBSCAN_cluster()

    rough6DOF_list = dataDistributionAnalysis.get_rough6DOF_from_all_candidateT_list()
    
    
