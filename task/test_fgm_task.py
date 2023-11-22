import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

import module.fgm as fgm

import sys
sys.path.append('./reader')
sys.path.append('./visualize')
import CooperativeReader
import module.similarity_utils as similarity_utils
from module.CoordinateConversion import CoordinateConversion
from module.Filter3dBoxes import Filter3dBoxes
from module.convert_utils import get_time
from BBoxVisualizer_open3d import BBoxVisualizer_open3d as BBoxVisualizer
from GenerateCorrespondingListTask import GenerateCorrespondingListTask


class FGMTask():

    '''
        v1.0: 逆向同权的有向边、全连接图
    '''

    def __init__(self):
        # self.reader = CooperativeReader.CooperativeReader('config.yml')
        # self.infra_bboxes_object_list, self.vehicle_bboxes_object_list = self.reader.get_cooperative_infra_vehicle_bboxes_object_list()

        self.filter3dBoxes = Filter3dBoxes()
        self.infra_bboxes_object_list, self.vehicle_bboxes_object_list = self.filter3dBoxes.get_filtered_infra_vehicle_according_to_size_distance_occlusion_truncation(topk=10)

        # print(self.infra_bboxes_object_list, self.vehicle_bboxes_object_list)

        infra_node_num = len(self.infra_bboxes_object_list)
        infra_edge_num = infra_node_num * (infra_node_num - 1) 
        vehicle_node_num = len(self.vehicle_bboxes_object_list)
        vehicle_edge_num = vehicle_node_num * (vehicle_node_num - 1)

        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float64)
        self.KQ = np.zeros((infra_edge_num, vehicle_edge_num), dtype=np.float64)
        self.Ct = np.ones((infra_node_num, vehicle_node_num), dtype=np.float64)
        self.asgTX = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float64)
        self.gph1 = {}
        self.gph2 = {}
        
        # self.gph1['H'] = np.hstack((self.gph1['G'], np.eye(self.gph1['G'].shape[0])))
        # self.gph2['H'] = np.hstack((self.gph2['G'], np.eye(self.gph2['G'].shape[0])))
        # print(self.gph1['G'].shape)
        # print(self.gph1['H'].shape)
        # print(self.gph2['G'].shape)
        # print(self.gph2['H'].shape)


    def trans_graph_GH(self, A):
        """
        BUILDS NODE-EDGE INCIDENCE MATRICES G AND H FROM GIVEN AFFINITY MATRIX

        Arguments:
        ----------
            - A: node-to-node adjaceny matrix

        Returns:
        --------
            - G and H: node-edge incidence matrices such that: A = G*H^T

        """

        # Get number of nodes
        n = A.shape[0]

        # Get all non-zero entries and build G and H
        entries = np.transpose((A != 0).nonzero()) 
        # print('entries')
        # print(entries.shape)

        # print(entries)

        # Init G and H
        G = np.zeros((n, entries.shape[0]), dtype=np.float64)
        H = np.zeros((n, entries.shape[0]), dtype=np.float64)

        for count, (i,j) in enumerate(entries, start=0):
            G[i, count] = 1
            H[j, count] = 1
            # G[j, count] = 1


        return G, H



    def cal_KP(self):
        for i, infra_bbox_object in enumerate(self.infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(self.vehicle_bboxes_object_list):

                if infra_bbox_object.get_bbox_type() != vehicle_bbox_object.get_bbox_type():
                    self.KP[i, j] = 0
                    continue

                # 检测框大小
                similarity_size = similarity_utils.cal_similarity_size(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())

                # 中心点的距离?
                # similarity_center = similarity_utils.cal_similarity_center(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())

                self.KP[i, j] = int(similarity_size * 10)

#KQ
    def build_FULLY_CONNECTED_graph_structure(self, bbox_object_list):
        centerxyz_list = similarity_utils.extract_centerxyz_from_object_list(bbox_object_list)
        dist_matrix = squareform(pdist(centerxyz_list))
        return dist_matrix

    def build_FULLY_CONNECTED_graph_structure_edges_list(self, bbox_object_list):
        num_nodes = len(bbox_object_list)
        edges_list = []

        for i in range(num_nodes):
            for j in range(num_nodes): 
                if i != j:
                    edges_list.append([i, j])

        return edges_list


    def cal_KQ_FULLY_CONNECTED(self):
        self.gph1['G'], self.gph1['H'] = self.trans_graph_GH(self.build_FULLY_CONNECTED_graph_structure(self.infra_bboxes_object_list))
        self.gph2['G'], self.gph2['H'] = self.trans_graph_GH(self.build_FULLY_CONNECTED_graph_structure(self.vehicle_bboxes_object_list))
        infra_edges_list = self.build_FULLY_CONNECTED_graph_structure_edges_list(self.infra_bboxes_object_list)
        print('infra_edges_list')
        print(len(infra_edges_list))
        vehicle_edges_list = self.build_FULLY_CONNECTED_graph_structure_edges_list(self.vehicle_bboxes_object_list)
        print('vehicle_edges_list')
        print(len(vehicle_edges_list))
        self.cal_KQ(infra_edges_list, vehicle_edges_list)



    def build_MST_graph_structure(self, bbox_object_list):
        '''
            1. 用中心点计算两个框的长度相似度
            2. 用几何体最近点计算两个框的长度相似度
        
        '''
        centerxyz_list = similarity_utils.extract_centerxyz_from_object_list(bbox_object_list)
        # 计算所有点对之间的距离
        dist_matrix = squareform(pdist(centerxyz_list))

        # 应用最小生成树算法
        mst = minimum_spanning_tree(dist_matrix)
        
        return mst

    def build_MST_graph_structure_edges_list(self, bbox_object_list):
        '''
            1. 用中心点计算两个框的长度相似度
            2. 用几何体最近点计算两个框的长度相似度
        
        '''
        mst = self.build_MST_graph_structure(bbox_object_list)

        # 将 MST 结果转换为边的列表
        mst_edges = np.vstack(mst.nonzero()).T

        # 将边的索引转换为所需格式
        edges_list = mst_edges.tolist()

        return edges_list


    def cal_KQ_MST(self):
        '''
            边的编号从 graph_structure 中入手
            从行入手
            结合 trans_graph_GH 中用 nonzero 搜边的策略，作解耦和重构
        
        '''
        self.gph1['G'], self.gph1['H'] = self.trans_graph_GH(self.build_MST_graph_structure(self.infra_bboxes_object_list).toarray())
        self.gph2['G'], self.gph2['H'] = self.trans_graph_GH(self.build_MST_graph_structure(self.vehicle_bboxes_object_list).toarray())
        infra_edges_list = self.build_MST_graph_structure_edges_list(self.infra_bboxes_object_list)
        vehicle_edges_list = self.build_MST_graph_structure_edges_list(self.vehicle_bboxes_object_list)
        self.cal_KQ(infra_edges_list, vehicle_edges_list)


    def cal_KQ_knearest(self):
        pass

    def cal_KQ(self, infra_edges_list, vehicle_edges_list):
        self.KQ = np.zeros((len(infra_edges_list), len(vehicle_edges_list)), dtype=np.float64)

        for i, infra_edge in enumerate(infra_edges_list):
            for j, vehicle_edge in enumerate(vehicle_edges_list):
                
                infra_edge_bbox = (self.infra_bboxes_object_list[infra_edge[0]].get_bbox3d_8_3(), self.infra_bboxes_object_list[infra_edge[1]].get_bbox3d_8_3())
                vehicle_edge_bbox = (self.vehicle_bboxes_object_list[vehicle_edge[0]].get_bbox3d_8_3(), self.vehicle_bboxes_object_list[vehicle_edge[1]].get_bbox3d_8_3())

                if self.infra_bboxes_object_list[infra_edge[0]].get_bbox_type() != self.vehicle_bboxes_object_list[vehicle_edge[0]].get_bbox_type() \
                    or self.infra_bboxes_object_list[infra_edge[1]].get_bbox_type() != self.vehicle_bboxes_object_list[vehicle_edge[1]].get_bbox_type():
                    self.KQ[i, j] = 0
                    continue

                # 检测框大小
                similarity_infra_size = similarity_utils.cal_similarity_size(infra_edge_bbox[0], vehicle_edge_bbox[0])
                similarity_vehicle_size = similarity_utils.cal_similarity_size(infra_edge_bbox[1], vehicle_edge_bbox[1])
                similarity_size = similarity_infra_size * similarity_vehicle_size
                    
                # 边长
                similarity_length = similarity_utils.cal_similarity_length(infra_edge_bbox, vehicle_edge_bbox)

                # 角度
                similarity_angle = similarity_utils.cal_similarity_angle(infra_edge_bbox, vehicle_edge_bbox)
                
                self.KQ[i, j] = similarity_size * similarity_length * similarity_angle * 10
                self.KQ[i, j] = int(self.KQ[i, j])
                if self.KQ[i, j] < 0:
                    self.KQ[i, j] = 0



    def plot_box_diagram_distribution(self, original_data):
        i = original_data.nonzero()
        data = original_data[i].flatten()
        print('before nonzero')
        print(len(original_data.flatten()))
        print('after nonzero')
        print(len(data))
        fig, ax = plt.subplots()
        ax.boxplot(data, patch_artist=True)
        plt.show()

    def cal_Ct(self):
        pass

    def cal_asgTX(self):
        pass


    def assignment_example(self):
        self.KP = np.array([[1, 1, 4], [10, 3, 2], [2, 9, 3], [1, 2, 8]], dtype=np.float64)
        self.KQ = np.array([[5, 2, 4, 1], [9, 4, 8, 5], [3, 10, 2, 9], [4, 1, 5, 2], [8, 5, 9, 4], [2, 9, 3, 10]], dtype=np.float64)
        self.Ct = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float64)
        self.asgTX = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        self.gph1 = {}
        self.gph2 = {}
        self.gph1['G'] = np.array([[1, 0, 0, 0, 0, 0], 
                                    [0, 1, 0, 1, 0, 0], 
                                    [0, 0, 1, 0, 1, 0], 
                                    [0, 0, 0, 0, 0, 1]], dtype=np.float64)
        self.gph1['H'] = np.array([[0, 0, 0, 1, 0, 0], 
                                    [1, 0, 0, 0, 1, 0], 
                                    [0, 1, 0, 0, 0, 1], 
                                    [0, 0, 1, 0, 0, 0]], dtype=np.float64)
        self.gph2['G'] = np.array([[1, 0, 0, 0],
                                    [0, 1, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float64)
        self.gph2['H'] = np.array([[0, 0, 1, 0],
                                    [1, 0, 0, 1],
                                    [0, 1, 0, 0]], dtype=np.float64)
        # self.gph1['graph'] = np.array([[0, 1, 0, 0],
        #                                 [1, 0, 1, 0],
        #                                 [0, 1, 0, 1],
        #                                 [0, 0, 1, 0]], dtype=np.float64)
        
        
    def execute(self):
        self.cal_KP()
        # self.cal_KQ_MST()
        print('start cal_KQ_FULLY_CONNECTED')
        self.cal_KQ_FULLY_CONNECTED()
        print('end cal_KQ_FULLY_CONNECTED')
        self.cal_Ct()
        self.cal_asgTX()

        print('start fgm')
        result, confidence = fgm.solve(self.KP, self.KQ, self.Ct, self.asgTX, self.gph1, self.gph2)
        print('fgm done')

        output_dir = './output'
        np.savetxt(f"{output_dir}/result.csv", result, delimiter=",", fmt='%d')

        print(result.tolist())
        print(confidence)


    def output_intermediate_result(self):
        self.cal_KP()
        self.cal_KQ_FULLY_CONNECTED()

        output_dir = './output'

        # Save matrices KP, KQ, Ct, asgTX
        np.savetxt(f"{output_dir}/KP.csv", self.KP, delimiter=",", fmt='%d')
        np.savetxt(f"{output_dir}/KQ.csv", self.KQ, delimiter=",", fmt='%d')
        np.savetxt(f"{output_dir}/Ct.csv", self.Ct, delimiter=",", fmt='%d')
        np.savetxt(f"{output_dir}/asgTX.csv", self.asgTX, delimiter=",", fmt='%d')

        # Save maps of matrices gph1, gph2
        for key, matrix in self.gph1.items():
            np.savetxt(f"{output_dir}/gph1_{key}.csv", matrix, delimiter=",", fmt='%d')
            # pd.DataFrame(matrix).to_csv(f"{output_dir}/gph1_{key}.csv", index=False, header=False)
        for key, matrix in self.gph2.items():
            np.savetxt(f"{output_dir}/gph2_{key}.csv", matrix, delimiter=",", fmt='%d')
            # pd.DataFrame(matrix).to_csv(f"{output_dir}/gph2_{key}.csv", index=False)

    def get_assignment_from_KP(self, topK = 5):
        filename = './output/KP.csv'
        KP = pd.read_csv(filename, header=None)
        indices_and_values = [(index, value) for index, value in np.ndenumerate(KP)]
        topK_matches = sorted(indices_and_values, key=lambda x: x[1], reverse=True)[:topK]
        return topK_matches

    @get_time
    def test_time_get_assignment_from_KP(self, topK = 5):
        self.cal_KP()
        indices_and_values = [(index, value) for index, value in np.ndenumerate(self.KP)]
        topK_matches = sorted(indices_and_values, key=lambda x: x[1], reverse=True)[:topK]
        return topK_matches


    def get_assignment_result(self):
        filename = './output/result.csv'
        result = pd.read_csv(filename, header=None)
        matches = []
        for i, row in result.iterrows():
            for j, value in enumerate(row):
                if value == 1:
                    matches.append((i, j))

        return matches

    def test_given_matches_visualization_view(self, matches):
        matched_infra_bboxes_object_list = []
        matched_vehicle_bboxes_object_list = []

        # cnt = 0

        for match in matches:
            # if cnt != 0:
            #     continue
            # cnt += 1
            matched_infra_bboxes_object_list.append(task.infra_bboxes_object_list[match[0]])
            matched_vehicle_bboxes_object_list.append(task.vehicle_bboxes_object_list[match[1]])

        for select_num in range(len(matched_infra_bboxes_object_list)):

            # print(len(matched_infra_bboxes_object_list))
            # print(len(matched_vehicle_bboxes_object_list))

            bbox_visualizer = BBoxVisualizer()
            # converted_matched_infra_bboxes_object_list = CoordinateConversion().convert_bboxes_object_list_infra_lidar_2_vehicle_lidar(matched_infra_bboxes_object_list)
            
            selected_infra_bboxes_object_list = [matched_infra_bboxes_object_list[select_num]]
            selected_vehicle_bboxes_object_list = [matched_vehicle_bboxes_object_list[select_num]]
            
            bbox_visualizer.plot_boxes3d_lists([matched_vehicle_bboxes_object_list, matched_infra_bboxes_object_list, selected_vehicle_bboxes_object_list, selected_infra_bboxes_object_list], 
                                                [(1, 0, 0), (0, 1, 0), (0, 0, 0), (0, 0, 1)])


            # converted_infra_bboxes_object_list = CoordinateConversion().convert_bboxes_object_list_infra_lidar_2_vehicle_lidar(task.infra_bboxes_object_list)
            # bbox_visualizer.plot_boxes3d_lists([selected_infra_bboxes_object_list, converted_infra_bboxes_object_list, selected_vehicle_bboxes_object_list, task.vehicle_bboxes_object_list], 
            #                                 [(0, 0, 0), (0, 1, 0), (0, 0, 0),(1, 0, 0)])

    def test_matches_visualization_view(self):
        matches = task.get_assignment_result()
        self.test_given_matches_visualization_view(matches)
        


if __name__ == "__main__":
    task = FGMTask()
    # task.execute()

    # task.output_intermediate_result()

    # task.test_matches_visualization_view()

    # matches = GenerateCorrespondingListTask('config.yml').load_infra_vehicle_corresponding_list()
    # task.test_given_matches_visualization_view(matches)

    # task.get_assignment_from_KP()

    task.test_time_get_assignment_from_KP()