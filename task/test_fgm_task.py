import numpy as np
import module.fgm as fgm

import sys
sys.path.append('./reader')

import CooperativeReader
import module.similarity_utils as similarity_utils

class FGMTask():

    '''
        v1.0: 逆向同权的有向边、全连接图
    '''

    def __init__(self):
        self.reader = CooperativeReader.CooperativeReader('config.yml')
        self.infra_bboxes_object_list, self.vehicle_bboxes_object_list = self.reader.get_cooperative_infra_vehicle_bboxes_object_list()
        # print(self.infra_bboxes_object_list, self.vehicle_bboxes_object_list)

        infra_node_num = len(self.infra_bboxes_object_list)
        infra_edge_num = infra_node_num #* (infra_node_num - 1) 
        vehicle_node_num = len(self.vehicle_bboxes_object_list)
        vehicle_edge_num = vehicle_node_num #* (vehicle_node_num - 1)

        self.KP = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float64)
        self.KQ = np.ones((infra_edge_num, vehicle_edge_num), dtype=np.float64)
        self.Ct = np.ones((infra_node_num, vehicle_node_num), dtype=np.float64)
        self.asgTX = np.zeros((infra_node_num, vehicle_node_num), dtype=np.float64)
        self.gph1 = {}
        self.gph2 = {}
        # self.gph1['G'], self.gph1['H'] = self.trans_graph_GH(self.get_infra_graph_structure())
        # self.gph2['G'], self.gph2['H'] = self.trans_graph_GH(self.get_vehicle_graph_structure())
        

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

        # Count number of ones in the adj. matrix to get number of edges
        nr_edges = np.sum(A).astype(np.int32)

        # Init G and H
        G = np.zeros((n, nr_edges), dtype=np.float64)
        H = np.zeros((n, nr_edges), dtype=np.float64)

        # Get all non-zero entries and build G and H
        entries = np.transpose((A != 0).nonzero()) 
        for count, (i,j) in enumerate(entries, start=0):
            G[i, count] = 1
            H[j, count] = 1

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


    def cal_KQ(self):
        '''
            边的编号从 graph_structure 中入手
            从行入手
            结合 trans_graph_GH 中用 nonzero 搜边的策略，作解耦和重构
        
        '''
        # 边的编号？
        infra_node_cnt = 0
        for i1, infra_bbox_object1 in enumerate(self.infra_bboxes_object_list):
            for i2, infra_bbox_object2 in enumerate(self.infra_bboxes_object_list):
                if i1 == i2:
                    continue
                vehicle_node_cnt = 0
                for j1, vehicle_bbox_object1 in enumerate(self.vehicle_bboxes_object_list):
                    for j2, vehicle_bbox_object2 in enumerate(self.vehicle_bboxes_object_list):
                        if j1 == j2:
                            continue
                        if infra_bbox_object1.get_bbox_type() != vehicle_bbox_object1.get_bbox_type() or infra_bbox_object2.get_bbox_type() != vehicle_bbox_object2.get_bbox_type():
                            self.KQ[infra_node_cnt, vehicle_node_cnt] = 0
                            continue

                        infra_edge = (infra_bbox_object1.get_bbox3d_8_3(), infra_bbox_object2.get_bbox3d_8_3())
                        vehicle_edge = (vehicle_bbox_object1.get_bbox3d_8_3(), vehicle_bbox_object2.get_bbox3d_8_3())

                        # 检测框大小
                        similarity_infra_size = similarity_utils.cal_similarity_size(infra_edge[0], vehicle_edge[0])
                        similarity_vehicle_size = similarity_utils.cal_similarity_size(infra_edge[1], vehicle_edge[1])
                        similarity_size = similarity_infra_size * similarity_vehicle_size
                        
                        # 边长
                        # similarity_length = similarity_utils.cal_similarity_length(infra_edge, vehicle_edge)

                        # 角度
                        # similarity_angle = similarity_utils.cal_similarity_angle(infra_edge, vehicle_edge)
                        
                        self.KQ[infra_node_cnt, vehicle_node_cnt] = similarity_size #* similarity_length * similarity_angle
                        vehicle_node_cnt += 1
                
                infra_node_cnt += 1

                

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
        print('cal_KP done')
        # self.cal_KQ()
        self.cal_Ct()
        self.cal_asgTX()

        # self.assignment_example()

        print('start fgm')
        result, confidence = fgm.solve(self.KP, self.KQ, self.Ct, self.asgTX, self.gph1, self.gph2)
        print('fgm done')

        print(result)
        print(confidence)

if __name__ == "__main__":
    task = FGMTask()
    task.execute()
