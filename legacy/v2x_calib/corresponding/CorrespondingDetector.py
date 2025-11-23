import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from collections import defaultdict
import numpy as np
from sklearn.neighbors import KDTree
from ..utils import cal_3dIoU, get_volume_from_bbox3d_8_3, get_xyz_from_bbox3d_8_3

class CorrespondingDetector():
    '''
    CorrespondingDetector is a class to obtain the extent of spartial alignment between two sets of bounding boxes.
    Here we hypothesize that the two sets of bounding boxes are already transformered by the extrinsic matrix.
    param:
        core_similarity_component: 'iou' or 'centerpoint_distance' or 'vertex_distance' or 'overall_distance'
    '''
    def __init__(self, infra_bboxes_object_list, vehicle_bboxes_object_list, core_similarity_component = 'overall_distance', distance_threshold=3, parallel=False):
        self.infra_bboxes_object_list = infra_bboxes_object_list
        self.vehicle_bboxes_object_list = vehicle_bboxes_object_list

        self.corresponding_score_dict = {}
        self.Y = 0

        if core_similarity_component == 'iou':
            self.cal_IoU_corresponding()
        elif core_similarity_component == 'centerpoint_distance':
            if parallel:
                self.cal_distance_corresponding_parallel(distance_strategy = ['centerpoint'], distance_threshold_ = distance_threshold)
            else:
                self.cal_distance_corresponding(distance_strategy = 'centerpoint', distance_threshold_ = distance_threshold)
        elif core_similarity_component == 'vertex_distance':
            if parallel:
                self.cal_distance_corresponding_parallel(distance_strategy = ['vertexpoint'], distance_threshold_ = distance_threshold)
            else:
                self.cal_distance_corresponding(distance_strategy = 'vertexpoint', distance_threshold_ = distance_threshold)
        elif core_similarity_component == 'overall_distance':
            if parallel:
                self.cal_distance_corresponding_parallel(distance_strategy = ['centerpoint', 'vertexpoint'], distance_threshold_ = distance_threshold)
            else:
                self.cal_distance_corresponding(distance_strategy = ['centerpoint', 'vertexpoint'], distance_threshold_ = distance_threshold)
        else:
            raise ValueError('core_similarity_component should be one of the following: [IoU, centerpoint_distance, vertex_distance, overall_distance]')


    def cal_IoU_corresponding(self):
        for i, infra_bbox_object in enumerate(self.infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(self.vehicle_bboxes_object_list):
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    box3d_IoU_score = cal_3dIoU(infra_bbox_object.get_bbox3d_8_3(), vehicle_bbox_object.get_bbox3d_8_3())
                    if box3d_IoU_score > 0:
                        # infra_volume = get_volume_from_bbox3d_8_3(infra_bbox_object.get_bbox3d_8_3())
                        # vehicle_volume = get_volume_from_bbox3d_8_3(vehicle_bbox_object.get_bbox3d_8_3())
                        # volume = (infra_volume + vehicle_volume) / 2
                        self.corresponding_score_dict[(i, j)] = box3d_IoU_score
                        # if volume >= 1:
                        #     self.corresponding_score_dict[(i, j)] = box3d_IoU_score / volume * 10
        if len(self.corresponding_score_dict) != 0:
            self.Y = np.sum(list(self.corresponding_score_dict.values()))

    def cal_distance_corresponding(self, distance_strategy = 'centerpoint', distance_threshold_ = {}):
        '''
        为了让整体 distance 作为 score 的时候 score 越大表示效果越好，对 distance 取负数，一对一匹配对约束 和 阈值筛选 获取匹配对
        '''
        occupation_dict = {}
        distance_threshold = {}
        for type, threshold in distance_threshold_.items():
            distance_threshold[type] = -threshold
        for i, infra_bbox_object in enumerate(self.infra_bboxes_object_list):
            for j, vehicle_bbox_object in enumerate(self.vehicle_bboxes_object_list):
                update_flag = False
                if infra_bbox_object.get_bbox_type() == vehicle_bbox_object.get_bbox_type():
                    distance = 0
                    if distance_strategy == 'centerpoint' or 'centerpoint' in distance_strategy:
                        infra_bbox_centerpoint = get_xyz_from_bbox3d_8_3(infra_bbox_object.get_bbox3d_8_3())
                        vehicle_bbox_centerpoint = get_xyz_from_bbox3d_8_3(vehicle_bbox_object.get_bbox3d_8_3())
                        distance += -np.linalg.norm(infra_bbox_centerpoint - vehicle_bbox_centerpoint)
                    if distance_strategy == 'vertexpoint' or 'vertexpoint' in distance_strategy:
                        infra_bbox_vertex = infra_bbox_object.get_bbox3d_8_3()
                        vehicle_bbox_vertex = vehicle_bbox_object.get_bbox3d_8_3()
                        distance += -np.linalg.norm(infra_bbox_vertex - vehicle_bbox_vertex) / 8

                    if 'centerpoint' in distance_strategy and 'vertexpoint' in distance_strategy:
                        distance /= 2

                    # print(f'{i} - {j} inf_type:{infra_bbox_object.get_bbox_type()} veh_type:{vehicle_bbox_object.get_bbox_type()} distance: {distance}')

                    # 潜在空间换时间的策略
                    if distance <= distance_threshold[infra_bbox_object.get_bbox_type()] :
                        continue
                    elif i in occupation_dict.keys():
                        if distance <= self.corresponding_score_dict[(i, occupation_dict[i])]:
                            continue
                        else:
                            update_flag = True
                            del self.corresponding_score_dict[(i, occupation_dict[i])]
                            # del occupation_dict[i]
                    elif j in occupation_dict.values():
                        deleting = []
                        for k, v in occupation_dict.items():
                            if v == j:
                                if distance <= self.corresponding_score_dict[(k, j)]:
                                    break
                                else:
                                    update_flag = True
                                    del self.corresponding_score_dict[(k, j)]
                                    deleting.append(k)
                        for k in deleting:
                            del occupation_dict[k]
                    else:
                        update_flag = True

                    if update_flag:
                        # if i in occupation_dict.keys():
                        #     print(f'primary: {i} - {occupation_dict[i]} distance: {self.corresponding_score_dict[(i, occupation_dict[i])]}')
                        # else:
                        #     print(f'{i} has no primary key')
                        # print(f'update: {i} - {j} distance: {distance}')
                        self.corresponding_score_dict[(i, j)] = distance
                        occupation_dict[i] = j

        if len(self.corresponding_score_dict) != 0:
            self.Y = np.sum(list(self.corresponding_score_dict.values()))


    def cal_distance_corresponding_parallel(self,
                                            distance_strategy='centerpoint',
                                            distance_threshold_={}):
        """
        基于 KD-Tree 的加速一对一匹配：
        1) 按类别分组
        2) KD-Tree 查找候选
        3) 全局贪心配对
        """
        # 1. 负阈值（为了 score 越大越好）
        distance_threshold = {t: -thr for t, thr in distance_threshold_.items()}

        # 2. 按 type 分组收集 centerpoints
        infra_by_type = defaultdict(list)
        veh_by_type   = defaultdict(list)
        for i, inf in enumerate(self.infra_bboxes_object_list):
            infra_by_type[inf.get_bbox_type()].append(i)
        for j, veh in enumerate(self.vehicle_bboxes_object_list):
            veh_by_type[veh.get_bbox_type()].append(j)

        candidates = []  # 存 (i, j, score)

        # 3. 针对每个类型做 KD-Tree 查询
        for t in infra_by_type:
            if t not in veh_by_type:
                continue
            infra_idxs = infra_by_type[t]
            veh_idxs   = veh_by_type[t]

            # 提取点云
            P = np.stack([
                get_xyz_from_bbox3d_8_3(self.infra_bboxes_object_list[i].get_bbox3d_8_3())
                for i in infra_idxs
            ], axis=0)  # Ni×3

            Q = np.stack([
                get_xyz_from_bbox3d_8_3(self.vehicle_bboxes_object_list[j].get_bbox3d_8_3())
                for j in veh_idxs
            ], axis=0)  # Mi×3

            # KD-Tree 建树
            tree = KDTree(Q, leaf_size=40)

            # 对每个 infra 点，查询所有在阈值内的 vehicle 点
            thr = abs(distance_threshold[t])  # 正值半径
            for idx_p, i in enumerate(infra_idxs):
                # query_radius 返回索引列表
                neighbors = tree.query_radius(P[[idx_p]], r=thr)[0]
                if len(neighbors) == 0:
                    continue

                for nb in neighbors:
                    j = veh_idxs[nb]
                    # 计算混合距离 score
                    score = 0.0
                    if 'centerpoint' in distance_strategy:
                        score += -np.linalg.norm(P[idx_p] - Q[nb])
                    if 'vertexpoint' in distance_strategy:
                        inf_v = self.infra_bboxes_object_list[i].get_bbox3d_8_3()
                        veh_v = self.vehicle_bboxes_object_list[j].get_bbox3d_8_3()
                        score += -np.linalg.norm(inf_v - veh_v) / 8
                    if 'centerpoint' in distance_strategy and 'vertexpoint' in distance_strategy:
                        score /= 2

                    # 只保留超过阈值的
                    if score > distance_threshold[t]:
                        candidates.append((i, j, score))

        # 4. 全局按 score 排序，贪心一对一匹配
        candidates.sort(key=lambda x: x[2], reverse=True)
        self.corresponding_score_dict.clear()
        occupied_i = set()
        occupied_j = set()

        for i, j, score in candidates:
            if i in occupied_i or j in occupied_j:
                continue
            occupied_i.add(i)
            occupied_j.add(j)
            self.corresponding_score_dict[(i, j)] = score

        if len(self.corresponding_score_dict) != 0:
            self.Y = np.sum(list(self.corresponding_score_dict.values()))


    def get_matched_num(self):
        return len(self.corresponding_score_dict)
    
    def get_matches(self):
        return self.corresponding_score_dict.keys()
    
    def get_matches_with_score(self):
        return self.corresponding_score_dict
    
    def get_total_num(self):
        return min(len(self.infra_bboxes_object_list), len(self.vehicle_bboxes_object_list))
    
    def get_Yscore(self):
        total_num = self.get_total_num()
        if total_num == 0:
            return 0
        # return self.Y / self.get_total_num()
        return self.get_matched_num() - 1
    
    def get_distance_corresponding_precision(self):
        if self.get_matched_num() == 0:
            return -3
        return self.Y / self.get_matched_num()
