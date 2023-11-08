#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/8/20
# @Author  : github.com/guofei9987

import math
import numpy as np
from utils import convert_T_to_6DOF, convert_6DOF_to_T


class PSO():
    """
    Do PSO (Particle swarm optimization) algorithm.

    This algorithm was adapted from the earlier works of J. Kennedy and
    R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

    The position update can be defined as:

    .. math::

       x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

    Where the position at the current step :math:`t` is updated using
    the computed velocity at :math:`t+1`. Furthermore, the velocity update
    is defined as:

    .. math::

       v_{ij}(t + 1) = w * v_{ij}(t) + c_{p}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                       + c_{g}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

    Here, :math:`cp` and :math:`cg` are the cognitive and social parameters
    respectively. They control the particle's behavior given two choices: (1) to
    follow its *personal best* or (2) follow the swarm's *global best* position.
    Overall, this dictates if the swarm is explorative or exploitative in nature.
    In addition, a parameter :math:`w` controls the inertia of the swarm's
    movement.

    .. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.

    Parameters
    --------------------
    func : function
        The func you want to do optimal
    dim : int
        Number of dimension, which is number of parameters of func.
    pop : int
        Size of population, which is the number of Particles. We use 'pop' to keep accordance with GA
    max_iter : int
        Max of iter iterations
    lb : array_like
        The lower bound of every variables of func
    ub : array_like
        The upper bound of every variables of func
    constraint_eq : tuple
        equal constraint. Note: not available yet.
    constraint_ueq : tuple
        unequal constraint
    Attributes
    ----------------------
    pbest_x : array_like, shape is (pop,dim)
        best location of every particle in history
    pbest_y : array_like, shape is (pop,1)
        best image of every particle in history
    gbest_x : array_like, shape is (1,dim)
        general best location for all particles in history
    gbest_y : float
        general best image  for all particles in history
    gbest_y_hist : list
        gbest_y of every iteration


    """

#
##
# r1 r2 每一次都随机
# W 固定
# cp cg 固定
# 构建一个渐变模型，对不同IoU水平给出不同移动步长
##
#

    def __init__(self, func, infra_boxes, infra_boxes_precision, vehicle_boxes_precision, vehicle_boxes, init_T = np.eye(4), n_dim=6, pop=40, max_iter=150, lb=[-100, -100, -10, -math.pi, -math.pi, -math.pi], ub=[100, 100, 10, math.pi, math.pi, math.pi], v_max_scope_rate=1,
                 w=(0.8, 0.8), c1=0.5, c2=0.5, constraint_eq=tuple(), constraint_ueq=tuple(), verbose=True):

        n_dim = n_dim 
        
        self.infra_boxes = infra_boxes
        self.vehicle_boxes = vehicle_boxes
        self.infra_boxes_precision = infra_boxes_precision
        self.vehicle_boxes_precision = vehicle_boxes_precision

        self.func = func
 
        self.w_max, self.w_min = w
        self.w = self.w_max  # inertia
        
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        # self.w, self.cp, self.cg = 0, 0, 0
        self.pop = pop  # number of particles
        self.n_dim = n_dim  # dimension of particles, which is the number of variables of func
        self.max_iter = max_iter  # max iter
        self.verbose = verbose  # print the result of each iter or not

        self.lb, self.ub = lb, ub
        assert self.n_dim == len(self.lb) == len(self.ub), 'dim == len(lb) == len(ub) is not True'
        assert np.all(self.ub > self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.constraint_eq = constraint_eq

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        self.X[0, :] = convert_T_to_6DOF(init_T)
        # self.X = np.array(convert_T_to_6DOF(init_T)).reshape(1, -1).repeat(self.pop, axis=0)
        v_high = [(ub_i - lb_i) / v_max_scope_rate for ub_i, lb_i in zip(self.ub, self.lb)]
        self.V = np.random.uniform(low=[-v for v in v_high], high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        # self.V = np.zeros((self.pop, self.n_dim), dtype=np.float64)  # speed of particles
        self.Y = np.zeros((pop, 1), dtype=np.float64)
        self.cal_y()  # y = f(x) for all particles
        self.pbest_x = np.array(self.X.copy())  # personal best location of every particle in history
        self.pbest_y = np.zeros((pop, 1), dtype=np.float64)  # best image of every particle in history
        self.update_pbest()
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(1, -1)  # global best location for all particles
        self.gbest_y = 0  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()
        if self.verbose:
            print('Iter: 0, Best fit: {} at {}'.format(self.gbest_y, self.gbest_x))
        self.gbest_y_hist.append(self.gbest_y)

        # record verbose values
        self.record_mode = False
        self.record_value = {'X': [], 'V': [], 'Y': []}
        
        self.best_x, self.best_y = self.gbest_x, self.gbest_y  # history reasons, will be deprecated
        

    


    def check_constraint(self, x):
        # gather all unequal constraint functions
        for constraint_func in self.constraint_ueq:
            if constraint_func(x) > 0:
                return False
        return True

    def update_w(self, iter_num):
        self.w = self.w_max - (self.w_max - self.w_min) * iter_num / self.max_iter

    def update_V(self):
        r1 = np.random.rand(self.pop, self.n_dim)
        r2 = np.random.rand(self.pop, self.n_dim)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.X) + \
                 self.cg * r2 * (self.gbest_x - self.X)

    def update_X(self):
        self.X = self.X + self.V
        self.X = np.clip(self.X, self.lb, self.ub)

    def cal_y(self):
        # calculate y for every x in X
        # self.Y = self.func(self.X).reshape(-1, 1)
        
        # 外参处理的两种策略：
        # 1. 每次去掉上一次外参的，再乘以当前的外参
        # 2. 每次乘上一个微小的变化量，最后把这个变化量叠起来构成整体变化量
        # 选择第一种策略

        def cancel_previous_X_T(X_T_formated):
            # vehicle2world
            for box_type, boxes in self.infra_boxes.items():
                R = X_T_formated[:3, :3]
                t = X_T_formated[:3, 3]

                rev_R = np.array(np.matrix(R).I)
                rev_t = -np.dot(rev_R, t)

                boxes = boxes.reshape(-1, 3).T
                boxes = np.dot(rev_R, boxes) + rev_t.reshape(3, 1)
                boxes = boxes.T.reshape(-1, 8, 3)
                self.infra_boxes[box_type] = boxes

        def implement_X_T(X_T_formated):
            # infra2vehicle
            for box_type, boxes in self.infra_boxes.items():
                boxes = boxes.reshape(-1, 3).T
                R = X_T_formated[:3, :3]
                t = X_T_formated[:3, 3]
                boxes = np.dot(R, boxes) + t.reshape(3, 1)
                boxes = boxes.T.reshape(-1, 8, 3)
                self.infra_boxes[box_type] = boxes

        def cal_single_Y():
            # 对应关系怎么搞，给干沉默了
            
            # 类别严格对应吗？如果是会不会有点难找到初值
            # 总之先按照严格对应写一版

            type_list = set(list(self.infra_boxes.keys()) + list(self.vehicle_boxes.keys()))
            
            Y = 0

            IoU_matrix_list = []

            for box_type in type_list:
                if box_type not in self.infra_boxes or box_type not in self.vehicle_boxes:
                    continue

                IoU_sum = 0
                mutual_cnt = 0
                IoU_matrix = np.zeros((len(self.infra_boxes[box_type]), len(self.vehicle_boxes[box_type])))
                
                for i, infra_box in enumerate(self.infra_boxes[box_type]):
                    for j, vehicle_box in enumerate(self.vehicle_boxes[box_type]):
                        # boxes shape(8 ,3)

                        

                        IoU_matrix[i, j] = self.func(infra_box, vehicle_box) 

                        
                        if IoU_matrix[i, j] > 0:
                            IoU_sum += IoU_matrix[i, j]
                            mutual_cnt += 1

                # print(IoU_matrix)
                if mutual_cnt != 0: 
                    IoU_matrix_list.append(IoU_sum / mutual_cnt)
                else:
                    IoU_matrix_list.append(0)
            
            if len(IoU_matrix_list) != 0:
                Y = np.mean(IoU_matrix_list)

            # 隐患：杂项的IoU会被算进去

            return Y

        

        for idx, X in enumerate(self.X):
            # temp_infra_boxes = self.infra_boxes.copy()

            T = convert_6DOF_to_T(X)
            # print('before implement:')
            # print(self.infra_boxes['Bus'])
            implement_X_T(T)
            self.Y[idx][0] = cal_single_Y()
            cancel_previous_X_T(T)
            # print('after cancel:')
            # print(self.infra_boxes['Bus'])
        

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y < self.Y
        for idx, x in enumerate(self.X):
            if self.need_update[idx]:
                self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(self.need_update, self.X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.Y, self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_max = self.pbest_y.argmax()
        if self.gbest_y < self.pbest_y[idx_max]:
            self.gbest_x = self.X[idx_max, :].copy()
            self.gbest_y = self.pbest_y[idx_max]

    def recorder(self):
        if not self.record_mode:
            return
        self.record_value['X'].append(self.X)
        self.record_value['V'].append(self.V)
        self.record_value['Y'].append(self.Y)

    def run(self, max_iter=None, precision=1e-4, N=20):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        for iter_num in range(self.max_iter):
            self.update_w(iter_num)
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()
            if precision is not None:
                tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
                if tor_iter < precision:
                    c = c + 1
                    if c > N:
                        break
                else:
                    c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {}'.format(iter_num, self.gbest_y, self.gbest_x))

            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y

