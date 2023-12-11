import math
import numpy as np
import sys
sys.path.append('./reader')
sys.path.append('./process')
sys.path.append('./process/utils')
sys.path.append('./process/corresponding')
from extrinsic_utils import convert_T_to_6DOF, convert_6DOF_to_T, get_time, implement_T_3dbox_object_list
from CorrespondingDetector import CorrespondingDetector

# 尝试解构搜索范围，使其可以支持离散形式的搜索范围 

class PSO_deconstructX():

    def __init__(self, infra_box_object_list, vehicle_box_object_list, true_T_6DOF_format = None, init_T_list = None, n_dim=6, pop=40, max_iter=150, 
                 lb=[-100, -100, -10, -math.pi, -math.pi, -math.pi], ub=[100, 100, 10, math.pi, math.pi, math.pi], 
                 w=(0.8, 0.8), c1=0.5, c2=0.5, constraint_eq=tuple(), constraint_ueq=tuple(), verbose=True):
        
        self.infra_box_object_list = infra_box_object_list
        self.vehicle_box_object_list = vehicle_box_object_list

        self.true_T_6DOF_format = true_T_6DOF_format
 
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
        assert np.all(self.ub >= self.lb), 'upper-bound must be greater than lower-bound'

        self.has_constraint = bool(constraint_ueq)
        self.constraint_ueq = constraint_ueq
        self.constraint_eq = constraint_eq

        self.X = np.random.uniform(low=self.lb, high=self.ub, size=(self.pop, self.n_dim))
        if init_T_list is not None:
            init_num = max(init_T_list.shape[0], self.pop)
            for i, init_T in enumerate(init_T_list):
                if init_num <= 0:
                    break
                init_num -= 1
                self.X[i, :] = init_T

        v_high = [ub_i - lb_i for ub_i, lb_i in zip(self.ub, self.lb)]
        self.V = np.random.uniform(low=[-v for v in v_high], high=v_high, size=(self.pop, self.n_dim))  # speed of particles
        self.Y = np.zeros((pop, 1), dtype=np.float64)
        self.cal_y()  # y = f(x) for all particles
        self.pbest_x = np.array(self.X.copy())  # personal best location of every particle in history
        self.pbest_y = np.zeros((pop, 1), dtype=np.float64)  # best image of every particle in history
        self.update_pbest()
        self.gbest_x = self.pbest_x.mean(axis=0)  # global best location for all particles
        self.gbest_y = 0  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.update_gbest()
        if self.verbose:
            print('Iter: -, Best fit: {} at {}'.format(self.gbest_y, self.gbest_x))
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

        for idx, X in enumerate(self.X):
            T = convert_6DOF_to_T(X)
            converted_infra_box_object_list_copy = implement_T_3dbox_object_list(T, self.infra_box_object_list)
            self.Y[idx][0] = CorrespondingDetector(converted_infra_box_object_list_copy, self.vehicle_box_object_list).get_Yscore()
        

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

    # @get_time
    def run(self, max_iter=None, precision=1e-2, N=5):
        '''
        precision: None or float
            If precision is None, it will run the number of max_iter steps
            If precision is a float, the loop will stop if continuous N difference between pbest less than precision
        N: int
        '''
        self.max_iter = max_iter or self.max_iter
        c = 0
        last_gbest_y = self.gbest_y
        for iter_num in range(self.max_iter):
            self.update_w(iter_num)
            self.update_V()
            self.recorder()
            self.update_X()
            self.cal_y()
            self.update_pbest()
            self.update_gbest()

            if precision is not None:
                tor_iter = self.gbest_y - last_gbest_y
                last_gbest_y = self.gbest_y
                if tor_iter < precision:
                    c = c + 1
                    if c >= N:
                        break
                else:
                    c = 0

            # if precision is not None:
            #     tor_iter = np.amax(self.pbest_y) - np.amin(self.pbest_y)
            #     if tor_iter < precision:
            #         c = c + 1
            #         if c > N:
            #             break
            #     else:
            #         c = 0
            if self.verbose:
                print('Iter: {}, Best fit: {} at {} '.format(iter_num, self.gbest_y, self.gbest_x))
                if self.true_T_6DOF_format is not None:
                    print('delta_true: {}'.format(self.true_T_6DOF_format - self.gbest_x))
                    
            self.gbest_y_hist.append(self.gbest_y)
        self.best_x, self.best_y = self.gbest_x, self.gbest_y
        return self.best_x, self.best_y
