import argparse
import math
from typing import Tuple, Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor

class CBM():
    # All the transform are defined in right hand coordinates

    def __init__(self, args=None):
        if args == None:
            self.args = self.parser()
        else:
            self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # def parser(self):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--sigma1', default=10 * math.pi / 180,
    #                         help='rad')
    #     parser.add_argument('--sigma2', default=3,
    #                         help='m')
    #     parser.add_argument('--sigma3', default=1,
    #                         help='m')
    #     parser.add_argument('--absolute_dis_lim', default=20,
    #                         help='m')
    #     args = parser.parse_args()

    #     return args

    def parser(self):
        pass

    def __call__(self, *args: torch.Tensor)-> torch.Tensor:
        #Input  Ego: torch.Tensor, Cav: torch.Tensor, transform: torch.Tensor
        args += (torch.eye(4),) if len(args) == 2 else ()
        args = self.check_numpy_to_torch(args)

        Ego, Cav, transform = args[0], args[1], args[2]
        Ego, Cav = Ego.to(self.device), Cav.to(self.device)

        # Unify the orientation
        Ego, Cav = self.Uni_Ori(Ego, Cav, transform)

        # Construct local context
        P, Q = self.CLC(Ego), self.CLC(Cav)
        self.m, self.n = len(P), len(Q)

        # Local matching
        M, M_G = self.LM(P, Q)
       
        # Global matching
        A_ = self.GM(M, M_G, Ego, Cav)
        
        # Convert matching matrix to the form [[i,j]]
        m = torch.where(A_ > 0)
        matching = torch.hstack((m[0].reshape(-1, 1), m[1].reshape(-1, 1)))

        return matching

    def check_numpy_to_torch(self,args: tuple) -> tuple:
        args_=()
        for (i,j) in enumerate(args):
            if isinstance(j, np.ndarray):
                args_ += (torch.tensor(j, dtype= torch.float32).to(self.device),)
            else:
                args_ += (j.to(self.device),)
        return args_

    def Uni_Ori(self, Ego: torch.Tensor, Cav: torch.Tensor, transform: torch.Tensor) -> Tuple[Tensor, Tensor]:
        # Ego, Cav: Mx7, Nx7
        R, t = transform[:-1, :-1], transform[:-1, -1].reshape(-1, 1)
        angle = Rotation.from_matrix(R.cpu()).as_euler('xyz', degrees=False)
        R, t, angle = R.to(self.device), t.to(self.device), torch.Tensor(angle)

        Cav[:, -1] += angle[-1]
        Cav[:, 0:3] = (torch.mm(R, Cav[:, 0:3].T) + t).T

        v = Ego[:, -1]
        index_ego = torch.where(v <0)[0]
        if index_ego.shape[0] != 0:
            Ego[index_ego, -1] += math.pi
        v = Cav[:, -1]
        index_cav = torch.where(v <0)[0]
        if index_cav.shape[0] != 0:
            Cav[index_cav, -1] += math.pi
            
        Ego[:, -1] = torch.fmod(Ego[:, -1], math.pi)
        Cav[:, -1] = torch.fmod(Cav[:, -1], math.pi)
        v = Ego[:, -1]
        index_ego = torch.where((v > math.pi / 2) & (v < math.pi * 3 / 2))[0]
        if index_ego.shape[0] != 0:
            Ego[index_ego, -1] -= math.pi
        v = Cav[:, -1]
        index_cav = torch.where((v > math.pi / 2) & (v < math.pi * 3 / 2))[0]
        if index_cav.shape[0] != 0:
            Cav[index_cav, -1] -= math.pi

        return Ego, Cav

    def GM(self, M: torch.Tensor, M_G: torch.Tensor, Ego: torch.Tensor, Cav: torch.Tensor) -> torch.Tensor:
        m, n = self.m, self.n
        count = torch.zeros((m, n))

        # select the onlookers
        column_sum = torch.sum(M, dim=2).reshape(-1, m, 1)
        row_sum = torch.sum(M, dim=1).reshape(-1, 1, n)
        sum_ = column_sum + row_sum
        sum_mask = (sum_ == 2)
        sum_mask = (sum_mask * M == 1).type(torch.float32)
        G_set = torch.where(sum_mask != 0)

        Gij = torch.zeros_like(M)
        cache = 10000
        for G_set_ in G_set[0]:
            if G_set_ != cache:
                ind = torch.where(G_set[0] == G_set_)[0]
                Gij[G_set_] = torch.sum(M_G[G_set[1][ind] * n + G_set[2][ind], :, :], dim=0)
                cache = G_set_

        Gij_ = (Gij * (M - sum_mask) >= 1)
        Mij_ = ((sum_mask + Gij_) >= 1)

        ind_ = torch.where(torch.sum(Mij_, dim=2) > 1)
        if len(ind_[0]) != 0:
            Mij_[ind_[0], ind_[1], :] = 0

        ind_ = torch.where(torch.sum(Mij_, dim=1) > 1)
        if len(ind_[0]) != 0:
            Mij_[ind_[0], :, ind_[1]] = 0

        Ego_, Cav_ = Ego.repeat(m * n, 1, 1), Cav.repeat(m * n, 1, 1)
        Aij = Mij_

        D_ = torch.where(Aij != 0)
        dis = Ego_[D_[0], D_[1], 0:2] - Cav_[D_[0], D_[2], 0:2]
        D = torch.norm(dis, dim=1)

        DIS = torch.zeros_like(M).to(self.device)
        DIS[D_[0], D_[1], D_[2]] = D.squeeze()

        D__ = torch.mean(DIS, dim=(1, 2)) * m * n / torch.sum(Aij, dim=(1, 2))
        Aij[torch.where(D__ > self.args.absolute_dis_lim)[0], :, :] = 0

        count = torch.sum(Aij, dim=(1, 2))

        # if i_j_ is not unique, who last occured will be selected.
        ij = torch.where(torch.max(count) == count)[0][-1]
        A_ = Aij[ij]

        return A_

    def LM(self, P: torch.Tensor, Q: torch.Tensor) -> Tuple[Any, Any]:
        # P/Q: M/N x 2 xM/N
        m, n = self.m, self.n
        sigma1, sigma2, sigma3 = self.args.sigma1, self.args.sigma2, self.args.sigma3

        M, M_G = torch.zeros((m * n, m, n)).to(self.device), torch.zeros((m * n, m, n)).to(self.device)

        P_ = P.repeat_interleave(n, dim=0)
        Q_ = Q.repeat(m, 1, 1)
        a = torch.matmul(P_.transpose(1, 2), Q_)
        b = torch.norm(P_, dim=1).reshape(m * n, -1, 1) * torch.norm(Q_, dim=1, keepdim=True)
        c = abs(a / b)
        c[c > 1] = 1
        Sz1 = torch.acos(c) / sigma1
        sta = torch.where(Sz1 <= 1)
        dis = P_[sta[0], :, sta[1]] - Q_[sta[0], :, sta[2]]
        Sz2 = torch.norm(dis, dim=1, p=1)
        ind_2 = torch.where(Sz2 <= sigma2)[0]
        ind_3 = torch.where(Sz2 <= sigma3)[0]

        m_ , n_ = torch.floor(sta[0]/n).type(torch.int64), sta[0]%n
        M[sta[0], m_, n_] = 1
        M_G[sta[0], m_, n_] = 1

        if len(ind_2) > 0:
            M[sta[0][ind_2], sta[1][ind_2], sta[2][ind_2]] = 1
        if len(ind_3) > 0:
            M_G[sta[0][ind_3], sta[1][ind_3], sta[2][ind_3]] = 1

        return M, M_G

    def CLC(self, Ego: torch.Tensor) -> torch.Tensor:
        # construct local context
        # Input: Mx7 or Nx7 torch tensor
        X = Ego[:, [0, 1, -1]]  # x, y, heading
        P = {}

        X_ = torch.zeros((X.shape[0], X.shape[0], 2)).to(self.device)
        X_[:] = X[:, :-1]

        X_ -= X[:, 0:2].reshape(X.shape[0], 1, -1)
        theta = X[:, -1]
        c, s = theta.cos(), theta.sin()
        R = torch.zeros((X.shape[0], 2, 2)).to(self.device)
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        Pi = torch.matmul(X_, R)
        P = Pi.transpose(1, 2)

        return P