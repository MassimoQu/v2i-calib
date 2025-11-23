import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize
from itertools import product

import numpy as np
import sys
from numba import jit, prange
import numba as nb

# M[i][i_prime][i][i_prime]
@nb.njit()
def calculate_node_similarity(N1=np.array([]), N2=np.array([])):
    # Calculate node similarity based on attributes
    lambda_1 = 0.5
    lambda_2 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    f_1_i_ip = (N1[0] == N2[0])
    f_2_i_ip = np.exp(-lambda_1 * np.square(np.linalg.norm(N1[4:6] - N2[4:6])))
    # f_3_i_ip = (in our case, the trajectory affinity is not available since other algorithm doesnt have velocity data, so default set it to 1)
    f_3_i_ip = 1
    f_4_i_ip = np.exp(-lambda_2 * np.linalg.norm(N1[7:9] - N2[7:9]))

    miu_1 = 0.5
    miu_2 = 0.5
    # according to original paper, miu_1 and miu_2 are weights to balance those affinities and as they told that should all be 0.5
    return f_1_i_ip * f_3_i_ip * (miu_1 * f_2_i_ip + miu_2 * f_4_i_ip)


# M[i][i_prime][j][j_prime]
@nb.njit()
def calculate_edge_similarity(e1n1=np.array([]), e1n2=np.array([]), e2n1=np.array([]), e2n2=np.array([])):
    # Calculate edge similarity based on attributes
    lambda_3 = 0.5
    lambda_4 = 0.1
    # according to original paper, they set them pirically 0.5 and 0.1 and we do as so
    g_1_i_ip_j_jp = (((e1n1[0] == e2n1[0]) and (
                e1n2[0] == e2n2[0])) or ((
                                 e1n2[0] == e2n1[0]) and (
                                 e1n1[0] == e2n2[0])))
    g_2_i_ip_j_jp = np.exp(
        -lambda_3 * (np.linalg.norm(e1n1[1:3] - e1n2[1:3]) - np.linalg.norm(
            e2n1[1:3] - e2n2[1:3]))**2)
    g_3_i_ip_j_jp = np.exp(-lambda_4 * np.abs(
        np.sin(e1n1[10] - e1n2[10]) - np.sin(
            e2n1[10] - e2n2[10])))

    miu_3 = 0.5
    miu_4 = 0.5
    # according to original paper, miu_3 and miu_4 also are weights to balance those affinities and as they told that should all be 0.5
    return g_1_i_ip_j_jp * (miu_3 * g_2_i_ip_j_jp + miu_4 * g_3_i_ip_j_jp)


@nb.njit(parallel=True)
def create_affinity_matrix(N1=np.array([[]]), N2=np.array([[]]), L1=np.int32, L2=np.int32):
    M = np.zeros((L1 * L2, L1 * L2))
    len_N1 = L1
    len_N2 = L2

    # for (i, j), (i_prime, j_prime) in product(product(range(len_N1), repeat=2), product(range(len_N2), repeat=2)):
    #         if (i == j and i_prime == j_prime):
    #             M[i * L2 + i_prime, j * L2 + j_prime] = af.calculate_node_similarity(
    #                 N1[i], N2[i_prime])
    #         else:
    #             M[i * L2 + i_prime, j * L2 + j_prime] = af.calculate_edge_similarity(
    #                 N1[i], N1[j], N2[i_prime], N2[j_prime])

    for i in prange(len_N1):
        for j in prange(len_N1):
            for i_prime in prange(len_N2):
                for j_prime in prange(len_N2):
                    M[i * L2 + i_prime, j * L2 + j_prime] = calculate_edge_similarity(
                        N1[i], N1[j], N2[i_prime], N2[j_prime])

    for i in prange(len_N1):
        for i_prime in prange(len_N2):
            M[i * L2 + i_prime, i * L2 + i_prime] = calculate_node_similarity(
                N1[i], N2[i_prime])

    return M



def find_optimal_matching(w, n, m, threshold=0.5):
    G = np.ndarray((n, m))
    for i in range(n):
        for j in range(m):
            G[i][j] = w[(i * m) + j]
    rows, cols = opt.linear_sum_assignment(-np.array(G))
    matching_indices = [[rows[i], cols[i]] for i in range(len(rows)) if G[rows[i]][cols[i]] >= threshold]
    return matching_indices


def main(car1, car2):
    # Step 0: measure L1 and L2 (length)
    L1 = len(car1['category'])
    L2 = len(car2['category'])

    # Step 1: Construct information matrices
    G1 = np.zeros((L1, 11), dtype=np.float32)
    G2 = np.zeros((L2, 11), dtype=np.float32)
    # [0: category, position: 1, 2, 3, bounding_box: 4, 5, 6, world_position: 7, 8, 9, heading: 10]

    # Step 2: Define node and edge attributes
    for i in range(L1):
        G1[i]=[car1['category'][i], car1['position'][i][0], car1['position'][i][1], car1['position'][i][2], 
                    car1['bounding_box'][i][0], car1['bounding_box'][i][1], car1['bounding_box'][i][2],
                    car1['world_position'][i][0], car1['world_position'][i][1], car1['world_position'][i][2],
                    car1['heading'][i][0]]
        
    for i in range(L2):
        G2[i]=[car2['category'][i], car2['position'][i][0], car2['position'][i][1], car2['position'][i][2], 
                    car2['bounding_box'][i][0], car2['bounding_box'][i][1], car2['bounding_box'][i][2],
                    car2['world_position'][i][0], car2['world_position'][i][1], car2['world_position'][i][2],
                    car2['heading'][i][0]]

    # Step 3: Create affinity matrix
    M = create_affinity_matrix(G1, G2, L1, L2)
    
    w = np.zeros(len(M))

    # 定义目标函数
    @nb.njit()
    def objective_function(x=np.array([]), A=np.array([[]])):
        return -(x.T @ A @ x)

    # 定义约束条件：||x||^2 - 1 = 0
    @nb.njit()
    def constraint(x=np.array([])):
        return np.linalg.norm(x)**2 - 1

    constraint_eq = {'type': 'eq', 'fun': constraint}
    w_a_sol = minimize(objective_function, w, args=(M,), method='SLSQP', constraints=constraint_eq)
    # print(w_a_sol)

    w_a = w_a_sol.x
    w_a -= np.min(w_a)
    w_a /= np.max(w_a)

    # print(w_a)
    # print(np.where(w_a > 0.9))

    # Step 4: Solve graph matching problem
    matching_results = find_optimal_matching(w_a, L1, L2, threshold=0.5)
    # print(matching_results)

    return matching_results

