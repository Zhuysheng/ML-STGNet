import sys

sys.path.extend(['../'])
from graph import tools

import numpy as np
num_node = 17
self_link = [(i, i) for i in range(num_node)]
#inward_ori_index = [ (14,1), (15, 14), (9, 15), (8, 15), (10, 8), (12, 10), (11, 9), (13, 11), (3, 14), (5, 3), (7, 5), (2, 14),
#        (4, 2), (6, 4)]
inward_ori_index = [(5, 4), (4, 3), (3, 2), (2, 1), (6, 3), (8, 6), (10, 8), (7, 3), (9, 7), (11, 9), (12, 5), (14, 12), (16, 14),
        (13, 5), (15, 13), (17, 15)]

inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Hypergraph:
    def __init__(self):
        self.G_part = self.generate_G_part()
        self.G_body = self.generate_G_body()

    def generate_G_part(self, variable_weight=False):
        H = np.zeros((17, 10))
        H[2][0], H[3][0] = 1, 1                                         #torso = [2,3]
        H[0][1], H[1][1] = 1, 1                                 #head = [0,1]
        H[8][2], H[10][2] = 1, 1                                        #left_arm_down = [8,10]
        H[6][3], H[2][3] = 1, 1                                         #left_arm_up = [6,2]
        H[2][4], H[5][4] = 1, 1                                         #right_arm_up = [2,5]
        H[7][5], H[9][5] = 1, 1                                        #right_arm_down = [7,9]
        H[13][6], H[15][6] = 1, 1                                       #left_leg_down = [13,15]
        H[11][7], H[4][7] = 1, 1                                        #left_leg_up = [11,4]
        H[4][8], H[12][8] = 1, 1                                       #right_leg_up = [4,12]
        H[14][9], H[16][9] = 1, 1                                       #right_leg_down = [14,16]

        H = np.array(H) #【25，10】
        n_edge = H.shape[1] #10
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1) #每个节点连接几条边 [25,]
        # the degree of the hyperedge
        DE = np.sum(H, axis=0) #每条边连了几个节点 [10,]

        invDE = np.mat(np.diag(np.power(DE, -1)))   #[10,10]
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))   #[25,25]
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2  #[25,25],[25,10],[10,10],[10,10],[10,25],[25,25]
            return G

    def generate_G_body(self, variable_weight=False):
        H = np.zeros((17, 5))
        H[0][0], H[1][0], H[2][0], H[3][0], H[4][0] = 1, 1, 1, 1, 1                             #torso = [0, 1, 2, 3, 4 ]
        H[2][1], H[6][1], H[8][1], H[10][1] =  1, 1, 1,1                                                               #left_arm = [2,6,8,10]
        H[2][2], H[5][2], H[7][2], H[9][2] = 1, 1, 1,1                                                                 #right_arm = [2, 5, 7, 9]
        H[4][3], H[11][3], H[13][3], H[15][3] =  1, 1, 1 ,1                                                    #left_leg = [4, 11,13,15]
        H[4][4], H[12][4], H[14][4], H[16][4] =  1, 1, 1  ,1                                                   #right_leg = [4, 12,14,16]

        H = np.array(H)
        n_edge = H.shape[1]
        # the weight of the hyperedge
        W = np.ones(n_edge)
        # the degree of the node
        DV = np.sum(H * W, axis=1)
        # the degree of the hyperedge
        DE = np.sum(H, axis=0)

        invDE = np.mat(np.diag(np.power(DE, -1)))
        DV2 = np.mat(np.diag(np.power(DV, -0.5)))
        W = np.mat(np.diag(W))
        H = np.mat(H)
        HT = H.T

        if variable_weight:
            DV2_H = DV2 * H
            invDE_HT_DV2 = invDE * HT * DV2
            return DV2_H, W, invDE_HT_DV2
        else:
            G = DV2 * H * W * invDE * HT * DV2
            return G


class Graph:
    def __init__(self):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A = tools.get_spatial_graph(self.num_nodes, self_link, inward, outward)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)