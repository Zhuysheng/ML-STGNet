import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# #part_10
# num_node_1 = 10
# self_link_1 = [(i, i) for i in range(num_node_1)]
# inward_ori_index_1 = [(1,2),(3,4),(1,5),(3,5),(5,6),(5,7),(7,8),(5,9),(9,10)]
# inward_1 = [(i - 1, j - 1) for (i, j) in inward_ori_index_1]
# outward_1 = [(j, i) for (i, j) in inward_1]
# neighbor_1 = inward_1 + outward_1
#
# #part_5
# num_node_2 = 5
# self_link_2 = [(i, i) for i in range(num_node_2)]
# inward_ori_index_2 = [(1,3),(2,3),(3,4),(3,5)]
# inward_2 = [(i - 1, j - 1) for (i, j) in inward_ori_index_2]
# outward_2 = [(j, i) for (i, j) in inward_2]
# neighbor_2 = inward_2 + outward_2


class Hypergraph:
    def __init__(self):
        self.G_part = self.generate_G_part()
        self.G_body = self.generate_G_body()

    def generate_G_part(self, variable_weight=False):
        H = np.zeros((25, 10))
        H[0][0], H[1][0], H[20][0] = 1, 1, 1                            #torso = [0, 1, 20]
        H[2][1], H[3][1] = 1, 1                                         #head = [2, 3]
        H[10][2], H[11][2], H[23][2], H[24][2] = 1, 1, 1, 1             #left_arm_down = [10, 11, 23, 24]
        H[8][3], H[9][3] = 1, 1                                         #left_arm_up = [8, 9]
        H[4][4], H[5][4] = 1, 1                                         #right_arm_up = [4, 5]
        H[6][5], H[7][5], H[21][5], H[22][5] = 1, 1, 1, 1               #right_arm_down = [6, 7, 21, 22]
        H[18][6], H[19][6] = 1, 1                                       #left_leg_down = [18, 19]
        H[16][7], H[17][7] = 1, 1                                       #left_leg_up = [16, 17]
        H[12][8], H[13][8] = 1, 1                                       #right_leg_up = [12, 13]
        H[14][9], H[15][9] = 1, 1                                       #right_leg_down = [14, 15]

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
        H = np.zeros((25, 5))
        H[0][0], H[1][0], H[2][0], H[3][0], H[20][0] = 1, 1, 1, 1, 1                                            #torso = [0, 1, 2, 3, 20]
        H[8][1], H[9][1], H[10][1], H[11][1], H[23][1], H[24][1] = 1, 1, 1, 1, 1, 1                             #left_arm = [8, 9, 10, 11, 23, 24]
        H[4][2], H[5][2], H[6][2], H[7][2], H[21][2], H[22][2] = 1, 1, 1, 1, 1, 1                               #right_arm = [4, 5, 6, 7, 21, 22]
        H[16][3], H[17][3], H[18][3], H[19][3] = 1, 1, 1, 1                                                     #left_leg = [16, 17, 18, 19]
        H[12][4], H[13][4], H[14][4], H[15][4] = 1, 1, 1, 1                                                     #right_leg = [12, 13, 14, 15]

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

class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.inward = inward
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        #self.A = tools.normalize_adjacency_matrix(self.A_binary)
        self.A = tools.get_spatial_graph(self.num_nodes, self_link, inward, outward)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
