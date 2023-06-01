import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from graph import tools

# Joint index:
# {0,  "Nose"}
# {1,  "Neck"},
# {2,  "RShoulder"},
# {3,  "RElbow"},
# {4,  "RWrist"},
# {5,  "LShoulder"},
# {6,  "LElbow"},
# {7,  "LWrist"},
# {8,  "RHip"},
# {9,  "RKnee"},
# {10, "RAnkle"},
# {11, "LHip"},
# {12, "LKnee"},
# {13, "LAnkle"},
# {14, "REye"},
# {15, "LEye"},
# {16, "REar"},
# {17, "LEar"},

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11), (10, 9), (9, 8),
          (11, 5), (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0), (17, 15),
          (16, 14)]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Hypergraph:
    def __init__(self):
        self.G_part = self.generate_G_part()
        self.G_body = self.generate_G_body()

    def generate_G_part(self, variable_weight=False):
        H = np.zeros((18, 10))
        H[0][0], H[1][0] = 1, 1                                         #torso = [0, 1]
        H[0][1], H[14][1], H[15][1], H[16][1], H[17][1] = 1, 1, 1, 1, 1 #head = [0,14,15,16,17]
        H[3][2], H[4][2] = 1, 1                                        #left_arm_down = [3,4]
        H[2][3], H[3][3] = 1, 1                                         #left_arm_up = [2,3]
        H[5][4], H[6][4] = 1, 1                                         #right_arm_up = [5,6]
        H[6][5], H[7][5] = 1, 1                                        #right_arm_down = [6,7]
        H[9][6], H[10][6] = 1, 1                                       #left_leg_down = [9,10]
        H[8][7], H[9][7] = 1, 1                                        #left_leg_up = [8,9]
        H[11][8], H[12][8] = 1, 1                                       #right_leg_up = [11,12]
        H[12][9], H[13][9] = 1, 1                                       #right_leg_down = [12,13]

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
        H = np.zeros((18, 5))
        H[0][0], H[1][0], H[14][0], H[15][0], H[16][0], H[17][0] = 1, 1, 1, 1, 1, 1                             #torso = [0, 1,14,15,16,17]
        H[2][1], H[3][1], H[4][1] =  1, 1, 1                                                                        #left_arm = [2,3,4]
        H[5][2], H[6][2], H[7][2] = 1, 1, 1                                                                 #right_arm = [5,6,7]
        H[8][3], H[9][3], H[10][3] =  1, 1, 1                                                     #left_leg = [8,9,10]
        H[11][4], H[12][4], H[13][4] =  1, 1, 1                                                     #right_leg = [11,12,13]

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
        self.num_nodes = num_node
        self.edges = neighbor
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = tools.get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = tools.get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = tools.get_spatial_graph(self.num_nodes, self_link, inward, outward)

if __name__ == '__main__':
    graph = AdjMatrixGraph()
    A_binary = graph.A_binary
    import matplotlib.pyplot as plt
    print(A_binary)
    plt.matshow(A_binary)
    plt.show()
