import sys
# sys.path.insert(0, '')
# sys.path.append('/home/zys/code/MS-G3D')
#

import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# from graph.tools import k_adjacency, normalize_adjacency_matrix
# from model.mlp import MLP

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


'''
unit_tcn
输入为[b,c,t,v]，kernle_size=[k,1],只在t维度上做卷积，pad保证stride=1时t大小不变
'''
class unit_tcn(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2) 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

'''
TemporalConv
输入为[b,c,t,v]，kernle_size=[k,1],只在t维度上做卷积，dilation膨胀率，pad保证stride=1时t大小不变
'''
class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

'''
MultiScale_TemporalConv
self.num_branchs: 4（2组带dilation的+maxpooling的+常规的）-> MSTCN
branch_channels:每个branch的c维度
kernel_size：2组dilation[1,2],2组kernel_size[5,5]，dilation=1时kernel_size=5，dilation=2时kernel_size=5
residual：TCN之后是否加入残差链接
residual_kernel_size=1
'''
class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,1), stride=(stride,1), padding=(1,0)),
            nn.BatchNorm2d(branch_channels)  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class mlgcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, num_subset=3):
        super(mlgcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_subset = num_subset
        self.num_point = num_point
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)

        self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.G_part = nn.Parameter(torch.from_numpy(G_part.astype(np.float32)))
        self.G_body = nn.Parameter(torch.from_numpy(G_body.astype(np.float32)))
        self.num_point = num_point

        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        self.conv_part = nn.Conv2d(in_channels, out_channels, 1)
        self.conv_body = nn.Conv2d(in_channels, out_channels, 1)

        d = int(out_channels / 2)
        self.fc = nn.Linear(out_channels, d)
        self.fc_branch = nn.ModuleList([])
        for i in range(3):
            self.fc_branch.append(
                nn.Linear(d, out_channels)
            )
        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    def norm(self, A):
        D_list = torch.sum(A, 0).view(1, self.num_point)
        D_list_12 = (D_list + 0.001) ** (-1)
        D_12 = torch.eye(self.num_point).to(device=A.device) * D_list_12
        A = torch.matmul(A, D_12)
        return A

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.cuda(x.get_device())
        A = A + self.PA  # A+B
        G_part = self.G_part.cuda(x.get_device())
        G_body = self.G_body.cuda(x.get_device())

        y = None
        for i in range(self.num_subset):
            A1 = self.norm(A[i])
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z

        x_part = x.view(N, C * T, V)
        x_part = torch.matmul(x_part, self.norm(G_part)).view(N, C, T, V)
        x_part = self.conv_part(x_part)
        x_body = x.view(N, C * T, V)
        x_body = torch.matmul(x_body, self.norm(G_body)).view(N, C, T, V)
        x_body = self.conv_body(x_body)

        x_joint = y.unsqueeze_(dim=1)
        x_part = x_part.unsqueeze_(dim=1)
        x_body = x_body.unsqueeze_(dim=1)
        x_local = torch.cat([x_joint,x_part,x_body],dim=1)
        x_sum = torch.sum(x_local, dim=1).mean(-1).mean(-1)
        temp_feat = self.fc(x_sum)
        attention_vectors = None
        for i,fc in enumerate(self.fc_branch):
            vector = fc(temp_feat).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = nn.Softmax(1)(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        y = (x_local * attention_vectors).sum(dim=1)

        y = self.bn(y)
        y += self.down(x)
        return self.relu(y)


class SDE(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=17,
                 stride=1, bias=False, width=True):
        assert out_planes % groups == 0
        super(SDE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = nn.Conv1d(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True) #[16,49]
        query_index = torch.arange(kernel_size).unsqueeze(0) #[1,25]
        key_index = torch.arange(kernel_size).unsqueeze(1) #[25,1]
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if (self.in_planes != self.out_planes) and self.stride!=1:
            self.pooling = nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=(self.stride,1))
            self.bn = nn.BatchNorm2d(out_planes)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.reset_parameters()

    def forward(self, x):
        # pdb.set_trace()
        y = x.permute(0, 2, 1, 3)
        N, T, C, V = y.shape
        y = torch.mean(y,dim=1,keepdim=True)
        y = y.contiguous().view(N * 1, C, V) #N,C,V

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(y)) #
        q, k, v = torch.split(qkv.reshape(N * 1, self.groups, self.group_planes * 2, V),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size) #[16,25,25]
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding) #[512,8,25,25]
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3) #[512,8,25,25]

        qk = torch.einsum('bgci, bgcj->bgij', q, k) #[512,8,25,25]

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * 1, 3, self.groups, V, V).sum(dim=1) #[512,8,25,25]
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * 1, self.out_planes * 2, V)
        output = self.bn_output(stacked_output).view(N, 1, self.out_planes, 2, V).sum(dim=-2)
        output = self.sigmoid(output) #[512,1,64,25]
        output = output.permute(0, 2, 1, 3)#[512,64,1,25]

        if (self.in_planes != self.out_planes) and self.stride!=1:
            x = self.bn(self.pooling(x))
        output = x + x * output.expand_as(x)
        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class TME(nn.Module):
    """ Motion exciation module
    
    :param reduction=16
    :param n_segment=8/16
    """
    def __init__(self, channel, reduction=4):
        super(TME, self).__init__()
        self.channel = channel
        self.reduction = reduction
        self.conv1 = nn.Conv2d(
            in_channels=self.channel,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=self.channel//self.reduction)

        self.conv2 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel//self.reduction,
            kernel_size=1,
            groups=channel//self.reduction,
            bias=False)

        self.sigmoid = nn.Sigmoid()

        self.pad = (0, 0, 0, 1)

        self.conv3 = nn.Conv2d(
            in_channels=self.channel//self.reduction,
            out_channels=self.channel,
            kernel_size=1,
            bias=False)
        self.bn3 = nn.BatchNorm2d(num_features=self.channel)

        self.identity = nn.Identity()

    def forward(self, x):
        N, C, T, V = x.size()
        bottleneck = self.conv1(x) # n, c//r, t, v
        bottleneck = self.bn1(bottleneck) # n, c//r, t, v

        # t feature
        t_fea, _ = bottleneck.split([T-1, 1], dim=2) # n, c//r, t-1, v

        # apply transformation conv to t+1 feature
        conv_bottleneck = self.conv2(bottleneck)  # n, c//r, t, v
        # reshape fea: 
        __, tPlusone_fea = conv_bottleneck.split([1, T-1], dim=2)  # n, c//r, t-1, v
        
        # pad the last timestamp
        diff_fea = tPlusone_fea - t_fea # n, c//r, t-1, v
        # pad = (0,0,0,0,0,0,0,1)
        diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, c//r, t, v
        y = torch.mean(diff_fea_pluszero, dim=3, keepdim=True)  #  n, c//r, t, 1
        y = self.conv3(y)  # n, c, t, 1
        y = self.bn3(y)  # n, c, t, 1
        y = self.sigmoid(y)  # n, c, t, 1

        output = x + x * y.expand_as(x)
        return output

'''
basic GCN+TCN block
self.residual: GCN和TCN做完之后的残差链接
'''
class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, G_part, G_body, num_point, kernel_size=5, stride=1, dilations=[1,2], residual=True):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = mlgcn(in_channels, out_channels, A, G_part, G_body, num_point) #fixed+global pattern        
        self.spa_attn = SDE(out_channels,out_channels,stride=stride) # data_driven v2v relations
        self.motion = TME(out_channels) # Motion Exciation
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU()
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        temp = x
        x = self.gcn1(x)
        x = self.spa_attn(x)
        x = self.motion(x)
        x = self.tcn1(x)
        x = x + self.residual(temp)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            Hypergraph = import_class('graph.smarthome.Hypergraph')
            self.graph = Graph(**graph_args)
            self.hypergraph = Hypergraph()

        A = self.graph.A
        G_part = self.hypergraph.G_part
        G_body = self.hypergraph.G_body
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        self.l1 = TCN_GCN_unit(in_channels, 64, A, G_part, G_body, num_point, residual=False)
        self.l2 = TCN_GCN_unit(64, 64, A, G_part, G_body, num_point)
        self.l3 = TCN_GCN_unit(64, 64, A, G_part, G_body, num_point)
        self.l4 = TCN_GCN_unit(64, 64, A, G_part, G_body, num_point)
        self.l5 = TCN_GCN_unit(64, 128, A, G_part, G_body, num_point, stride=2)
        self.l6 = TCN_GCN_unit(128, 128, A, G_part, G_body, num_point)
        self.l7 = TCN_GCN_unit(128, 128, A, G_part, G_body, num_point)
        self.l8 = TCN_GCN_unit(128, 256, A, G_part, G_body, num_point, stride=2)
        self.l9 = TCN_GCN_unit(256, 256, A, G_part, G_body, num_point)
        self.l10 = TCN_GCN_unit(256, 256, A, G_part, G_body, num_point)

        self.fc = nn.Linear(256, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x
            
    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
