# -*- coding: utf-8 -*-
"""
GCN Model - Graph Convolutional Network (Pure PyTorch Implementation)

Created on Fri Apr  2 15:11:58 2021
@author: Suncy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """Pure PyTorch Graph Convolution Layer."""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        """
        # Linear transformation
        x = self.linear(x)
        
        # Aggregate messages from neighbors
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Compute degree for normalization
        degree = torch.zeros(num_nodes, device=x.device)
        degree = degree.scatter_add(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = degree.clamp(min=1)
        
        # Message passing with optional edge weights
        if edge_weight is not None:
            messages = x[src] * edge_weight.unsqueeze(-1)
        else:
            messages = x[src]
        
        # Aggregate messages
        out = torch.zeros_like(x)
        out = out.scatter_add(0, dst.unsqueeze(-1).expand_as(messages), messages)
        
        # Normalize by degree
        out = out / degree.unsqueeze(-1)
        
        return out


class Net(nn.Module):
    """Graph Convolutional Network - Pure PyTorch."""
    
    def __init__(self, args):
        super(Net, self).__init__()
        h_feat = 128
        n_iter = 20
        in_feats = 16
        num_classes = 4
        self.device = torch.device("cuda" if args.num_gpus > 0 else "cpu")
        
        self.conv1 = GraphConvLayer(in_feats, h_feat)
        self.conv2 = GraphConvLayer(h_feat, h_feat)
        self.conv3 = GraphConvLayer(h_feat, h_feat)
        self.conv4 = GraphConvLayer(h_feat, h_feat)
        self.conv5 = GraphConvLayer(h_feat, h_feat)
        self.conv6 = GraphConvLayer(h_feat, h_feat)
        self.conv7 = GraphConvLayer(h_feat, h_feat)
        self.conv8 = GraphConvLayer(h_feat, h_feat)
        self.conv9 = GraphConvLayer(h_feat, h_feat)
        self.conv10 = GraphConvLayer(h_feat, h_feat)
        self.conv11 = GraphConvLayer(h_feat, h_feat)
        self.conv12 = GraphConvLayer(h_feat, h_feat)
        self.conv13 = GraphConvLayer(h_feat, h_feat)
        self.conv14 = GraphConvLayer(h_feat, h_feat)
        self.conv15 = GraphConvLayer(h_feat, h_feat)
        self.conv16 = GraphConvLayer(h_feat, h_feat)
        self.conv17 = GraphConvLayer(h_feat, h_feat)
        self.conv18 = GraphConvLayer(h_feat, h_feat)
        self.conv19 = GraphConvLayer(h_feat, h_feat)
        self.conv20 = GraphConvLayer(h_feat, h_feat)
        self.m = nn.LeakyReLU()
        
        self.fc = nn.Linear(h_feat, num_classes)
    
    def forward(self, data):
        """
        Args:
            data: Graph data object with attributes:
                - x: Node features
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features
        """
        info = dict()
        
        # Extract features
        node_feat = data.x
        edge_index = data.edge_index
        edge_feat = data.edge_attr[:, 0] if data.edge_attr is not None else None
        
        h = self.conv1(node_feat, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv2(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv3(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv4(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv5(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv6(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv7(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv8(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv9(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv10(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv11(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv12(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv13(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv14(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv15(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv16(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv17(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv18(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv19(h, edge_index, edge_feat)
        h = self.m(h)
        h = self.conv20(h, edge_index, edge_feat)
        h = self.m(h)
        
        output = self.fc(h)
        
        return output, info
    
    def ce_loss(self, y_pred, y_true, weight=None):
        """Cross-entropy loss."""
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}