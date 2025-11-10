# -*- coding: utf-8 -*-
"""
GIN Model - Graph Isomorphism Network (Pure PyTorch Implementation)

Created on Fri Apr  2 15:11:58 2021
@author: Suncy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class GINLayer(nn.Module):
    """Pure PyTorch GIN Layer."""
    
    def __init__(self, in_dim, out_dim, aggregator='max'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.aggregator = aggregator
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: Node features [num_nodes, in_dim]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Optional edge weights [num_edges]
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # Aggregate neighbor features
        messages = x[src]
        
        # Apply edge weights if provided
        if edge_weight is not None:
            messages = messages * edge_weight.unsqueeze(-1)
        
        # Aggregation
        if self.aggregator == 'max':
            # Max aggregation
            neighbor_agg = torch.zeros_like(x).fill_(-1e9)
            neighbor_agg = torch.scatter_reduce(neighbor_agg, 0, 
                                                dst.unsqueeze(-1).expand_as(messages), 
                                                messages, reduce='amax', include_self=False)
            neighbor_agg[neighbor_agg == -1e9] = 0  # Handle nodes with no neighbors
        else:  # sum aggregation
            neighbor_agg = torch.zeros_like(x)
            neighbor_agg = neighbor_agg.scatter_add(0, dst.unsqueeze(-1).expand_as(messages), messages)
        
        # GIN update: (1 + eps) * x + aggregated neighbors
        out = self.mlp((1 + self.eps) * x + neighbor_agg)
        
        return out


class Net(nn.Module):
    """Graph Isomorphism Network - Pure PyTorch."""
    
    def __init__(self, args):
        super(Net, self).__init__()
        h_feat = 128
        self.num_iter = 20
        in_feats = 16
        e_feats = 20
        num_classes = 4
        self.device = torch.device("cuda" if args.num_gpus > 0 else "cpu")
        
        self.conv1 = GINLayer(in_feats, h_feat, 'max')
        self.conv2 = GINLayer(h_feat, h_feat, 'max')
        self.conv3 = GINLayer(h_feat, h_feat, 'max')
        self.conv4 = GINLayer(h_feat, h_feat, 'max')
        self.conv5 = GINLayer(h_feat, h_feat, 'max')
        self.conv6 = GINLayer(h_feat, h_feat, 'max')
        self.conv7 = GINLayer(h_feat, h_feat, 'max')
        self.conv8 = GINLayer(h_feat, h_feat, 'max')
        self.conv9 = GINLayer(h_feat, h_feat, 'max')
        self.conv10 = GINLayer(h_feat, h_feat, 'max')
        self.conv11 = GINLayer(h_feat, h_feat, 'max')
        self.conv12 = GINLayer(h_feat, h_feat, 'max')
        self.conv13 = GINLayer(h_feat, h_feat, 'max')
        self.conv14 = GINLayer(h_feat, h_feat, 'max')
        self.conv15 = GINLayer(h_feat, h_feat, 'max')
        self.conv16 = GINLayer(h_feat, h_feat, 'max')
        self.conv17 = GINLayer(h_feat, h_feat, 'max')
        self.conv18 = GINLayer(h_feat, h_feat, 'max')
        self.conv19 = GINLayer(h_feat, h_feat, 'max')
        self.conv20 = GINLayer(h_feat, h_feat, 'max')

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