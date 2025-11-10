# -*- coding: utf-8 -*-
"""
PDGLSTM Model - Pairwise Deep Graph LSTM (Pure PyTorch Implementation)

Created on Fri Apr  2 15:11:58 2021
@author: Suncy
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Net(nn.Module):
    """PDGLSTM - Pairwise Deep Graph LSTM with Message Passing."""
    
    def __init__(self, args):
        super(Net, self).__init__()
        self.h_feat = 128
        self.num_iter = 20
        n_feats = 16
        e_feats = 20
        num_classes = 4
        self.device = torch.device("cuda" if args.num_gpus > 0 else "cpu")
        
        # node LSTM & edge LSTM
        # 16 dim node feature & 20 dim edge feature
        self.Node_LSTM = nn.LSTMCell(n_feats, self.h_feat)
        self.Edge_LSTM = nn.LSTMCell(e_feats, self.h_feat)
        
        self.m = nn.LeakyReLU()
        
        # message passing network
        self.node_mpn = nn.Linear(2*self.h_feat, n_feats)
        self.edge_mpn = nn.Linear(3*self.h_feat, e_feats)
        
        # linear classifier
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(self.num_iter):
            self.linears_prediction.append(
                    nn.Linear(self.h_feat, num_classes))

        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2, e_feats)
    
    def forward(self, data):
        """
        Args:
            data: Graph data object with attributes:
                - x: Node features [num_nodes, n_feats]
                - edge_index: Edge connectivity [2, num_edges]
                - edge_attr: Edge features [num_edges, 2]
        """
        info = dict()
        
        # Extract features
        node_feat = data.x
        edge_index = data.edge_index
        edge_feat = self.fc1(data.edge_attr)
        
        num_nodes = node_feat.shape[0]
        num_edges = edge_index.shape[1]
        
        hidden_rep = []
        
        # initialization of hidden state and cell state
        h_node = torch.zeros(num_nodes, self.h_feat).to(self.device)
        c_node = torch.zeros(num_nodes, self.h_feat).to(self.device)
        h_edge = torch.zeros(num_edges, self.h_feat).to(self.device)
        c_edge = torch.zeros(num_edges, self.h_feat).to(self.device)
        
        node_input = node_feat
        edge_input = edge_feat
        
        for i in range(self.num_iter):
            # Update node and edge LSTM states
            h_node, c_node = self.Node_LSTM(node_input, (h_node, c_node))
            h_edge, c_edge = self.Edge_LSTM(edge_input, (h_edge, c_edge))
            
            hidden_rep.append(h_node)
            
            # Message passing
            src_idx, dst_idx = edge_index
            
            # Edge message passing: concatenate source, edge, and destination features
            edge_cat = torch.cat([h_node[src_idx], h_edge, h_node[dst_idx]], dim=1)
            edge_msg = self.m(self.edge_mpn(edge_cat))
            
            # Node message passing: aggregate edge messages
            # Mean aggregation of incoming edge features
            h_edge_agg = torch.zeros(num_nodes, self.h_feat, device=self.device)
            h_edge_agg = h_edge_agg.index_add(0, dst_idx, h_edge)
            
            # Count incoming edges for each node
            counts = torch.bincount(dst_idx, minlength=num_nodes).float().clamp(min=1)
            h_edge_agg = h_edge_agg / counts.unsqueeze(1)
            
            # Concatenate aggregated edge features with node features
            node_cat = torch.cat([h_edge_agg, h_node], dim=1)
            node_msg = self.m(self.node_mpn(node_cat))
            
            # Update inputs for next iteration
            node_input = node_msg
            edge_input = edge_msg
        
        # Linear classifier - accumulate predictions from all iterations
        output = 0
        for i, h in enumerate(hidden_rep):
            output += self.drop(self.linears_prediction[i](h))
        
        return output, info
    
    def ce_loss(self, y_pred, y_true, weight=None):
        """Cross-entropy loss."""
        ce = F.cross_entropy(y_pred, y_true, weight=weight, size_average=None, reduce=None, reduction='mean')
        return {"loss": ce}