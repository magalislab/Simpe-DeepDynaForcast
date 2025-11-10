"""Dataset loader using standard PyTorch."""

import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter


class GraphData:
    """Simple graph data structure."""
    def __init__(self, x, edge_index, edge_attr, y, org_feat, num_nodes):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.org_feat = org_feat
        self.num_nodes = num_nodes
    
    def to(self, device):
        """Move data to device."""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)
        self.org_feat = self.org_feat.to(device)
        return self


class TreeGraphDataset(Dataset):
    """Dataset for tree-structured graphs."""

    def __init__(self, args, phase, node_csv: str = None, edge_csv: str = None):
        """
        Args:
            args: Configuration arguments
            phase: 'train', 'valid', or 'test'
            node_csv: optional full path to nodes CSV
            edge_csv: optional full path to edges CSV
        """
        super().__init__()

        self.args = args
        self.phase = phase

        if (node_csv is None) or (edge_csv is None):
            # fallback to legacy directory structure
            ds_folder = osp.join(args.ds_dir, args.ds_name, args.ds_split)
            if node_csv is None:
                node_csv = f"{ds_folder}/{ 'val' if phase=='valid' else phase }.csv"
            if edge_csv is None:
                edge_csv = f"{ds_folder}/{ 'val' if phase=='valid' else phase }_edge.csv"

        # Load data
        self.node_df = pd.read_csv(node_csv, low_memory=False)
        self.edge_df = pd.read_csv(edge_csv, low_memory=False)
        self.tree_ids = self.node_df["sim"].unique()

        # Feature columns
        self.edge_feat_cols = ['weight1_arsinh-norm', 'weight2_arsinh-norm']
        self.node_label_col = 'dynamic_cat'

        # Mappings
        self.cid_dict = {'Background': 0, 'c1': 1, 'c2': 2, 'c3': 3, 'c4': 4, 'c5': 5, 'c6': 6}
        self.state_dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}

        # Process node features
        self.node_df['c_id'] = self.node_df['cluster_id'].map(self.cid_dict)
        self.node_df['s'] = self.node_df['state'].map(self.state_dict)
        self.node_feat_org = ['sim', 'c_id', 's', 'node']
        
    def __len__(self):
        return len(self.tree_ids)

    def __getitem__(self, idx):
        """Get a single graph."""
        tree_id = self.tree_ids[idx]
        
        # Get tree-specific data
        tree_nodes = self.node_df[self.node_df['sim'] == tree_id]
        tree_edges = self.edge_df[self.edge_df['sim'] == tree_id]
        
        # Build edge index (subtract 1 for 0-indexing)
        src = torch.tensor(tree_edges['new_from'].values - 1, dtype=torch.long)
        dst = torch.tensor(tree_edges['new_to'].values - 1, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
        
        # Sort nodes and get features
        sorted_nodes = tree_nodes.sort_values(by='node')
        num_nodes = len(sorted_nodes)
        
        # Initialize node features and labels
        node_feat = torch.ones(num_nodes, 16) * 0.5
        node_label = torch.full((num_nodes,), 3, dtype=torch.long)
        
        # Find leaf nodes
        leaves = set(tree_edges['new_to'].values) - set(tree_edges['new_from'].values)
        leaves = [i - 1 for i in leaves]
        
        # Assign labels to leaf nodes
        for leaf in leaves:
            node_label[leaf] = sorted_nodes[self.node_label_col].values[leaf]
        
        # Edge features
        edge_attr = torch.tensor(tree_edges[self.edge_feat_cols].values, dtype=torch.float32)
        
        # Original features for analysis
        org_feat = torch.tensor(sorted_nodes[self.node_feat_org].values, dtype=torch.float32)
        
        # Add self-loops and bidirectional edges if needed
        if self.args.add_self_loop:
            loop_index = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, loop_index], dim=1)
            loop_attr = torch.zeros(num_nodes, edge_attr.size(1))
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
        
        if self.args.bidirection:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
        
        # Create graph data object
        data = GraphData(
            x=node_feat,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=node_label,
            org_feat=org_feat,
            num_nodes=num_nodes
        )
        
        return data


def collate_graphs(batch):
    """
    Collate function to batch multiple graphs.
    Creates a single large graph with disconnected components.
    """
    # Accumulate node and edge counts
    node_offset = 0
    batched_x = []
    batched_edge_index = []
    batched_edge_attr = []
    batched_y = []
    batched_org_feat = []
    batch_indices = []
    
    for i, data in enumerate(batch):
        batched_x.append(data.x)
        batched_y.append(data.y)
        batched_org_feat.append(data.org_feat)
        batched_edge_attr.append(data.edge_attr)
        
        # Offset edge indices
        edge_index_offset = data.edge_index + node_offset
        batched_edge_index.append(edge_index_offset)
        
        # Track which graph each node belongs to
        batch_indices.extend([i] * data.num_nodes)
        
        node_offset += data.num_nodes
    
    # Concatenate everything
    batched_data = GraphData(
        x=torch.cat(batched_x, dim=0),
        edge_index=torch.cat(batched_edge_index, dim=1),
        edge_attr=torch.cat(batched_edge_attr, dim=0),
        y=torch.cat(batched_y, dim=0),
        org_feat=torch.cat(batched_org_feat, dim=0),
        num_nodes=node_offset
    )
    batched_data.batch = torch.tensor(batch_indices, dtype=torch.long)
    
    return batched_data


def get_label_weights(args):
    """Calculate class weights for imbalanced data."""
    # Pre-computed weights for the dataset
    label_weights = [0.3406808892621469, 51.519496230514754, 22.079112613356273]
    
    if args.loss_ignore_bg:
        label_weights.append(0.0)
    
    return label_weights