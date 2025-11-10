"""Graph Neural Network models - Router to import from model-specific files."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_model(args):
    """
    Factory function to create models by importing from model-specific files.
    
    Each model is stored in its own file:
    - gcn.py -> GCN model
    - gin.py -> GIN model  
    - pdglstm_bn.py -> PDGLSTM model with batch norm
    - gat.py -> GAT model (if exists)
    """
    
    model_name = args.model.lower()
    
    try:
        if model_name == 'gcn':
            from gcn import Net
            return Net(args)
        
        elif model_name == 'gin':
            from gin import Net
            return Net(args)
        
        elif model_name in ['pdglstm_bn', 'pdglstm', 'lstm']:
            from pdglstm_bn import Net
            return Net(args)
        
        elif model_name == 'gat':
            try:
                from gat import Net
                return Net(args)
            except ImportError:
                raise ValueError(f"GAT model file (gat.py) not found")
        
        else:
            raise ValueError(
                f"Unknown model: {args.model}. "
                f"Available models: gcn, gin, pdglstm_bn, gat"
            )
    
    except ImportError as e:
        raise ImportError(
            f"Could not import model '{model_name}'. "
            f"Make sure {model_name}.py exists in the same directory. "
            f"Error: {str(e)}"
        )


# For backward compatibility - if someone imports models directly
try:
    from gcn import Net as GCNNet
except ImportError:
    GCNNet = None

try:
    from gin import Net as GINNet
except ImportError:
    GINNet = None

try:
    from pdglstm_bn import Net as PDGLSTMNet
except ImportError:
    PDGLSTMNet = None