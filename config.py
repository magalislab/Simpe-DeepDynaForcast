"""Configuration file for graph neural network training."""
import argparse


def get_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Graph Neural Network Training")
    
    # General options
    general = parser.add_argument_group("General")
    general.add_argument("--seed", type=int, default=123, help="Random seed")
    general.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    general.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    general.add_argument("--restore", action="store_true", help="Restore model weights")
    general.add_argument("--restore_metric", type=str, default="loss", choices=["loss"])
    general.add_argument("--log_level", type=str, default="info", choices=["debug", "info", "warning"])
    general.add_argument("--log_train_freq", type=int, default=1)
    general.add_argument("--log_valid_freq", type=int, default=1)
    general.add_argument("--num_gpus", type=int, default=1)
    
    # Data options
    data = parser.add_argument_group("Data")
    # Explicit CSV (optional – overrides ds_dir/ds_name/ds_split layout)
    data.add_argument("--train_csv", type=str, default=None, help="Full path to train nodes CSV")
    data.add_argument("--train_edge_csv", type=str, default=None, help="Full path to train edges CSV")
    data.add_argument("--val_csv", type=str, default=None, help="Full path to val nodes CSV")
    data.add_argument("--val_edge_csv", type=str, default=None, help="Full path to val edges CSV")
    data.add_argument("--test_csv", type=str, default=None, help="Full path to test nodes CSV")
    data.add_argument("--test_edge_csv", type=str, default=None, help="Full path to test edges CSV")

    data.add_argument("--num_workers", type=int, default=0)
    data.add_argument("--batch_size", type=int, default=4)
    data.add_argument("--node_label_cols", type=str, default="dynamic_cat")
    data.add_argument("--edge_feat_cols", type=str, default="norm_edge_feats_arsinh")
    data.add_argument("--add_self_loop", action="store_true")
    data.add_argument("--bidirection", action="store_true", default=True)
  
    # Model options
    model = parser.add_argument_group("Model")
    model.add_argument("--model", type=str, default="lstm",
                      choices=["gcn", "gat", "gin", "lstm", "pdglstm_bn", "pdglstm"])
    model.add_argument("--model_num", type=int, default=1)
    model.add_argument("--hidden_dim", type=int, default=128)
    model.add_argument("--num_layers", type=int, default=20)
    model.add_argument("--dropout", type=float, default=0.5)
    model.add_argument("--loss_ignore_bg", action="store_true", default=True)
    model.add_argument("--checkpoint", type=str, default=None,
                      help="Path to checkpoint file for evaluation or resuming training")
    
    # Training options
    train = parser.add_argument_group("Training")
    train.add_argument("--max_epochs", type=int, default=100)
    train.add_argument("--optimizer", type=str, default="Adam", choices=["SGD", "Adam", "RMSprop"])
    train.add_argument("--lr", type=float, default=0.001)
    train.add_argument("--min_lr", type=float, default=1e-6)
    train.add_argument("--lr_decay_mode", type=str, default="step",
                      choices=["step", "plateau"])
    train.add_argument("--lr_decay_step", type=int, default=50)
    train.add_argument("--lr_decay_rate", type=float, default=0.1)
    train.add_argument("--grad_clip", type=float, default=15.0)
    train.add_argument("--weight_decay", type=float, default=0.0004)
    train.add_argument("--early_stopping", type=int, default=50)
    
    args = parser.parse_args()
    return args