"""Main entry point for training and evaluation."""

import os
import logging
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_arguments
from dataset import TreeGraphDataset, collate_graphs
from models import create_model
from trainer import Trainer


def setup_logging(log_level):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _infer_edge_csv(node_csv: str) -> str:
    """If edge CSV not provided, try to infer '<name>_edge.csv' next to node_csv."""
    if node_csv is None:
        return None
    if node_csv.lower().endswith(".csv"):
        candidate = node_csv[:-4] + "_edge.csv"
    else:
        candidate = node_csv + "_edge.csv"
    return candidate if os.path.exists(candidate) else None


def _resolve_phase_paths(args, phase: str):
    """
    Decide which node/edge CSV paths to use for a given phase.
    Priority:
      1) Explicit --{phase}_csv / --{phase}_edge_csv
      2) Infer edge path from node CSV (replace '.csv' with '_edge.csv')
      3) Fallback to ds_dir/ds_name/ds_split/{phase}.csv and {phase}_edge.csv
    """
    # Read explicit flags if present
    node_csv = getattr(args, f"{phase}_csv", None)
    edge_csv = getattr(args, f"{phase}_edge_csv", None)

    # Infer edge CSV if node CSV is given but edge CSV is not
    if node_csv and not edge_csv:
        edge_csv = _infer_edge_csv(node_csv)

    # Fallback to default layout
    if not node_csv:
        node_csv = os.path.join(args.ds_dir, args.ds_name, args.ds_split, f"{phase}.csv")
    if not edge_csv:
        edge_csv = os.path.join(args.ds_dir, args.ds_name, args.ds_split, f"{phase}_edge.csv")

    return node_csv, edge_csv


def _require_exists(path: str, what: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def main():
    """Main function."""
    # Parse arguments
    args = get_arguments()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Using seed: {args.seed}")

    # Create save directory
    save_dir = f"experiments/{args.model}_{args.model_num}"
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Save directory: {save_dir}")

    # Check device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------
    # Resolve CSV paths
    # ------------------------
    if args.mode == "train":
        tr_nodes, tr_edges = _resolve_phase_paths(args, "train")
        va_nodes, va_edges = _resolve_phase_paths(args, "val")
        te_nodes, te_edges = _resolve_phase_paths(args, "test")

        # Validate existence early (clear error messages)
        _require_exists(tr_nodes, "Train nodes CSV")
        _require_exists(tr_edges, "Train edges CSV")
        _require_exists(va_nodes, "Val nodes CSV")
        _require_exists(va_edges, "Val edges CSV")
        _require_exists(te_nodes, "Test nodes CSV")
        _require_exists(te_edges, "Test edges CSV")

        logger.info("Loading datasets with explicit/derived CSV paths (train mode).")
        logger.debug(f"TRAIN nodes: {tr_nodes}")
        logger.debug(f"TRAIN edges: {tr_edges}")
        logger.debug(f"VAL   nodes: {va_nodes}")
        logger.debug(f"VAL   edges: {va_edges}")
        logger.debug(f"TEST  nodes: {te_nodes}")
        logger.debug(f"TEST  edges: {te_edges}")

        train_dataset = TreeGraphDataset(args, phase='train', node_csv=tr_nodes, edge_csv=tr_edges)
        val_dataset   = TreeGraphDataset(args, phase='valid', node_csv=va_nodes, edge_csv=va_edges)
        test_dataset  = TreeGraphDataset(args, phase='test',  node_csv=te_nodes, edge_csv=te_edges)

    else:  # eval mode: only test paths needed
        te_nodes, te_edges = _resolve_phase_paths(args, "test")
        _require_exists(te_nodes, "Test nodes CSV")
        _require_exists(te_edges, "Test edges CSV")

        logger.info("Loading dataset with explicit/derived CSV paths (eval mode).")
        logger.debug(f"TEST nodes: {te_nodes}")
        logger.debug(f"TEST edges: {te_edges}")

        # You can pass any 'phase' string here; loader logic uses provided CSVs
        train_dataset = None
        val_dataset = None
        test_dataset = TreeGraphDataset(args, phase='test', node_csv=te_nodes, edge_csv=te_edges)

    # Report sizes (where available)
    if train_dataset is not None:
        logger.info(f"Train graphs: {len(train_dataset)}")
    if val_dataset is not None:
        logger.info(f"Val graphs:   {len(val_dataset)}")
    logger.info(f"Test graphs:  {len(test_dataset)}")

    # Create data loaders with custom collate function
    pin_mem = torch.cuda.is_available()
    train_loader = None
    val_loader = None
    if train_dataset is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_graphs
        )
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_mem,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate_graphs
        )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_graphs
    )

    # Create model
    logger.info(f"Creating {args.model.upper()} model...")
    model = create_model(args)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create trainer
    trainer = Trainer(model, train_loader, val_loader, test_loader, args, save_dir, args.checkpoint)

    # Train or evaluate
    if args.mode == 'train':
        logger.info("Starting training...")
        trainer.train()
        logger.info("Evaluating on test set...")
        _ = trainer.test()
    elif args.mode == 'eval':
        logger.info("Evaluating on test set...")
        _ = trainer.test()

    logger.info("Done!")


if __name__ == "__main__":
    main()
