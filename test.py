#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict_edges_only.py
Test-time predictions using ONLY edge CSV, with a pretrained PDGLSTM model.

- No node CSV required.
- Handles multiple graphs via 'sim' column (optional). If absent, treats all edges as one graph.
- Flexible column detection for edge endpoints and features.
- Writes node-level and edge-level prediction files.

Usage:
    python predict_edges_only.py \
        --edge_csv path/to/test_edge.csv \
        --model_py path/to/pdglstm.py \
        --checkpoint experiments/PDGLSTM_0/best_model.pth \
        --output edge_predictions.csv \
        --device auto \
        --leaf_only 1

Notes:
- If --output ends with ".xlsx", an Excel workbook is written (requires openpyxl).
- Otherwise two CSVs are written next to --output:
    <stem>_nodes.csv and <stem>_edges.csv
"""

import os
import sys
import math
import argparse
import importlib.util
import numpy as np
import pandas as pd
import torch

# We reuse your GraphData container to match model forward()
try:
    from dataset import GraphData  # must exist in project
except Exception:
    # Minimal fallback if importing from dataset fails
    class GraphData:
        def __init__(self, x, edge_index, edge_attr, y, org_feat, num_nodes):
            self.x = x
            self.edge_index = edge_index
            self.edge_attr = edge_attr
            self.y = y
            self.org_feat = org_feat
            self.num_nodes = num_nodes
        def to(self, device):
            self.x = self.x.to(device)
            self.edge_index = self.edge_index.to(device)
            self.edge_attr = self.edge_attr.to(device)
            self.y = self.y.to(device)
            self.org_feat = self.org_feat.to(device)
            return self

CLASS_NAMES = {0: "static", 1: "decay", 2: "growth", 3: "background"}


# -------------------------------
# CSV helpers
# -------------------------------
def _read_edges_any(edge_csv: str) -> pd.DataFrame:
    """Robust CSV reader for edges; tries multiple separators."""
    for sep in [r"\s+", ",", None]:
        try:
            if sep is None:
                df = pd.read_csv(edge_csv, delim_whitespace=True, skipinitialspace=True, engine="python")
            else:
                df = pd.read_csv(edge_csv, sep=sep, engine="python")
            # simple sanity: must have >= 2 columns
            if df.shape[1] >= 2:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse edge CSV: {edge_csv}")


def _detect_endpoint_cols(df: pd.DataFrame):
    """Detect (from,to) column names."""
    candidates = [
        ("new_from", "new_to"),
        ("src", "dst"),
        ("from", "to"),
        ("source", "target"),
        ("u", "v")
    ]
    cols = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in cols}
    for a, b in candidates:
        if a in lower and b in lower:
            return lower[a], lower[b]
    # try fallback by heuristic: pick two integer-like columns
    intish = [c for c in cols if pd.api.types.is_integer_dtype(df[c]) or pd.api.types.is_numeric_dtype(df[c])]
    if len(intish) >= 2:
        return intish[0], intish[1]
    raise KeyError("Could not detect edge endpoint columns (from/to).")


def _detect_feature_cols(df: pd.DataFrame, exclude):
    """
    Detect two edge-feature columns. Prefer normalized weights if present.
    """
    # Preferred names
    preferred_pairs = [
        ("weight1_arsinh-norm", "weight2_arsinh-norm"),
        ("weight1", "weight2"),
    ]
    cols = [c for c in df.columns if c not in exclude]
    lower = {c.lower(): c for c in cols}
    for a, b in preferred_pairs:
        if a in lower and b in lower:
            return [lower[a], lower[b]]
    # Fallback: first two numeric columns not in exclude
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric) < 2:
        raise KeyError("Need at least two numeric columns for edge features.")
    return numeric[:2]


def _ensure_sim(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure there is a 'sim' column for grouping. If absent, create one constant group.
    """
    if "sim" not in df.columns:
        df = df.copy()
        df["sim"] = "graph_0"
    return df


# -------------------------------
# Graph builders
# -------------------------------
def build_graphdata_from_edges(edge_df: pd.DataFrame, from_col: str, to_col: str, feat_cols, sim_id) -> (GraphData, dict):
    """
    Build a GraphData (x, edge_index, edge_attr) from a single-sim edge DF.
    Returns GraphData and a mapping idx->original_node_id.
    """
    # Gather node set
    nodes = sorted(set(edge_df[from_col].tolist()) | set(edge_df[to_col].tolist()))
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for n, i in node_to_idx.items()}

    # Remap edges
    src = edge_df[from_col].map(node_to_idx).astype(np.int64).values
    dst = edge_df[to_col].map(node_to_idx).astype(np.int64).values
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

    # Edge features (2-dim expected by your PDGLSTM -> fc1 to 20)
    e_attr = torch.tensor(edge_df[feat_cols].values.astype(np.float32), dtype=torch.float32)

    # Node features: 16-d constant (as in your dataset)
    num_nodes = len(nodes)
    x = torch.ones(num_nodes, 16, dtype=torch.float32) * 0.5

    # Dummy labels (unused at test time)
    y = torch.zeros(num_nodes, dtype=torch.long)

    # org_feat for bookkeeping: [sim_number, original_node_id] (float tensor)
    sim_numeric = 0.0
    try:
        # try to parse trailing digits in sim id
        sim_numeric = float(str(sim_id).split("_")[-1])
    except Exception:
        pass
    org_feat_np = np.stack([np.full((num_nodes,), sim_numeric, dtype=np.float32),
                            np.array([float(idx_to_node[i]) if str(idx_to_node[i]).isdigit() else float(i)
                                      for i in range(num_nodes)], dtype=np.float32)], axis=1)
    org_feat = torch.tensor(org_feat_np, dtype=torch.float32)

    data = GraphData(
        x=x,
        edge_index=edge_index,
        edge_attr=e_attr,
        y=y,
        org_feat=org_feat,
        num_nodes=num_nodes
    )
    return data, idx_to_node


def find_leaf_indices(edge_df: pd.DataFrame, from_col: str, to_col: str, node_to_idx: dict):
    """Leaf = appears as destination but not as source."""
    leaves = set(edge_df[to_col].tolist()) - set(edge_df[from_col].tolist())
    return [node_to_idx[n] for n in leaves if n in node_to_idx]


# -------------------------------
# Model loading
# -------------------------------
def load_model_from_py(model_py: str, device: torch.device):
    """Dynamically import the .py that defines class Net(args)."""
    spec = importlib.util.spec_from_file_location("edge_model", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class Args:
        num_gpus = 1 if (torch.cuda.is_available() and device.type == "cuda") else 0

    model = mod.Net(Args())
    model.to(device)
    return model


def load_checkpoint_into_model(model, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    # Strip 'module.' if present
    new_state = {}
    for k, v in state.items():
        new_state[k[7:]] = v if k.startswith("module.") else v
    try:
        model.load_state_dict(new_state, strict=True)
    except Exception:
        # Last resort: try loading original state keys
        model.load_state_dict(state, strict=False)
    model.eval()
    epoch = ckpt.get("epoch", None)
    return epoch


# -------------------------------
# Inference
# -------------------------------
def infer_one_graph(model, data: GraphData, device: torch.device):
    data = data.to(device)
    with torch.no_grad():
        out = model(data)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1).detach().cpu().numpy()  # (N, C=4)
        preds = probs.argmax(axis=1)
    return probs, preds


# -------------------------------
# Main runner
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict from EDGES ONLY using pretrained PDGLSTM")
    parser.add_argument("--edge_csv", required=True, type=str, help="Path to test edges CSV")
    parser.add_argument("--model_py", required=True, type=str, help="Path to PDGLSTM .py defining Net(args)")
    parser.add_argument("--checkpoint", required=True, type=str, help="Path to checkpoint (e.g., best_model.pth)")
    parser.add_argument("--output", default="edge_predictions.csv", type=str,
                        help="Output file. If ends with .xlsx, writes Excel; otherwise writes CSV(s).")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    parser.add_argument("--leaf_only", default=1, type=int, help="1: report only leaf nodes; 0: all nodes")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[Info] Using device: {device}")

    # Load edges
    df = _read_edges_any(args.edge_csv)
    # Clean headers
    df.columns = [c.strip() for c in df.columns]
    df = _ensure_sim(df)

    # Detect columns
    from_col, to_col = _detect_endpoint_cols(df)
    feat_cols = _detect_feature_cols(df, exclude={from_col, to_col, "sim"})
    print(f"[Info] Endpoint cols: {from_col!r}, {to_col!r}")
    print(f"[Info] Feature cols: {feat_cols}")
    multi = df["sim"].nunique() > 1
    print(f"[Info] Detected {df['sim'].nunique()} graph(s) via 'sim'")

    # Load model + checkpoint
    model = load_model_from_py(args.model_py, device)
    epoch = load_checkpoint_into_model(model, args.checkpoint, device)
    if epoch is not None:
        print(f"[Info] Loaded checkpoint (epoch={epoch})")

    # Per-graph inference
    node_rows = []
    edge_rows = []

    for sim_id, gdf in df.groupby("sim"):
        gdf = gdf.reset_index(drop=True)
        data, idx_to_node = build_graphdata_from_edges(gdf, from_col, to_col, feat_cols, sim_id)
        probs, preds = infer_one_graph(model, data, device)

        # Map nodes
        node_to_idx = {v: k for k, v in idx_to_node.items()}

        # Leaf filter (on original node ids)
        leaf_idx = find_leaf_indices(gdf, from_col, to_col, node_to_idx)
        report_mask = np.zeros(len(idx_to_node), dtype=bool)
        report_mask[leaf_idx] = True
        if not bool(args.leaf_only):
            report_mask[:] = True

        for i in np.where(report_mask)[0]:
            node_id = idx_to_node[i]
            row = {
                "sim": sim_id,
                "node_id": node_id,
                "pred_class_id": int(preds[i]),
                "pred_class_name": CLASS_NAMES.get(int(preds[i]), f"class_{int(preds[i])}"),
                "prob_static": float(probs[i, 0]) if probs.shape[1] > 0 else 0.0,
                "prob_decay": float(probs[i, 1]) if probs.shape[1] > 1 else 0.0,
                "prob_growth": float(probs[i, 2]) if probs.shape[1] > 2 else 0.0,
                "prob_background": float(probs[i, 3]) if probs.shape[1] > 3 else 0.0,
            }
            node_rows.append(row)

        # Edge-level: attach destination prediction
        # Build a quick lookup from dst node -> prediction
        dst_pred = {idx_to_node[i]: {
            "pred_class_name": CLASS_NAMES.get(int(preds[i]), f"class_{int(preds[i])}"),
            "prob_static": float(probs[i, 0]) if probs.shape[1] > 0 else 0.0,
            "prob_decay": float(probs[i, 1]) if probs.shape[1] > 1 else 0.0,
            "prob_growth": float(probs[i, 2]) if probs.shape[1] > 2 else 0.0,
            "prob_background": float(probs[i, 3]) if probs.shape[1] > 3 else 0.0,
        } for i in range(len(idx_to_node))}
        eg = gdf.copy()
        eg["predicted_class_dst"] = eg[to_col].map(lambda x: dst_pred.get(x, {}).get("pred_class_name", "unknown"))
        eg["prob_static_dst"] = eg[to_col].map(lambda x: dst_pred.get(x, {}).get("prob_static", 0.0))
        eg["prob_decay_dst"] = eg[to_col].map(lambda x: dst_pred.get(x, {}).get("prob_decay", 0.0))
        eg["prob_growth_dst"] = eg[to_col].map(lambda x: dst_pred.get(x, {}).get("prob_growth", 0.0))
        eg["prob_background_dst"] = eg[to_col].map(lambda x: dst_pred.get(x, {}).get("prob_background", 0.0))
        edge_rows.append(eg)

    node_df_out = pd.DataFrame(node_rows)
    edge_df_out = pd.concat(edge_rows, ignore_index=True) if len(edge_rows) else df.copy()

    # Save
    out_path = args.output
    stem, ext = os.path.splitext(out_path)
    if ext.lower() == ".xlsx":
        try:
            with pd.ExcelWriter(out_path, engine="openpyxl") as wr:
                node_df_out.to_excel(wr, index=False, sheet_name="nodes")
                edge_df_out.to_excel(wr, index=False, sheet_name="edges")
            print(f"\n✓ Saved predictions to {out_path}")
        except Exception as e:
            print(f"[Warn] Could not write .xlsx ({e}). Writing CSVs instead.")
            node_df_out.to_csv(stem + "_nodes.csv", index=False)
            edge_df_out.to_csv(stem + "_edges.csv", index=False)
            print(f"✓ Saved {stem+'_nodes.csv'} and {stem+'_edges.csv'}")
    else:
        # Write two CSVs: nodes + edges
        node_csv = stem + "_nodes.csv"
        edge_csv = stem + "_edges.csv"
        node_df_out.to_csv(node_csv, index=False)
        edge_df_out.to_csv(edge_csv, index=False)
        print(f"\n✓ Saved {node_csv} and {edge_csv}")

    # Summary
    print("\n========== SUMMARY ==========")
    print(f"Graphs (sim): {node_df_out['sim'].nunique()}")
    if "pred_class_name" in node_df_out.columns and len(node_df_out):
        print("\nNode prediction counts:")
        print(node_df_out["pred_class_name"].value_counts())


if __name__ == "__main__":
    main()
