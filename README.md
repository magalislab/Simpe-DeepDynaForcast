# Simple Implementation of DeepDynaForecast (Pure PyTorch)

A clean, **pure PyTorch** re-implementation of DeepDynaForecast — no DGL, no PyG — with a flexible data loader, robust training utilities, and a **test-time predictor that needs only the edge CSV**.

---

## Key Features

- ✅ **Pure PyTorch Implementation** — no external graph libraries  
- ✅ **Single-/Multi-GPU Friendly** — simple device handling  
- ✅ **Clean, Modular Codebase** — easy to read and extend  
- ✅ **Multiple Architectures** — `GCN`, `GAT`, `GIN`, `PDGLSTM` (LSTM-based message passing)  
- ✅ **From-Scratch Graph Ops** — collate, message passing, metrics/plots  
- ✅ **Edges-Only Inference** — run predictions with **only** `test_edge.csv`  

---

## Installation

```bash
# Python 3.8+ recommended
pip install -r requirements.txt
# (Optional) For Excel outputs in prediction:
pip install openpyxl
```

---

## Project Structure

```
.
├── config.py              # CLI args
├── dataset.py             # Pure-PyTorch dataset + collator
├── models.py              # create_model() dispatcher
├── gcn.py                 # GCN (pure PyTorch)
├── gat.py                 # GAT (pure PyTorch)
├── gin.py                 # GIN (pure PyTorch)
├── pdglstm.py             # PDGLSTM (pure PyTorch)
├── trainer.py             # Training/eval, metrics, plots, checkpoints
├── main.py                # Train/Eval entry point
├── predict_edges_only.py  # Inference from EDGES ONLY
├── requirements.txt
└── README.md
```

---

## Dataset Setup

You can obtain the original data from the
[DeepDynaForecast repository](https://github.com/lab-smile/DeepDynaForecast/tree/main).

The loader supports two usage modes:

1) **Standard layout (no explicit flags):**  
   The code will look for CSVs under:
   ```
   {ds_dir}/{ds_name}/{ds_split}/
       train.csv, train_edge.csv
       val.csv,   val_edge.csv
       test.csv,  test_edge.csv
   ```
   where `ds_split` is typically `ddf_resp+TB_20230222` or similar.

2) **Explicit paths (recommended for custom layouts):**  
   You can override any phase with:
   ```
   --train_csv /path/to/train.csv           --train_edge_csv /path/to/train_edge.csv
   --val_csv   /path/to/val.csv             --val_edge_csv   /path/to/val_edge.csv
   --test_csv  /path/to/test.csv            --test_edge_csv  /path/to/test_edge.csv
   ```
   If you provide only `--{phase}_csv`, the code will **infer** the edge file as `..._edge.csv`.

> **Columns expected by `dataset.py` (train/val/test):**  
> - **Nodes CSV:** must include `sim`, `node`, `cluster_id`, `state`, and `dynamic_cat` (labels on **leaf** nodes).  
> - **Edges CSV:** must include `sim`, endpoints (`new_from`, `new_to`), and two features
>   `weight1_arsinh-norm`, `weight2_arsinh-norm` (or compatible names; see inference below).

---

## Usage

### Training

```bash
python main.py --mode train   --ds_dir /data/cleaned   --ds_name ddf_resp+TB_20230222   --ds_split splitA   --model pdglstm   --batch_size 4 --lr 1e-3 --max_epochs 100   --optimizer Adam --weight_decay 4e-4   --num_workers 4 --seed 123
```

**Optional data flags**
- `--train_csv/--train_edge_csv`, `--val_csv/--val_edge_csv`, `--test_csv/--test_edge_csv`
- Graph tweaks (if exposed in `config.py`): `--add_self_loop`, `--bidirection`

**Outputs** (under `experiments/{model}_{model_num}/`):
- Checkpoints: `best_model.pth`, `checkpoint_{epoch}.pth`
- Plots/CSVs per phase: `ROC_per_class.png`, `ROC_merged.png`, `confusion_mat.png/.eps`,
  `metrics.json`, macro ROC CSV, CM pairs & matrix CSVs

### Evaluation (Full pipeline)

```bash
python main.py --mode eval   --ds_dir /data/cleaned   --ds_name ddf_resp+TB_20230222   --ds_split splitA   --model pdglstm   --checkpoint experiments/PDGLSTM_0/best_model.pth
```

### **Edges-Only Inference (No node CSV required)**

Use the dedicated script to predict classes directly from *just* the edge CSV:

```bash
python predict_edges_only.py   --edge_csv /data/cleaned/ddf_resp+TB_20230222/splitA/test_edge.csv   --model_py pdglstm.py   --checkpoint experiments/PDGLSTM_0/best_model.pth   --output predictions.xlsx   --leaf_only 1
```

**What it does**
- Auto-detects endpoint columns among: `new_from/new_to`, `src/dst`, `from/to`, etc.  
- Auto-detects two numeric edge features, preferring `weight1_arsinh-norm` & `weight2_arsinh-norm`.  
- Groups by `sim` (if absent, treats the file as one graph).  
- Synthesizes node features (16-D constant `0.5`), matching training.  
- Produces:
  - **Nodes sheet/CSV**: node-level predictions (`static/decay/growth/background` + probabilities)
  - **Edges sheet/CSV**: original rows + destination node prediction columns (`*_dst`)

**`--leaf_only` flag**
- `--leaf_only 1` (default): Output **only leaf nodes** (dst not in src). Matches labeling scheme.
- `--leaf_only 0`: Output predictions for **all nodes** (useful for debugging/analysis).

> If `--output` ends with `.xlsx`, results are written to a workbook (`nodes`, `edges`).  
> Otherwise two CSVs are written: `<stem>_nodes.csv`, `<stem>_edges.csv`.

---

## Configuration Options (selected)

### General
- `--mode {train,eval}`
- `--device {cuda,cpu}` (auto-handled in code)
- `--seed` (default: 123)
- Logging level configurable via `--log_level`

### Data
- `--ds_dir`, `--ds_name`, `--ds_split`
- Optional per-phase overrides: `--train_csv/--train_edge_csv`, `--val_csv/--val_edge_csv`, `--test_csv/--test_edge_csv`
- Loader knobs (if enabled): `--add_self_loop`, `--bidirection`

### Model
- `--model {gcn,gat,gin,pdglstm}`
- Typical knobs: `--hidden_dim`, `--num_layers`, `--dropout`

### Training
- `--optimizer {Adam,SGD,RMSprop}`
- `--lr`, `--weight_decay`
- LR schedulers via `--lr_decay_mode {step,plateau}`, `--lr_decay_step`, `--lr_decay_rate`
- `--max_epochs`, `--early_stopping`, `--grad_clip`

### Loss / Labels
- Classes: `{0: static, 1: decay, 2: growth, 3: background}`
- `--loss_ignore_bg` to zero-weight background during CE
- `--background_class` (default: 3)

---

## Metrics & Plots

`trainer.py` computes and saves:
- Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision/Recall (macro)
- Brier score, Cross-entropy
- ROC-AUC (`ovo` and `ovr`, macro and weighted)
- Confusion matrices (PNG/EPS + CSVs)
- Per-class and merged ROC curves + CSV export of macro ROC

---

## Tips & Troubleshooting

- **Missing columns?**  
  Use explicit per-phase flags to point to the correct CSVs. For edges-only inference, ensure you have two numeric features per edge; names are auto-detected.
- **Checkpoint loading issues?**  
  Loader handles `state_dict` vs `model_state_dict` and strips `module.` if saved via `DataParallel`.
- **Leaf logic**  
  During training, labels typically exist **only for leaves** (`dynamic_cat`); `leaf_only=1` keeps predictions aligned with evaluation.

---

## Acknowledgments

This project uses datasets and ideas from  
**DeepDynaForecast** — <https://github.com/lab-smile/DeepDynaForecast>.  
Please cite their work if you use their datasets or concepts.

---
