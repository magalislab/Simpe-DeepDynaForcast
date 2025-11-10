# Simple Implementation of DeepDynaForecast

A clean implementation of DeepDynaForecast using **pure PyTorch** (no external graph libraries!).

## Key Features

✅ **Pure PyTorch Implementation** - No DGL, no PyTorch Geometric, just PyTorch!  
✅ **Single GPU Support** - Simple and efficient  
✅ **Clean Code Structure** - Easy to read and modify  
✅ **Multiple GNN Architectures** - GCN, GAT, GIN, LSTM-based models  
✅ **Custom Graph Operations** - All graph convolutions implemented from scratch  
✅ **Direct CSV Input** - Load train/val/test data directly from CSV files  
✅ **Edge-Only Prediction** - Make predictions using only edge CSV files  

---

## Installation

```bash
# Install dependencies (NO graph libraries needed!)
pip install -r requirements.txt
```

---

## Available Models

All models are implemented in pure PyTorch:

- **GCN** - Graph Convolutional Network
- **GAT** - Graph Attention Network
- **GIN** - Graph Isomorphism Network
- **PDGLSTM** - Position-aware Dynamic Graph LSTM

---

## Project Structure

```
.
├── config.py               # Configuration parameters
├── dataset.py              # Dataset loading (pure PyTorch)
├── gcn.py                  # GCN model (pure PyTorch)
├── gin.py                  # GIN model (pure PyTorch)
├── pdglstm.py              # PDGLSTM model (pure PyTorch)
├── trainer.py              # Training/evaluation logic
├── utils.py                # Metrics and visualization
├── main.py                 # Entry point for training
├── predict_edges_only.py   # Edge-only prediction script
├── requirements.txt        # Dependencies (no graph libraries!)
└── README.md               # Documentation
```

---

## Dataset Format

### Node CSV Format
The node CSV should contain:
- Node identifiers
- Node features (if any)
- Labels for supervised learning
- Simulation ID (`sim` column) for grouping multiple graphs

### Edge CSV Format
The edge CSV should contain:
- Source node IDs (e.g., `new_from`, `src`, `from`)
- Target node IDs (e.g., `new_to`, `dst`, `to`)
- Edge features (e.g., `weight1`, `weight2` or normalized versions)
- Simulation ID (`sim` column) for grouping multiple graphs

**Note:** The script automatically detects column names, so you can use various naming conventions.

---

## Usage

### Training with Direct CSV Input

You can now provide direct paths to your CSV files instead of relying on directory structures:

```bash
python main.py \
  --mode train \
  --model gcn \
  --model_num 1 \
  --train_csv "/path/to/train.csv" \
  --train_edge_csv "/path/to/train_edge.csv" \
  --val_csv "/path/to/valid.csv" \
  --val_edge_csv "/path/to/valid_edge.csv" \
  --test_csv "/path/to/test.csv" \
  --test_edge_csv "/path/to/test_edge.csv" \
  --hidden_dim 128 \
  --num_layers 20 \
  --batch_size 4 \
  --lr 0.001 \
  --max_epochs 100
```

**Automatic Edge CSV Detection:** If you only provide node CSV paths, the script will automatically look for corresponding edge CSV files by replacing `.csv` with `_edge.csv`:

```bash
python main.py \
  --mode train \
  --model gcn \
  --train_csv "/path/to/train.csv" \
  --val_csv "/path/to/valid.csv" \
  --test_csv "/path/to/test.csv"
  # Automatically looks for train_edge.csv, valid_edge.csv, test_edge.csv
```

**Legacy Directory-Based Approach:** You can still use the old approach with `--ds_dir`, `--ds_name`, and `--ds_split`:

```bash
python main.py \
  --mode train \
  --ds_dir 'cleaned_data' \
  --ds_name 'ddf_resp_20230131' \
  --ds_split 'split_rs123' \
  --model gcn \
  --batch_size 4
```

### Evaluation

Evaluate a trained model on the test set:

```bash
python main.py \
  --mode eval \
  --model gcn \
  --checkpoint 'experiments/gcn_1/best_model.pth' \
  --test_csv "/path/to/test.csv" \
  --test_edge_csv "/path/to/test_edge.csv"
```

### Edge-Only Prediction

Make predictions using **only edge data** (no node CSV required):

```bash
python predict_edges_only.py \
  --edge_csv "/path/to/edges.csv" \
  --model_py pdglstm.py \
  --checkpoint "experiments/PDGLSTM_0/best_model.pth" \
  --output predictions.xlsx \
  --leaf_only 1
```

**Parameters:**
- `--edge_csv`: Path to the edge CSV file
- `--model_py`: Python file defining the model architecture (e.g., `pdglstm.py`, `gcn.py`)
- `--checkpoint`: Path to the trained model checkpoint
- `--output`: Output file path (`.xlsx` for Excel, otherwise CSV)
- `--leaf_only`: Set to `1` to predict only leaf nodes, `0` for all nodes
- `--device`: Device selection (`auto`, `cpu`, or `cuda`)

**Output Formats:**
- **Excel (`.xlsx`)**: Creates a workbook with two sheets - `nodes` and `edges`
- **CSV**: Creates two files - `{name}_nodes.csv` and `{name}_edges.csv`

**Output Columns:**
- Node predictions include: `sim`, `node_id`, `pred_class_id`, `pred_class_name`, and probability scores
- Edge predictions include: original edge data plus destination node predictions

---

## Configuration Options

### General
- `--seed`: Random seed (default: 123)
- `--device`: Device to use (cuda/cpu)
- `--mode`: train or eval
- `--log_level`: Logging level (INFO/DEBUG/WARNING)

### Data - Direct CSV Input (Recommended)
- `--train_csv`: Path to training node CSV
- `--train_edge_csv`: Path to training edge CSV (auto-detected if omitted)
- `--val_csv`: Path to validation node CSV
- `--val_edge_csv`: Path to validation edge CSV (auto-detected if omitted)
- `--test_csv`: Path to test node CSV
- `--test_edge_csv`: Path to test edge CSV (auto-detected if omitted)

### Data - Legacy Directory-Based
- `--ds_name`: Dataset name
- `--ds_dir`: Dataset directory
- `--ds_split`: Dataset split subdirectory

### Data Loading
- `--batch_size`: Batch size (default: 4)
- `--num_workers`: DataLoader workers (default: 0)

### Model
- `--model`: Model type (gcn/gat/gin/pdglstm)
- `--model_num`: Model instance number for saving
- `--hidden_dim`: Hidden dimension (default: 128)
- `--num_layers`: Number of layers (default: 20)
- `--dropout`: Dropout rate (default: 0.5)

### Training
- `--max_epochs`: Maximum epochs (default: 100)
- `--optimizer`: Optimizer (Adam/SGD/RMSprop)
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay (default: 0.0004)
- `--early_stopping`: Early stopping patience (default: 10)

### Evaluation
- `--checkpoint`: Path to model checkpoint file

---

## Examples

### Example 1: Training GCN with Direct CSV Paths

```bash
python main.py \
  --mode train \
  --model gcn \
  --model_num 1 \
  --train_csv "/data/train.csv" \
  --train_edge_csv "/data/train_edge.csv" \
  --val_csv "/data/valid.csv" \
  --val_edge_csv "/data/valid_edge.csv" \
  --test_csv "/data/test.csv" \
  --test_edge_csv "/data/test_edge.csv" \
  --hidden_dim 64 \
  --num_layers 10 \
  --batch_size 8 \
  --lr 0.001 \
  --max_epochs 50
```

### Example 2: Training PDGLSTM with Auto-Detection

```bash
python main.py \
  --mode train \
  --model pdglstm \
  --train_csv "/data/train.csv" \
  --val_csv "/data/valid.csv" \
  --test_csv "/data/test.csv" \
  --batch_size 4
```

### Example 3: Prediction on New Data

```bash
python predict_edges_only.py \
  --edge_csv "/data/new_edges.csv" \
  --model_py pdglstm.py \
  --checkpoint "experiments/PDGLSTM_1/best_model.pth" \
  --output results.xlsx \
  --leaf_only 1
```

---

## Output Files

### Training Mode
- `experiments/{model}_{model_num}/best_model.pth` - Best model checkpoint
- `experiments/{model}_{model_num}/training_log.txt` - Training logs

### Prediction Mode (Edge-Only)
- Excel format (`.xlsx`): Single workbook with `nodes` and `edges` sheets
- CSV format: Two files - `{name}_nodes.csv` and `{name}_edges.csv`

**Node Predictions Include:**
- Simulation ID
- Node ID
- Predicted class (ID and name)
- Probability scores for all classes (static, decay, growth, background)

**Edge Predictions Include:**
- All original edge data
- Destination node predictions
- Probability scores for destination nodes

---

## Class Labels

The model predicts one of four classes:
- **0**: Static
- **1**: Decay
- **2**: Growth
- **3**: Background

---

## Troubleshooting

### Issue: Edge CSV not found
**Solution:** Make sure edge CSV files follow the naming convention `{name}_edge.csv` or explicitly provide paths using `--{phase}_edge_csv` flags.

### Issue: Column detection fails
**Solution:** The script auto-detects various column names (`new_from/new_to`, `src/dst`, `from/to`, etc.). Check your CSV headers match these patterns.

### Issue: Multiple graphs not recognized
**Solution:** Ensure your CSV has a `sim` column to group edges into different graphs. If missing, all edges are treated as a single graph.

---

## Acknowledgments

This implementation is based on the original [DeepDynaForecast](https://github.com/lab-smile/DeepDynaForecast) repository. Please cite their work if you use their datasets or methodology.

---

## License

Please refer to the original [DeepDynaForecast repository](https://github.com/lab-smile/DeepDynaForecast) for licensing information.
