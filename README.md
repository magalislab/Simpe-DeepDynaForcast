# Simple Implementation of DeepDynaForecast

A clean implementation of DeepDynaForecast using **pure PyTorch** (no external graph libraries!).

## Key Features 
✅ **Clean Code Structure** - Easy to read and modify  

---

## Installation

```bash
# Install dependencies (NO graph libraries needed!)
pip install -r requirements.txt
```

---

## Available Models

All models are implemented in pure PyTorch:

- **GCN** 
- **GIN**
- **PDGLSTM** 

GCN and GIN serve as baseline models, while PDGLSTM is the proposed DeepDynaForecast architecture.

---

## Dataset Setup

For training the model, you need to download and preprocess the dataset from the original [DeepDynaForecast GitHub repository](https://github.com/lab-smile/DeepDynaForecast/tree/main). 

### Steps:
1. Download the dataset from the original repository
2. Preprocess the data as needed
3. Place the datasets into the appropriate folder

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

For evaluation on already saved models:

1. Download the checkpoints from the original [DeepDynaForecast GitHub repository](https://github.com/lab-smile/DeepDynaForecast/tree/main)
2. Place these models in the appropriate folder
3. Run the evaluation command:

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

## Class Labels

The model predicts one of four classes:
- **0**: Static
- **1**: Decay
- **2**: Growth
- **3**: Background

---

## Acknowledgments

This implementation is based on the original [DeepDynaForecast](https://github.com/lab-smile/DeepDynaForecast) repository. Please cite their work if you use their datasets or methodology.

