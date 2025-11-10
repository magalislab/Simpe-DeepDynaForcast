"""Utility functions for metrics and visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, auc, roc_auc_score, log_loss, brier_score_loss
)
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
import os.path as osp


def calculate_metrics(y_true, y_pred, y_scores, num_classes=3):
    """
    Calculate comprehensive metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_scores: Prediction scores (logits or probabilities)
        num_classes: Number of classes (excluding background)
    
    Returns:
        Dictionary of metrics
    """
    # Filter out background class (label 3)
    mask = y_true != 3
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    y_scores_filtered = y_scores[mask, :num_classes]
    
    if len(y_true_filtered) == 0:
        return {}
    
    # Convert scores to probabilities
    y_prob = softmax(y_scores_filtered, axis=1)
    
    # Binarize labels for multi-class metrics
    y_true_bin = label_binarize(y_true_filtered, classes=list(range(num_classes)))
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true_filtered, y_pred_filtered),
        'balanced_accuracy': balanced_accuracy_score(y_true_filtered, y_pred_filtered),
        'f1_weighted': f1_score(y_true_filtered, y_pred_filtered, average='weighted'),
        'f1_macro': f1_score(y_true_filtered, y_pred_filtered, average='macro'),
        'precision': precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0),
        'recall': recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0),
        'brier_score': np.mean(np.sum((y_true_bin - y_prob) ** 2, axis=1)),
        'cross_entropy': log_loss(y_true_bin, y_prob)
    }
    
    # ROC AUC scores
    try:
        metrics['macro_auc_ovo'] = roc_auc_score(y_true_bin, y_prob, 
                                                 multi_class='ovo', average='macro')
        metrics['weighted_auc_ovo'] = roc_auc_score(y_true_bin, y_prob, 
                                                     multi_class='ovo', average='weighted')
        metrics['macro_auc_ovr'] = roc_auc_score(y_true_bin, y_prob, 
                                                 multi_class='ovr', average='macro')
        metrics['weighted_auc_ovr'] = roc_auc_score(y_true_bin, y_prob, 
                                                     multi_class='ovr', average='weighted')
    except ValueError:
        metrics['macro_auc_ovo'] = 0.0
        metrics['weighted_auc_ovo'] = 0.0
        metrics['macro_auc_ovr'] = 0.0
        metrics['weighted_auc_ovr'] = 0.0
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels
        save_path: Path to save the plot
    """
    # Filter background class
    mask = y_true != 3
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.3f', cmap='YlGnBu', 
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Proportion'})
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curves(y_true, y_scores, num_classes, save_path):
    """
    Plot ROC curves for each class.
    
    Args:
        y_true: True labels
        y_scores: Prediction scores
        num_classes: Number of classes
        save_path: Path to save the plot
    """
    # Filter background
    mask = y_true != 3
    y_true = y_true[mask]
    y_scores = y_scores[mask, :num_classes]
    
    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
    y_prob = softmax(y_scores, axis=1)
    
    # Compute ROC curve for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    label_names = {0: 'static', 1: 'decay', 2: 'growth'}
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{label_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curves', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_predictions(y_true, y_pred, y_scores, save_path):
    """Save predictions to numpy file."""
    pred_dict = {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_scores': y_scores
    }
    np.save(save_path, pred_dict)


def print_metrics(metrics, logger=None):
    """Print metrics in a formatted way."""
    output = "\nMetrics:\n" + "="*50 + "\n"
    for key, value in metrics.items():
        if isinstance(value, float):
            output += f"{key:20s}: {value:.4f}\n"
        else:
            output += f"{key:20s}: {value}\n"
    output += "="*50
    
    if logger:
        logger.info(output)
    else:
        print(output)