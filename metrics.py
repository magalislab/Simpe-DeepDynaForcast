# metrics.py
"""Evaluation metrics and analysis."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
    confusion_matrix, roc_curve, auc, log_loss
)
from sklearn.preprocessing import label_binarize
from scipy.special import softmax
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """Calculate various classification metrics."""
    
    def __init__(self, num_classes: int = 3):
        self.num_classes = num_classes
        self.class_names = ['static', 'decay', 'growth']
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred_scores: np.ndarray
    ) -> Dict[str, float]:
        """Compute all metrics."""
        # Remove background class (class 3)
        mask = y_true != 3
        y_true = y_true[mask]
        y_pred_scores = y_pred_scores[mask][:, :self.num_classes]
        
        # Get predicted classes
        y_pred = np.argmax(y_pred_scores, axis=1)
        
        # Convert scores to probabilities
        y_pred_probs = softmax(y_pred_scores, axis=1)
        
        # One-hot encode true labels
        y_true_onehot = label_binarize(
            y_true, 
            classes=list(range(self.num_classes))
        )
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'cross_entropy': log_loss(y_true_onehot, y_pred_probs),
            'brier_score': self._brier_score(y_true_onehot, y_pred_probs),
        }
        
        # Add AUC scores
        try:
            metrics['auc_macro_ovo'] = roc_auc_score(
                y_true_onehot, y_pred_scores, 
                multi_class="ovo", average="macro"
            )
            metrics['auc_weighted_ovo'] = roc_auc_score(
                y_true_onehot, y_pred_scores,
                multi_class="ovo", average="weighted"
            )
        except ValueError:
            metrics['auc_macro_ovo'] = 0.0
            metrics['auc_weighted_ovo'] = 0.0
        
        return metrics
    
    @staticmethod
    def _brier_score(y_true_onehot: np.ndarray, y_pred_probs: np.ndarray) -> float:
        """Calculate Brier score."""
        return np.mean(np.sum((y_true_onehot - y_pred_probs) ** 2, axis=1))
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Path
    ):
        """Plot and save confusion matrix."""
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt=".3f", 
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cmap="YlGnBu", cbar_kws={'label': 'Proportion'}
        )
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.title('Confusion Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_path / "confusion_matrix.png", dpi=300)
        plt.close()
