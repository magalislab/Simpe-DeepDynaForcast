""""Training and evaluation logic — self-contained metrics & plots."""

import os
import os.path as osp
import json
import logging
from types import SimpleNamespace
from itertools import cycle

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_auc_score, roc_curve, log_loss, auc
)

class Trainer:
    """
    Handles model training, evaluation, plotting, and metric saving.

    Notes:
      • Set optional args:
          - args.model (str): used in filenames
          - args.device (str): "cuda" / "cpu" (defaults to cuda if available)
          - args.optimizer: 'Adam'|'SGD'|'RMSprop'
          - args.lr, args.weight_decay
          - args.lr_decay_mode: None|'step'|'plateau'
          - args.lr_decay_step (int), args.lr_decay_rate (float)
          - args.grad_clip (float)
          - args.max_epochs (int), args.early_stopping (int)
          - args.loss_ignore_bg (bool): if True, bg class weight=0 in CE
          - args.background_class (int|None): default 3; None = include all
          - args.label_names (list[str] | dict[int,str]): optional class names
          - args.log_level (str): 'INFO'|'DEBUG'|...
    """

    # ------------------------------
    # Init
    # ------------------------------
    def __init__(self, model, train_loader, val_loader, test_loader, args, save_dir, checkpoint_path):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args
        self.save_dir = save_dir
        self.checkpoint_path = checkpoint_path

        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(osp.join(self.save_dir, "val"), exist_ok=True)
        os.makedirs(osp.join(self.save_dir, "test"), exist_ok=True)

        # Logger
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            level = getattr(logging, getattr(args, "log_level", "INFO").upper(), logging.INFO)
            self.logger.setLevel(level)
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter('%(asctime)s %(levelname)s - %(message)s', "%H:%M:%S"))
            self.logger.addHandler(h)

        # Device
        device_str = getattr(self.args, "device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer / Scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Loss weights (0..2 foreground, 3=background by default)
        label_weights = [0.34, 51.52, 22.08]
        if getattr(self.args, "loss_ignore_bg", False):
            label_weights.append(0.0)
        else:
            label_weights.append(1.0)
        self.loss_weights = torch.tensor(label_weights, dtype=torch.float32, device=self.device)

        # Tracking
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.epochs_no_improve = 0

    # ------------------------------
    # Optimizer / Scheduler
    # ------------------------------
    def _create_optimizer(self):
        opt = getattr(self.args, "optimizer", "Adam")
        lr = float(getattr(self.args, "lr", 1e-3))
        wd = float(getattr(self.args, "weight_decay", 0.0))
        if opt == 'Adam':
            return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt == 'SGD':
            return SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
        elif opt == 'RMSprop':
            return RMSprop(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        raise ValueError(f"Unknown optimizer: {opt}")

    def _create_scheduler(self):
        mode = getattr(self.args, "lr_decay_mode", None)
        step = int(getattr(self.args, "lr_decay_step", 10))
        gamma = float(getattr(self.args, "lr_decay_rate", 0.1))
        if mode == 'step':
            return StepLR(self.optimizer, step_size=step, gamma=gamma)
        elif mode == 'plateau':
            return ReduceLROnPlateau(self.optimizer, mode='min', factor=gamma, patience=step, verbose=True)
        return None

    # ------------------------------
    # Training / Eval loops
    # ------------------------------
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss, num_batches = 0.0, 0
        grad_clip = float(getattr(self.args, "grad_clip", 1.0))

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} - Training"):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            out = self.model(batch)
            if isinstance(out, tuple):
                out = out[0]

            loss = F.cross_entropy(out, batch.y, weight=self.loss_weights, reduction='mean')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return {'loss': avg_loss}

    @torch.no_grad()
    def evaluate(self, loader, phase='val'):
        """Evaluate model, compute/save metrics & plots."""
        self.model.eval()
        total_loss, num_batches = 0.0, 0

        logits_all, labels_all, preds_all = [], [], []

        for batch in tqdm(loader, desc=f"Evaluating {phase}"):
            batch = batch.to(self.device)
            out = self.model(batch)
            if isinstance(out, tuple):
                out = out[0]

            loss = F.cross_entropy(out, batch.y, weight=self.loss_weights, reduction='mean')
            total_loss += loss.item()
            num_batches += 1

            logits = out.detach().cpu().numpy()
            labels = batch.y.detach().cpu().numpy()
            preds = np.argmax(logits, axis=1)

            logits_all.append(logits)
            labels_all.append(labels)
            preds_all.append(preds)

        avg_loss = total_loss / max(1, num_batches)
        all_scores = np.concatenate(logits_all, axis=0)
        all_labels = np.concatenate(labels_all, axis=0)
        all_preds_incl = np.concatenate(preds_all, axis=0)

        # Compute metrics (excluding background if requested)
        bg_cls = getattr(self.args, "background_class", 3)
        proc = self._process_predictions(all_labels, all_scores, background_class=bg_cls)

        metric_dict = self._compute_metrics(proc)
        metric_dict["loss"] = avg_loss

        # Also report accuracy including bg (original behavior sometimes needed)
        metric_dict["acc_including_bg"] = float((all_preds_incl == all_labels).mean())

        # Save plots and CSVs
        phase_dir = osp.join(self.save_dir, phase)
        os.makedirs(phase_dir, exist_ok=True)

        # ROC plots (only if at least 2 classes present)
        if proc["n_classes"] >= 2 and len(np.unique(proc["y_true"])) >= 2:
            fpr, tpr, roc_auc = self._get_fpr_tpr(proc["y_true_onehot"], proc["y_prob"])
            self._plot_roc_per_class(fpr, tpr, roc_auc, proc["labels_display"], osp.join(phase_dir, "ROC_per_class.png"))
            self._plot_roc_merged(fpr, tpr, roc_auc, proc["labels_display"], osp.join(phase_dir, "ROC_merged.png"))
            # Save macro ROC curve to CSV
            self._save_macro_roc_csv(fpr, tpr, osp.join(phase_dir, "ROC"), getattr(self.args, "model", "model"))
        else:
            self.logger.info(f"[{phase.upper()}] Skipping ROC curves (not enough classes present).")

        # Confusion matrix (normalized)
        self._plot_confusion_matrix(proc["y_true"], proc["y_pred"], proc["labels_display"],
                                    osp.join(phase_dir, "confusion_mat.png"),
                                    osp.join(phase_dir, "confusion_mat.eps"))
        # Save CM variants
        self._save_confusion_csvs(proc["y_true"], proc["y_pred"], proc["classes_kept"],
                                  osp.join(phase_dir, "cm"), getattr(self.args, "model", "model"))

        # Persist metrics
        with open(osp.join(phase_dir, "metrics.json"), "w") as f:
            json.dump(metric_dict, f, indent=2)

        # Compact log
        self.logger.info(
            f"[{phase.upper()}] loss={avg_loss:.4f} | "
            f"acc={metric_dict['acc']:.4f} | bal_acc={metric_dict['balance_acc']:.4f} | "
            f"f1_macro={metric_dict['f1_macro']:.4f} | f1_w={metric_dict['f1_weighted']:.4f} | "
            f"prec={metric_dict['precision_macro']:.4f} | rec={metric_dict['recall_macro']:.4f} | "
            f"auc_ovr_macro={metric_dict['macro_auc_ovr']:.4f} | auc_ovr_w={metric_dict['weighted_auc_ovr']:.4f}"
        )

        return {
            'loss': avg_loss,
            'metrics': metric_dict,
            'predictions': proc["y_pred"],
            'labels': proc["y_true"],
            'scores': proc["y_prob"],   # probabilities after softmax and bg removal
        }

    def train(self):
        """Main training loop with early stopping on val loss."""
        self.logger.info("Starting training...")
        early_patience = int(getattr(self.args, "early_stopping", 20))
        max_epochs = int(getattr(self.args, "max_epochs", 100))

        for epoch in range(1, max_epochs + 1):
            tr = self.train_epoch(epoch)
            self.logger.info(f"Epoch {epoch} - Train Loss: {tr['loss']:.4f}")

            val_out = self.evaluate(self.val_loader, phase='val')
            val_loss = val_out['loss']

            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, is_best=True)
                self.logger.info(f"New best model at epoch {epoch}!")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= early_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self.logger.info(f"Training completed. Best epoch: {self.best_epoch}")

    # ------------------------------
    # Checkpoint helpers
    # ------------------------------
    def save_checkpoint(self, epoch, is_best=False):
        state_dict = self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'state_dict': state_dict,  # duplicated for flexible loaders
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'args': self.args,
        }
        fn = 'best_model.pth' if is_best else f'checkpoint_{epoch}.pth'
        path = osp.join(self.save_dir, fn)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        if state is None:
            raise KeyError("Checkpoint missing 'state_dict' or 'model_state_dict'.")
        new_state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
        self.model.load_state_dict(new_state, strict=True)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', self.best_loss)
        self.logger.info(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)

    def load_from_saved_checkpoint(self, path):
        self.logger.info(f"Loading checkpoint (weights only) from: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        state = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        if state is None:
            raise KeyError("Checkpoint missing 'state_dict' or 'model_state_dict'.")
        new_state = {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in state.items()}
        self.model.load_state_dict(new_state, strict=True)
        self.model.eval()

    def test(self):
        """Evaluate on test set with full metrics & plots."""
        ckpt_path = self.checkpoint_path
        if not ckpt_path or not osp.exists(ckpt_path):
            cand = osp.join(self.save_dir, 'best_model.pth')
            if osp.exists(cand):
                ckpt_path = cand

        if ckpt_path and osp.exists(ckpt_path):
            try:
                self.load_from_saved_checkpoint(ckpt_path)
            except Exception:
                self.load_checkpoint(ckpt_path)

        out = self.evaluate(self.test_loader, phase='test')
        self.logger.info(
            f"Test Loss: {out['loss']:.4f}, "
            f"Accuracy: {out['metrics']['acc']:.4f}, "
            f"Balanced Acc: {out['metrics']['balance_acc']:.4f}, "
            f"F1(macro): {out['metrics']['f1_macro']:.4f}"
        )
        return out

    # ------------------------------
    # Helpers: metrics processing
    # ------------------------------
    @staticmethod
    def _stable_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _resolve_label_names(self, classes_kept):
        ln = getattr(self.args, "label_names", None)
        if isinstance(ln, dict):
            names = [ln.get(int(c), f"class_{c}") for c in classes_kept]
        elif isinstance(ln, (list, tuple)):
            # assume index == class id if long enough; else fallback
            names = []
            for c in classes_kept:
                if int(c) < len(ln):
                    names.append(str(ln[int(c)]))
                else:
                    names.append(f"class_{c}")
        else:
            # sensible defaults for first 3 classes
            default = {0: "static", 1: "decay", 2: "growth"}
            names = [default.get(int(c), f"class_{c}") for c in classes_kept]
        return names

    def _process_predictions(self, y_true_raw: np.ndarray, logits_raw: np.ndarray, background_class=3):
        """Filter background (if any), make one-hot, and map names."""
        num_classes = logits_raw.shape[1]
        classes_all = list(range(num_classes))

        # probabilities
        probs_all = self._stable_softmax(logits_raw, axis=1)

        # filter background samples
        if background_class is not None and background_class in classes_all:
            keep_mask = (y_true_raw != background_class)
            y_true = y_true_raw[keep_mask]
            probs = probs_all[keep_mask]
            # drop background column to keep dimensions consistent
            keep_cols = [c for c in classes_all if c != background_class]
            probs = probs[:, keep_cols]
            classes_kept = keep_cols
        else:
            y_true = y_true_raw.copy()
            probs = probs_all.copy()
            classes_kept = classes_all

        n_classes = len(classes_kept)
        labels_display = self._resolve_label_names(classes_kept)

        # one-hot of true labels over classes_kept (values must match class ids)
        y_true_onehot = label_binarize(y_true, classes=classes_kept) if n_classes >= 2 else np.zeros((len(y_true), n_classes))

        # predicted labels in original id space (map argmax position -> class id)
        if n_classes > 0:
            pred_pos = np.argmax(probs, axis=1)
            y_pred = np.array([classes_kept[i] for i in pred_pos], dtype=int)
        else:
            y_pred = np.array([], dtype=int)

        return {
            "y_true": y_true,                         # original label ids (filtered)
            "y_pred": y_pred,                         # original label ids (filtered)
            "y_prob": probs,                          # prob for kept classes (N, K)
            "y_true_onehot": y_true_onehot,           # one-hot over kept classes (N, K)
            "classes_kept": classes_kept,             # original ids for columns in y_prob
            "n_classes": n_classes,
            "labels_display": labels_display,         # names for plotting
        }

    def _compute_metrics(self, proc):
        y_true = proc["y_true"]
        y_pred = proc["y_pred"]
        y_prob = proc["y_prob"]
        y_onehot = proc["y_true_onehot"]
        classes_kept = proc["classes_kept"]
        n_classes = proc["n_classes"]

        # graceful defaults if degenerate
        if len(y_true) == 0 or n_classes == 0:
            return {
                "acc": 0.0, "balance_acc": 0.0,
                "precision_macro": 0.0, "recall_macro": 0.0,
                "f1_macro": 0.0, "f1_weighted": 0.0,
                "brier_score": 0.0, "cross_entropy": 0.0,
                "macro_auc_ovo": 0.0, "weighted_auc_ovo": 0.0,
                "macro_auc_ovr": 0.0, "weighted_auc_ovr": 0.0,
                "acc_excluding_bg": 0.0,
            }

        # Basic metrics (macro for precision/recall)
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        f1_w = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_m = f1_score(y_true, y_pred, average='macro', zero_division=0)
        prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)

        # Brier (multi-class) and CE
        if y_onehot.shape[1] == y_prob.shape[1] and y_onehot.shape[1] >= 2:
            brier = float(np.sum((y_onehot - y_prob) ** 2) / y_onehot.shape[0])
            ce = float(log_loss(y_onehot, y_prob))
        else:
            brier, ce = 0.0, 0.0

        # ROC-AUC (guards if fewer than 2 classes present)
        try:
            macro_auc_ovo = float(roc_auc_score(y_onehot, y_prob, multi_class="ovo", average="macro"))
        except Exception:
            macro_auc_ovo = 0.0
        try:
            weighted_auc_ovo = float(roc_auc_score(y_onehot, y_prob, multi_class="ovo", average="weighted"))
        except Exception:
            weighted_auc_ovo = 0.0
        try:
            macro_auc_ovr = float(roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            macro_auc_ovr = 0.0
        try:
            weighted_auc_ovr = float(roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="weighted"))
        except Exception:
            weighted_auc_ovr = 0.0

        return {
            "acc": float(acc),
            "balance_acc": float(bal_acc),
            "precision_macro": float(prec_m),
            "recall_macro": float(rec_m),
            "f1_macro": float(f1_m),
            "f1_weighted": float(f1_w),
            "brier_score": float(brier),
            "cross_entropy": float(ce),
            "macro_auc_ovo": macro_auc_ovo,
            "weighted_auc_ovo": weighted_auc_ovo,
            "macro_auc_ovr": macro_auc_ovr,
            "weighted_auc_ovr": weighted_auc_ovr,
            "acc_excluding_bg": float(acc),  # all metrics are on filtered set
        }

    # ------------------------------
    # Helpers: ROC/CM plots & CSVs
    # ------------------------------
    def _get_fpr_tpr(self, y_onehot: np.ndarray, y_prob: np.ndarray):
        n_classes = y_onehot.shape[1]
        fpr, tpr, roc_auc = {}, {}, {}

        # Per-class
        for i in range(n_classes):
            if np.sum(y_onehot[:, i]) == 0 or np.sum(1 - y_onehot[:, i]) == 0:
                # degenerate (no positives or no negatives) -> skip
                fpr[i], tpr[i], roc_auc[i] = np.array([0, 1]), np.array([0, 1]), 0.0
                continue
            fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average
        fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # Macro-average by interpolation
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if isinstance(i, int)]))
        mean_tpr = np.zeros_like(all_fpr)
        valid = 0
        for i in range(n_classes):
            if isinstance(i, int):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                valid += 1
        mean_tpr /= max(1, valid)
        fpr["macro"], tpr["macro"] = all_fpr, mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        return fpr, tpr, roc_auc

    def _plot_roc_per_class(self, fpr, tpr, roc_auc, labels, save_path):
        n_classes = len([k for k in fpr.keys() if isinstance(k, int)])
        n_row = 1 if n_classes <= 3 else 2
        n_col = n_classes if n_classes <= 3 else int(np.ceil(n_classes / 2))
        fig_w, fig_h = max(10, 6 * n_col), max(6, 5 * n_row)

        fig, axs = plt.subplots(n_row, n_col, figsize=(fig_w, fig_h))
        axs = np.array(axs).reshape(-1) if n_row * n_col > 1 else np.array([axs])

        for i in range(n_classes):
            ax = axs[i]
            ax.plot(fpr[i], tpr[i], linewidth=2, label=f"AUC = {roc_auc[i]:.2f}")
            ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
            ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate' if i % n_col == 0 else '')
            title = labels[i] if i < len(labels) else f"class_{i}"
            ax.set_title(f"ROC: {title}")
            ax.legend(loc="lower right")
        plt.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    def _plot_roc_merged(self, fpr, tpr, roc_auc, labels, save_path):
        fig = plt.figure(figsize=(8, 8))
        lw = 2

        if "micro" in fpr and "micro" in tpr:
            plt.plot(fpr["micro"], tpr["micro"], linestyle=':', linewidth=4, label=f"micro-average (AUC={roc_auc.get('micro', 0):.2f})")
        if "macro" in fpr and "macro" in tpr:
            plt.plot(fpr["macro"], tpr["macro"], linestyle=':', linewidth=4, label=f"macro-average (AUC={roc_auc.get('macro', 0):.2f})")

        colors = cycle([None, None, None, None, None])  # default matplotlib colors
        i = 0
        while isinstance(i, int) and i in fpr:
            plt.plot(fpr[i], tpr[i], linewidth=lw, label=f"class {labels[i] if i < len(labels) else i} (AUC={roc_auc[i]:.2f})")
            i += 1
            if i not in fpr: break

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC')
        plt.legend(loc="lower right", fontsize=9)
        plt.tight_layout()
        fig.savefig(save_path, dpi=200)
        plt.close(fig)

    def _plot_confusion_matrix(self, y_true, y_pred, labels, png_path, eps_path):
        cm = confusion_matrix(y_true, y_pred, labels=[*{*y_true, *y_pred}], normalize="true")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='YlGnBu', vmin=0.0, vmax=1.0)
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_ylabel('True labels'); ax.set_xlabel('Predicted labels')

        ticks = np.arange(len(cm))
        tick_labels = [labels[i] if i < len(labels) else f"class_{i}" for i in range(len(cm))]
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right"); ax.set_yticklabels(tick_labels)

        # annotate
        fmt = ".3f"
        thresh = (cm.max() + cm.min()) / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        fig.savefig(eps_path)
        plt.close(fig)

    def _save_macro_roc_csv(self, fpr, tpr, base_dir, model_name):
        os.makedirs(base_dir, exist_ok=True)
        macro_fpr = fpr.get("macro", np.array([]))
        macro_tpr = tpr.get("macro", np.array([]))
        if macro_fpr.size == 0 or macro_tpr.size == 0:
            return
        out_csv = osp.join(base_dir, f"roc_{model_name}.csv")
        import pandas as pd
        df = pd.DataFrame({"fpr": macro_fpr, "tpr": macro_tpr})
        df.to_csv(out_csv, index=False)

    def _save_confusion_csvs(self, y_true, y_pred, classes_kept, base_dir, model_name):
        os.makedirs(base_dir, exist_ok=True)
        import pandas as pd

        # pairs
        df_pairs = pd.DataFrame({"true": y_true, "predict": y_pred})
        df_pairs.to_csv(osp.join(base_dir, f"cm_pairs_{model_name}.csv"), index=False)

        # normalized matrix (rows: true, cols: pred)
        labels_sorted = sorted(list(set(list(y_true) + list(y_pred))))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted, normalize="true")
        df_cm = pd.DataFrame(cm, index=[f"class_{c}" for c in labels_sorted],
                                columns=[f"class_{c}" for c in labels_sorted])
        df_cm.to_csv(osp.join(base_dir, f"cm_matrix_{model_name}.csv"))
