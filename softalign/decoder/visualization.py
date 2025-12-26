"""
Metrics tracking and visualization for training.

This module provides:
- MetricsTracker class for accumulating training/validation metrics
- Plotting functions for training curves, confusion matrices, etc.
"""

import os
import io
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt

from .model import NUM_INDEX_CLASSES


class MetricsTracker:
    """Track training metrics for visualization."""

    def __init__(self, num_classes=NUM_INDEX_CLASSES):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.step_losses = []
        self.step_ce_losses = []
        self.step_wd_losses = []
        self.step_accs = []

        self.epoch_train_loss = []
        self.epoch_train_ce_loss = []
        self.epoch_train_wd_loss = []
        self.epoch_train_acc = []
        self.epoch_val_loss = []
        self.epoch_val_acc = []

        self.best_val_loss = float('inf')
        self.best_epoch = 0

        self.position_correct = np.zeros(self.num_classes)
        self.position_total = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    def add_step(self, loss, ce_loss, wd_loss, accuracy):
        """Record metrics for a training step."""
        self.step_losses.append(float(loss))
        self.step_ce_losses.append(float(ce_loss))
        self.step_wd_losses.append(float(wd_loss))
        self.step_accs.append(float(accuracy))

    def end_epoch(self, val_loss=None, val_acc=None):
        """Finalize epoch metrics."""
        self.epoch_train_loss.append(np.mean(self.step_losses))
        self.epoch_train_ce_loss.append(np.mean(self.step_ce_losses))
        self.epoch_train_wd_loss.append(np.mean(self.step_wd_losses))
        self.epoch_train_acc.append(np.mean(self.step_accs))

        if val_loss is not None:
            self.epoch_val_loss.append(val_loss)
        if val_acc is not None:
            self.epoch_val_acc.append(val_acc)

        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = len(self.epoch_train_loss)

        self.step_losses = []
        self.step_ce_losses = []
        self.step_wd_losses = []
        self.step_accs = []

    def update_position_stats(self, predictions, targets, mask):
        """Update per-position accuracy statistics."""
        predictions = np.array(predictions)
        targets = np.array(targets)
        mask = np.array(mask)

        for b in range(predictions.shape[0]):
            for t in range(predictions.shape[1]):
                if mask[b, t] > 0:
                    target_idx = int(targets[b, t])
                    pred_idx = int(predictions[b, t])

                    if 0 <= target_idx < self.num_classes:
                        self.position_total[target_idx] += 1
                        if pred_idx == target_idx:
                            self.position_correct[target_idx] += 1

                        if 0 <= pred_idx < self.num_classes:
                            self.confusion_matrix[target_idx, pred_idx] += 1

    def get_position_accuracy(self):
        """Get per-position accuracy."""
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = self.position_correct / self.position_total
            acc = np.nan_to_num(acc, nan=0.0)
        return acc

    def reset_position_stats(self):
        """Reset per-position statistics for new evaluation."""
        self.position_correct = np.zeros(self.num_classes)
        self.position_total = np.zeros(self.num_classes)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))


def plot_training_curves(metrics, output_dir, epoch):
    """Plot training and validation curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(metrics.epoch_train_loss) + 1)

    ax = axes[0, 0]
    ax.plot(epochs, metrics.epoch_train_loss, 'b-', label='Total Loss', linewidth=2)
    ax.plot(epochs, metrics.epoch_train_ce_loss, 'g--', label='CE Loss', linewidth=1.5)
    if any(wd > 0 for wd in metrics.epoch_train_wd_loss):
        ax.plot(epochs, metrics.epoch_train_wd_loss, 'r:', label='WD Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if metrics.epoch_val_loss:
        val_epochs = range(1, len(metrics.epoch_val_loss) + 1)
        ax.plot(val_epochs, metrics.epoch_val_loss, 'b-', linewidth=2, label='Val Loss')
        if metrics.best_epoch > 0 and metrics.best_epoch <= len(metrics.epoch_val_loss):
            ax.axvline(x=metrics.best_epoch, color='g', linestyle='--', alpha=0.7)
        ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CE Loss')
    ax.set_title('Validation Loss')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, [a * 100 for a in metrics.epoch_train_acc], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Training Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    ax = axes[1, 1]
    if metrics.epoch_val_acc:
        val_epochs = range(1, len(metrics.epoch_val_acc) + 1)
        ax.plot(val_epochs, [a * 100 for a in metrics.epoch_val_acc], 'b-', linewidth=2)
        if metrics.best_epoch > 0 and metrics.best_epoch <= len(metrics.epoch_val_acc):
            ax.axvline(x=metrics.best_epoch, color='g', linestyle='--', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Validation Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'training_curves_epoch_{epoch}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    return plot_path


def plot_all_metrics(metrics, output_dir, epoch):
    """Generate all plots and return paths."""
    paths = {}
    try:
        paths['training_curves'] = plot_training_curves(metrics, output_dir, epoch)
    except Exception as e:
        print(f"  Warning: Could not plot training curves: {e}")
    return paths


def save_metrics(metrics, output_dir, epoch):
    """Save metrics to pickle file for later analysis."""
    metrics_dict = {
        'epoch_train_loss': metrics.epoch_train_loss,
        'epoch_train_ce_loss': metrics.epoch_train_ce_loss,
        'epoch_train_wd_loss': metrics.epoch_train_wd_loss,
        'epoch_train_acc': metrics.epoch_train_acc,
        'epoch_val_loss': metrics.epoch_val_loss,
        'epoch_val_acc': metrics.epoch_val_acc,
        'best_val_loss': metrics.best_val_loss,
        'best_epoch': metrics.best_epoch,
        'position_correct': metrics.position_correct,
        'position_total': metrics.position_total,
        'confusion_matrix': metrics.confusion_matrix,
    }

    metrics_path = os.path.join(output_dir, f'metrics_epoch_{epoch}.pkl')
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics_dict, f)
    return metrics_path


def log_tensorboard_images(writer, metrics, step):
    """Log images to TensorBoard (stub for optional TensorBoard support)."""
    pass  # TensorBoard logging is optional
