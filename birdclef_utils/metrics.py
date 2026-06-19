"""
Metrics and evaluation utilities for BirdCLEF PyTorch models.

Functions for:
- Training history summarization
- Threshold sweeps
- Contest-style macro ROC-AUC computation
- Run metrics logging to CSV
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime, timezone
from sklearn.metrics import roc_auc_score


def summarize_training_history_pytorch(hist_dict, verbose=True):
    """
    Post-training metric summary for PyTorch history dict.
    
    Args:
        hist_dict: Dictionary with keys 'epoch_number', 'train_loss'/'loss', 'val_loss', 'val_auc'/'val_contest_auc'
        verbose: Whether to print summary
    
    Returns:
        Dictionary with summary statistics
    """
    if not hist_dict or 'epoch_number' not in hist_dict or len(hist_dict['epoch_number']) == 0:
        if verbose:
            print("No training history to summarize.")
        return {}
    
    epochs_ran = len(hist_dict['epoch_number'])
    
    # Support both old and new naming conventions
    train_loss_key = 'train_loss' if 'train_loss' in hist_dict else 'loss'
    val_auc_key = 'val_auc' if 'val_auc' in hist_dict else 'val_contest_auc'
    
    summary = {
        'epochs_ran': epochs_ran,
        'final_loss': hist_dict[train_loss_key][-1] if hist_dict[train_loss_key] else None,
        'final_val_loss': hist_dict['val_loss'][-1] if hist_dict['val_loss'] else None,
    }
    
    # Find best val_contest_auc
    if val_auc_key in hist_dict:
        val_aucs = [x for x in hist_dict[val_auc_key] if not np.isnan(x)]
        if val_aucs:
            best_idx = np.nanargmax(hist_dict[val_auc_key])
            summary['best_val_contest_auc'] = hist_dict[val_auc_key][best_idx]
            summary['best_epoch'] = hist_dict['epoch_number'][best_idx]
            summary['final_val_contest_auc'] = hist_dict[val_auc_key][-1]
    
    if verbose:
        print("="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Epochs run: {summary['epochs_ran']}")
        print(f"Final train loss: {summary['final_loss']:.4f}")
        print(f"Final val loss: {summary['final_val_loss']:.4f}")
        if 'best_val_contest_auc' in summary and not np.isnan(summary['best_val_contest_auc']):
            print(f"Best val contest AUC: {summary['best_val_contest_auc']:.6f} at epoch {summary['best_epoch']}")
            if not np.isnan(summary['final_val_contest_auc']):
                print(f"Final val contest AUC: {summary['final_val_contest_auc']:.6f}")
        print("="*60)
    
    return summary


def run_threshold_sweep_pytorch(
    model,
    val_loader,
    device='cpu',
    threshold_start=0.10,
    threshold_stop=0.95,
    threshold_step=0.10,
    max_samples=None,
    verbose=True
):
    """
    Run threshold sweep for PyTorch model to find optimal classification threshold.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        device: Device to run on ('cpu', 'cuda', 'mps')
        threshold_start: Start of threshold range
        threshold_stop: End of threshold range
        threshold_step: Step size for threshold sweep
        max_samples: Maximum number of samples to evaluate (None = all)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with sweep results and best threshold
    """
    model.eval()
    all_preds = []
    all_targets = []
    sample_count = 0
    
    if verbose:
        print("Collecting predictions for threshold sweep...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_samples and sample_count >= max_samples:
                break

            # Handle collate variations: (specs, targets), (specs, targets, ...), or dicts
            if batch is None:
                continue

            if isinstance(batch, dict):
                # try common keys first
                specs = batch.get('specs') or batch.get('inputs') or batch.get('audio')
                targets = batch.get('targets') or batch.get('labels')
                if specs is None or targets is None:
                    vals = list(batch.values())
                    if len(vals) >= 2:
                        specs, targets = vals[0], vals[1]
                    else:
                        raise ValueError("val_loader returned dict with fewer than 2 values")
                extra_info = {k: v for k, v in batch.items() if k not in ('specs', 'inputs', 'audio', 'targets', 'labels')}
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    specs, targets = batch[0], batch[1]
                    extra_info = batch[2:]
                else:
                    raise ValueError("val_loader must yield (specs, targets, ...) tuples")
            else:
                raise ValueError(f"Unexpected batch type from val_loader: {type(batch)}")

            # Warn if extra fields are present (windowed loader returns rec_idxs, starts)
            if extra_info:
                if verbose:
                    print(f"Warning: val_loader batch contains extra fields; using first two elements as (specs, targets). Extra: {type(extra_info)}")

            # Ensure tensors
            if not isinstance(specs, torch.Tensor):
                specs = torch.from_numpy(np.array(specs)).float()
            if not isinstance(targets, torch.Tensor):
                targets = torch.from_numpy(np.array(targets)).float()

            specs = specs.to(device)
            outputs = model(specs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            sample_count += specs.size(0)

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_count} samples...")
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    if verbose:
        print(f"Running threshold sweep on {len(y_true)} samples...")
    
    # Run threshold sweep
    thresholds = np.arange(threshold_start, threshold_stop + threshold_step/2, threshold_step)
    results = []
    
    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(int)
        
        # Compute precision, recall, F1
        true_pos = (y_pred_binary * y_true).sum()
        pred_pos = y_pred_binary.sum()
        actual_pos = y_true.sum()
        
        precision = true_pos / pred_pos if pred_pos > 0 else 0.0
        recall = true_pos / actual_pos if actual_pos > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
    
    results_arr = np.array([[r['threshold'], r['precision'], r['recall'], r['f1']] for r in results])
    
    # Find best F1
    best_idx = np.argmax(results_arr[:, 3])
    best_result = results[best_idx]
    
    sweep_summary = {
        'results_arr': results_arr,
        'best_t': best_result['threshold'],
        'best_p': best_result['precision'],
        'best_r': best_result['recall'],
        'best_f1': best_result['f1'],
        'eval_samples': len(y_true),
    }
    
    if verbose:
        print(f"\nThreshold sweep results:")
        print(f"  Best F1: {best_result['f1']:.4f} at threshold {best_result['threshold']:.2f}")
        print(f"  Precision: {best_result['precision']:.4f}, Recall: {best_result['recall']:.4f}")
    
    return sweep_summary


def append_run_metrics_csv_pytorch(
    hist_dict,
    run_context=None,
    threshold_summary=None,
    run_log_path='results/run_comparison.csv',
    verbose=True
):
    """
    Append run metrics to CSV for PyTorch training.
    
    Args:
        hist_dict: Training history dictionary
        run_context: Dictionary with run metadata
        threshold_summary: Dictionary with threshold sweep results
        run_log_path: Path to CSV file for logging
        verbose: Whether to print summary
    
    Returns:
        DataFrame with all run metrics
    """
    run_context = {} if run_context is None else dict(run_context)
    threshold_summary = {} if threshold_summary is None else dict(threshold_summary)
    
    output_path = Path(run_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not hist_dict or 'epoch_number' not in hist_dict or len(hist_dict['epoch_number']) == 0:
        raise ValueError("Training history is empty, cannot append metrics.")
    
    # Support both old and new naming conventions
    train_loss_key = 'train_loss' if 'train_loss' in hist_dict else 'loss'
    val_auc_key = 'val_auc' if 'val_auc' in hist_dict else 'val_contest_auc'
    
    epochs_ran = len(hist_dict['epoch_number'])
    epoch_start = hist_dict['epoch_number'][0]
    epoch_end = hist_dict['epoch_number'][-1]
    
    # Find best epoch by val_contest_auc
    val_aucs = [x if not np.isnan(x) else -np.inf for x in hist_dict[val_auc_key]]
    best_epoch_idx = int(np.argmax(val_aucs)) if val_aucs else 0
    best_epoch_value = hist_dict['epoch_number'][best_epoch_idx]
    
    row = {
        'run_id': run_context.get('run_id'),
        'model_name': hist_dict.get('model_name', run_context.get('model_type')),
        'train_subset_rows': run_context.get('train_subset_rows'),
        'val_subset_rows': run_context.get('val_subset_rows'),
        'batch_size': run_context.get('batch_size'),
        'epochs_ran': epochs_ran,
        'best_epoch': best_epoch_value,
        'best_val_contest_auc': float(hist_dict[val_auc_key][best_epoch_idx]) if val_aucs and val_aucs[best_epoch_idx] > -np.inf else None,
        'last_val_contest_auc': float(hist_dict[val_auc_key][-1]) if hist_dict[val_auc_key] and not np.isnan(hist_dict[val_auc_key][-1]) else None,
        'best_val_loss': float(hist_dict['val_loss'][best_epoch_idx]) if hist_dict['val_loss'] else None,
        'last_val_loss': float(hist_dict['val_loss'][-1]) if hist_dict['val_loss'] else None,
        'best_val_precision_at_0_2': threshold_summary.get('best_p'),
        'best_val_recall_at_0_2': threshold_summary.get('best_r'),        
        'epoch_start': epoch_start,
        'epoch_end': epoch_end,
        'run_tag': run_context.get('run_tag', ''),
        'timestamp_utc': hist_dict.get('timestamp_utc', datetime.now(timezone.utc).isoformat(timespec='seconds')),
        'training_time_seconds': hist_dict.get('training_time_seconds'),
        'checkpoint_best_path': run_context.get('checkpoint_best_path'),
        'model_type': run_context.get('model_type'),
        'quick_run': run_context.get('quick_run'),
        'train_rows_full_split': run_context.get('train_rows_full_split'),
        'val_rows_full_split': run_context.get('val_rows_full_split'),
        'num_classes': run_context.get('num_classes'),
        'num_workers': run_context.get('num_workers'),
        'use_spectrogram_cache': run_context.get('use_spectrogram_cache'),
        'feature_bins': run_context.get('feature_bins'),
        'use_log_mel': run_context.get('use_log_mel'),
        'last_loss': float(hist_dict[train_loss_key][-1]) if hist_dict[train_loss_key] else None,
        'threshold_best': threshold_summary.get('best_t'),
        'threshold_f1': threshold_summary.get('best_f1'),
        'threshold_eval_samples': threshold_summary.get('eval_samples'),
    }
    
    new_row_df = pd.DataFrame([row])
    
    if output_path.exists():
        run_log_df = pd.read_csv(output_path)
        run_log_df = pd.concat([run_log_df, new_row_df], ignore_index=True)
    else:
        run_log_df = new_row_df
    
    run_log_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Saved run metrics to: {output_path}")
        print(run_log_df.tail(3))
    
    return run_log_df


def collect_predictions_pytorch(model, val_loader, device='cpu', max_samples=None, verbose=True):
    """
    Collect predictions from a PyTorch model on validation data.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        device: Device to run on
        max_samples: Maximum number of samples to collect (None = all)
        verbose: Whether to print progress
    
    Returns:
        Tuple of (y_true, y_pred) as numpy arrays
    """
    model.eval()
    all_preds = []
    all_targets = []
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if max_samples and sample_count >= max_samples:
                break

            if batch is None:
                continue

            if isinstance(batch, dict):
                specs = batch.get('specs') or batch.get('inputs') or batch.get('audio')
                targets = batch.get('targets') or batch.get('labels')
                if specs is None or targets is None:
                    vals = list(batch.values())
                    if len(vals) >= 2:
                        specs, targets = vals[0], vals[1]
                    else:
                        raise ValueError("val_loader returned dict with fewer than 2 values")
            elif isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    specs, targets = batch[0], batch[1]
                else:
                    raise ValueError("val_loader must yield (specs, targets, ...) tuples")
            else:
                raise ValueError(f"Unexpected batch type from val_loader: {type(batch)}")

            if not isinstance(specs, torch.Tensor):
                specs = torch.from_numpy(np.array(specs)).float()
            if not isinstance(targets, torch.Tensor):
                targets = torch.from_numpy(np.array(targets)).float()

            specs = specs.to(device)
            outputs = model(specs)

            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            sample_count += specs.size(0)

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Processed {sample_count} samples...")
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    return y_true, y_pred


def contest_macro_roc_auc_pytorch(y_true, y_pred):
    """
    Compute contest-style macro ROC-AUC.
    
    Skips classes with no positive examples or all positive examples.
    
    Args:
        y_true: Ground truth binary labels (samples, classes)
        y_pred: Predicted probabilities (samples, classes)
    
    Returns:
        Dictionary with macro_roc_auc and class statistics
    """
    valid_class_indices = []
    skipped_no_positive = 0
    skipped_all_positive = 0
    
    for class_idx in range(y_true.shape[1]):
        y_col = y_true[:, class_idx]
        positives = int(np.sum(y_col))
        if positives == 0:
            skipped_no_positive += 1
            continue
        if positives == len(y_col):
            skipped_all_positive += 1
            continue
        valid_class_indices.append(class_idx)
    
    if not valid_class_indices:
        raise ValueError('No classes were eligible for contest-style ROC-AUC scoring.')
    
    macro_roc_auc = float(roc_auc_score(
        y_true[:, valid_class_indices],
        y_pred[:, valid_class_indices],
        average='macro',
    ))
    
    return {
        'macro_roc_auc': macro_roc_auc,
        'classes_scored': int(len(valid_class_indices)),
        'classes_skipped_no_positive': int(skipped_no_positive),
        'classes_skipped_all_positive': int(skipped_all_positive),
    }

def plot_training_curves_and_confusion(training_curves, model, dataloader, label_to_idx, idx_to_label, device='cpu',
                         phases=['train', 'val', 'test'],
                         metrics=['loss','acc', 'auc']):
    import math
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    
    epochs = list(range(len(training_curves['train_loss'])))

    num_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil((num_metrics + 1) / ncols)  # +1 for confusion matrix
    fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axs = axs.flatten() if (num_metrics + 1) > 1 else [axs]

    # Plot training curves
    for i, metric in enumerate(metrics):
        ax = axs[i]
        ax.set_title(f'Training curves - {metric}')
        for phase in phases:
            key = phase+'_'+metric
            if key in training_curves:
                ax.plot(epochs, training_curves[key], label=phase)
        ax.set_xlabel('epoch')
        ax.legend()

    # Plot confusion matrix in the last subplot
    ax_cm = axs[num_metrics]
    model.eval()
    all_preds = []
    all_actual = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_actual.extend(labels.cpu().numpy())
    all_class_indices = list(idx_to_label.keys())
    cm = confusion_matrix(all_actual, all_preds, labels=all_class_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[idx_to_label[i] for i in all_class_indices])
    disp.plot(ax=ax_cm, cmap='Blues', xticks_rotation=45)
    ax_cm.set_title('Confusion Matrix')
    
    # Hide any unused subplots
    for j in range(num_metrics + 1, len(axs)):
        fig.delaxes(axs[j])
    plt.tight_layout()
    
    plt.show()  # Do not show here, return fig instead
    return fig
