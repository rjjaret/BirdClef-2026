# data_processing.py --- IGNORE ---
# Data Synthesis and Augmentation Functions
import torch
import numpy as np
import random

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def spec_augment(spectrogram, time_mask_param=20, freq_mask_param=8, num_time_masks=1, num_freq_masks=1):
    if isinstance(spectrogram, np.ndarray):
        spec = spectrogram.copy()
    else:
        spec = spectrogram.clone()
    if spec.ndim == 2:
        spec = spec.unsqueeze(0) if isinstance(spec, torch.Tensor) else spec[None, ...]
    batch, n_mels, time_steps = spec.shape
    for b in range(batch):
        # Time masking
        for _ in range(num_time_masks):
            t = np.random.randint(0, time_mask_param + 1)
            t0 = np.random.randint(0, max(1, time_steps - t + 1))
            if isinstance(spec, torch.Tensor):
                spec[b, :, t0:t0 + t] = 0
            else:
                spec[b, :, t0:t0 + t] = 0
        # Frequency masking
        for _ in range(num_freq_masks):
            f = np.random.randint(0, freq_mask_param + 1)
            f0 = np.random.randint(0, max(1, n_mels - f + 1))
            if isinstance(spec, torch.Tensor):
                spec[b, f0:f0 + f, :] = 0
            else:
                spec[b, f0:f0 + f, :] = 0
    if spectrogram.ndim == 2:
        return spec[0]
    return spec

# Example: records = list of (spectrogram, label) tuples
# label_to_idx = {'loneliness': 0, 'discomfort': 1, 'hunger': 2}
from collections import defaultdict

def balance_by_removal(data_tensors, allowed_cry_types, data_tensor_index):

    from collections import Counter
    cry_type_counts = Counter(x[data_tensor_index.CRY_TYPE] for x in data_tensors)
    print("Cry type counts before balancing:", cry_type_counts)
    min_count = min(cry_type_counts[cry_type] for cry_type in allowed_cry_types)
    balanced_data_tensors = []
    for cry_type in allowed_cry_types:
        cry_type_data = [x for x in data_tensors if x[data_tensor_index.CRY_TYPE] == cry_type]
        balanced_data_tensors.extend(cry_type_data[:min_count])
    cry_type_counts = Counter(x[data_tensor_index.CRY_TYPE] for x in balanced_data_tensors)
    print("Cry type counts after balancing:", cry_type_counts)
    return balanced_data_tensors

def balance_by_spec_augmentation(records, label_to_idx, augment_fn, seed=42):
    random.seed(seed)
    from collections import defaultdict

    # # Defensive: check record structure
    # for i, record in enumerate(records[:5]):
    #     if not isinstance(record, (tuple, list)):
    #         print(f"Warning: record {i} is not a tuple or list: {record}")
    #     else:
    #         print(f"Sample record {i}: type={type(record)}, len={len(record)}, record={record}")

    print("Type of records:", type(records))

    # Group records by label, preserving full record structure
    class_records_by_label = defaultdict(list)
    for record in records:
        # # Defensive: ensure record has at least 2 elements
        # if not isinstance(record, (tuple, list)) or len(record) < 2:
        #     print(f"Skipping malformed record: {record}")
        #     continue
        class_label = record[data_tensor_index.CRY_TYPE]
        class_records_by_label[class_label].append(record)
    
    print("Class counts before balancing:")
    for label, v in class_records_by_label.items():
        print(label, len(v))    
    
    # Find max class size
    max_count = max(len(v) for v in class_records_by_label.values())
    balanced_records = []

    for label, class_records in class_records_by_label.items():
        n_to_add = max_count - len(class_records)
        balanced_records.extend(class_records)
        for _ in range(n_to_add):
            orig_record = random.choice(class_records)
            aug_spec = augment_fn(orig_record[0])
            aug_record = (aug_spec,) + orig_record[1:]
            balanced_records.append(aug_record)
    
    from collections import Counter
    print("Class counts after balancing:")
    label_counts = Counter(record[data_tensor_index.CRY_TYPE] for record in balanced_records)
    print("Class counts after balancing:", dict(label_counts))

    random.shuffle(balanced_records)
    return balanced_records

def balance_by_mixup_augmentation(records, label_to_idx, alpha=0.2, seed=42):
    import random
    import torch
    import numpy as np
    from collections import defaultdict
    random.seed(seed)
    np.random.seed(seed)

    # Group records by label
    class_records_by_label = defaultdict(list)
    for record in records:
        class_label = record[data_tensor_index.CRY_TYPE]
        class_records_by_label[class_label].append(record)

    # Find max class size
    max_count = max(len(v) for v in class_records_by_label.values())
    balanced_records = []

    for label, class_records in class_records_by_label.items():
        n_to_add = max_count - len(class_records)
        balanced_records.extend(class_records)
        for _ in range(n_to_add):
            rec1, rec2 = random.sample(class_records, 2) if len(class_records) > 1 else (class_records[0], class_records[0])
            x1, y1 = rec1[0], label_to_idx[rec1[data_tensor_index.CRY_TYPE]]
            x2, y2 = rec2[0], label_to_idx[rec2[data_tensor_index.CRY_TYPE]]
            # Convert to tensors if not already
            if not torch.is_tensor(x1):
                x1 = torch.tensor(x1)
            if not torch.is_tensor(x2):
                x2 = torch.tensor(x2)
            y1 = torch.tensor(y1)
            y2 = torch.tensor(y2)
            mixed_x, y_a, y_b, lam = mixup_data(x1.unsqueeze(0), y1.unsqueeze(0), alpha)
            # Remove batch dimension
            mixed_x = mixed_x.squeeze(0)
            # Use the original label for grouping, but you may want to store (mixed_x, y_a, y_b, lam) for training
            # Here, we keep the structure (mixed_x, label, ...rest)
            aug_record = (mixed_x, rec1[data_tensor_index.CRY_TYPE]) + rec1[2:] if len(rec1) > 2 else (mixed_x, rec1[data_tensor_index.CRY_TYPE])
            balanced_records.append(aug_record)

    random.shuffle(balanced_records)
    return balanced_records



# Function to look at the data
def analyze_data_tensors(data, h, label_set, include_plots=False):
    import matplotlib.pyplot as plt
    import numpy as np
    count_per_label = dict(zip(label_set, [0] * len(label_set)))
    ages = dict(zip(age_dict.keys(), [0] * len(age_dict.keys())))
    genders = {"m": 0, "f": 0}
    lengths = []

    for spec in data:
        label = spec[data_tensor_index.CRY_TYPE]
        age = spec[data_tensor_index.AGE]
        gender = spec[data_tensor_index.GENDER]

        # Defensive: skip if label, age, or gender is None
        if label is None or age is None or gender is None:
            continue
        if label not in count_per_label:
            continue
        if age not in ages:
            continue
        if gender not in genders:
            continue
        count_per_label[label] += 1
        ages[age] += 1
        genders[gender] += 1
        # Length in samples
        if hasattr(spec[data_tensor_index.WAV], 'shape'):
            lengths.append(spec[data_tensor_index.WAV].shape[1])
        else:
            lengths.append(len(spec[data_tensor_index.WAV][0]))

    # Print summary
    for cry_type in label_set:
        print(f"Label: {cry_type}: {count_per_label[cry_type]}")
    print()
    for key in ages.keys():
        print(f"Age: {age_dict[key]}: {ages[key]}")
    print()
    for key in genders.keys():
        print(f"Gender: {key}: {genders[key]}")
    print()

    if lengths:
        average = np.mean(lengths) / h["sample_rate"]  # avg number of seconds of file
        high = np.max(lengths) / h["sample_rate"]
        low = np.min(lengths) / h["sample_rate"]
        count_above_1_sec = sum(1 for l in lengths if l / h["sample_rate"] > 2)

        # print(f"Number of audio files above  seconds: {count_above_1_sec}")
        print("Average Length: ", average)
        print("Lowest Length: ", low)
        print("Highest Length: ", high)
        print()

        if include_plots:
            # --- Plots ---
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Histogram of audio lengths (in seconds)
            axs[0, 0].hist([l / h["sample_rate"] for l in lengths], bins=30, color='skyblue', edgecolor='black')
            axs[0, 0].set_title('Histogram of Audio Lengths (seconds)')
            axs[0, 0].set_xlabel('Length (seconds)')
            axs[0, 0].set_ylabel('Count')

            # Bar chart for cry types
            cry_type_names = [cry_type_dict.get(lab, str(lab)) for lab in label_set]
            axs[0, 1].bar(cry_type_names, list(count_per_label.values()), color='salmon', edgecolor='black')
            axs[0, 1].set_title('Number of Records per Cry Type')
            axs[0, 1].set_xlabel('Cry Type')
            axs[0, 1].set_ylabel('Count')
            axs[0, 1].tick_params(axis='x', rotation=45)

            # Bar chart for ages
            age_names = [age_dict.get(k, str(k)) for k in ages.keys()]
            axs[1, 0].bar(age_names, list(ages.values()), color='lightgreen', edgecolor='black')
            axs[1, 0].set_title('Number of Records per Age Group')
            axs[1, 0].set_xlabel('Age Group')
            axs[1, 0].set_ylabel('Count')
            axs[1, 0].tick_params(axis='x', rotation=45)

            # Bar chart for genders
            gender_names = ['Male', 'Female']
            gender_counts = [genders['m'], genders['f']]
            axs[1, 1].bar(gender_names, gender_counts, color='plum', edgecolor='black')
            axs[1, 1].set_title('Number of Records per Sex')
            axs[1, 1].set_xlabel('Sex')
            axs[1, 1].set_ylabel('Count')

            plt.tight_layout()
            plt.show()
            return plt
    else:
        print("No valid audio lengths found for plotting.")


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

def append_run_metrics_csv(
    history,
    run_log_path="run_comparison.csv",
    fig=None,
    save_model= False
):
    import pandas as pd
    from pathlib import Path
    
    output_path = Path(run_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not history:
        raise ValueError(
            "Training history is empty."
        )

    run_id = 1
    if output_path.exists():
        run_log_df = pd.read_csv(output_path)
        if 'run_id' in run_log_df.columns and not run_log_df.empty:
            run_id = int(run_log_df['run_id'].max()) + 1
        else:
            run_id = 1
    else:
        run_log_df = None
        run_id = 1
    
    run_id = int(run_id)
    history["run_id"] = run_id
    new_row_df = pd.DataFrame([history])

    # Desired column order
    desired_cols = [
        'run_id', 'model_name', 'dataset_name', 'num_epochs', 'learning_rate', 'batch_size', 'dataset_size', 'dataset_phase_sizes',
        'best_auc', 'best_epoch', 'best_acc', 'best_acc_epoch', 'best_loss', 'best_loss_epoch', 'dropout', 'device', 'training_duration_seconds'
    ]
    # Add any other columns at the end
    all_cols = list(new_row_df.columns)
    extra_cols = [col for col in all_cols if col not in desired_cols]
    final_cols = desired_cols + extra_cols
    new_row_df = new_row_df.reindex(columns=final_cols)

    if run_log_df is not None:
        run_log_df = pd.concat([run_log_df, new_row_df], ignore_index=True)

        # Ensure column order
        all_cols = list(run_log_df.columns)
        extra_cols = [col for col in all_cols if col not in desired_cols]
        final_cols = desired_cols + extra_cols
        run_log_df = run_log_df.reindex(columns=final_cols)
    else:
        run_log_df = new_row_df

    run_log_df.to_csv(output_path, index=False)

    print(f"Saved run metrics to: {output_path}")

    # --- Save training curves and confusion matrix as image ---
    model_name = history.get("model_name", "model")
    if fig is not None:
        fig_filename = f"train_conf_{model_name}_run-{run_id}.png"
        fig_path = output_path.parent / "figures" / fig_filename
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path)
        print(f"Saved training curves and confusion matrix to: {fig_path}")

    if save_model:
        model_path = Path('models/')
        model_path.mkdir(parents=True, exist_ok=True)
        model_filename = f"{model_name}_run-{run_id}.pt"
        model_save_path = model_path / model_filename
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved model state dict to: {model_save_path}")
    
    return run_log_df
