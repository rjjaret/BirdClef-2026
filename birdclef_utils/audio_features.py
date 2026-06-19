import numpy as np
import librosa
import tensorflow as tf
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass
class SweepDataRefs:
    """Runtime references required for threshold sweep evaluation."""

    val_file_paths: Any
    val_targets: Any
    load_and_process_train: Callable
    load_and_process_eval: Callable
    compute_spectrogram: Callable
    num_classes: int


@dataclass
class SweepConfig:
    """Threshold sweep knobs grouped into a single config object."""

    batch_size: int
    min_samples: int
    validation_eval_rows: int
    validation_num_crops: int

    quick_run: bool = False
    val_subset: int | None = None
    use_log_mel: bool = False
    n_mels: int = 128
    stft_fft_length: int = 1024

    fast_sweep_max_rows: int = 200
    fast_sweep_max_crops: int = 2
    full_sweep_max_rows: int | None = None

    fast_sweep_threshold_start: float = 0.10
    fast_sweep_threshold_stop: float = 0.95
    fast_sweep_threshold_step: float = 0.10

    full_sweep_threshold_start: float = 0.05
    full_sweep_threshold_stop: float = 0.95
    full_sweep_threshold_step: float = 0.05

    min_recall_floor: float = 0.10
    load_once_crops: bool = True
    inference_batch_size: int | None = None  # None → use batch_size; increase for faster sweeps
    sample_rate: int = 32000
    random_crop_samples: int | None = None


@dataclass
class RunContextConfig:
    """Run metadata stored with each training row in the comparison CSV."""

    run_id: str | None = None
    run_tag: str | None = None

    train_subset_rows: int | None = None
    val_subset_rows: int | None = None
    train_rows_full_split: int | None = None
    val_rows_full_split: int | None = None
    train_soundscape_rows: int | None = None
    use_train_soundscape_labels: bool | None = None

    batch_size: int | None = None
    steps_per_epoch: int | None = None
    validation_steps: int | None = None
    max_audio_seconds: float | None = None
    random_crop_seconds: float | None = None
    validation_num_crops: int | None = None
    validation_eval_rows: int | None = None
    min_rating: float | None = None
    min_samples: int | None = None
    num_classes: int | None = None
    hard_negative_enabled: bool | None = None
    hard_negative_db_threshold: float | None = None
    hard_negative_sample_prob: float | None = None

    use_log_mel: bool | None = None
    n_mels: int | None = None
    mel_fmin: float | None = None
    mel_fmax: float | None = None
    stft_frame_length: int | None = None
    stft_frame_step: int | None = None

    regularization_profile: str | None = None
    early_stopping_monitor: str | None = None
    checkpoint_best_path: str | None = None
    checkpoint_last_path: str | None = None


def build_run_context(config):
    """Convert a RunContextConfig instance to a dict for CSV logging."""
    if isinstance(config, RunContextConfig):
        return asdict(config)
    return dict(config)


def sweep_data_refs_from_globals(g):
    """Build SweepDataRefs from the notebook globals dict."""
    return SweepDataRefs(
        val_file_paths=g['val_file_paths'],
        val_targets=g['val_targets'],
        load_and_process_train=g['load_and_process_train'],
        load_and_process_eval=g['load_and_process_eval'],
        compute_spectrogram=g['compute_spectrogram'],
        num_classes=g['num_classes'],
    )


def sweep_config_from_globals(g):
    """Build SweepConfig from the notebook globals dict."""
    batch_size = g['BATCH_SIZE']
    return SweepConfig(
        batch_size=batch_size,
        min_samples=g['MIN_SAMPLES'],
        validation_eval_rows=g['VALIDATION_EVAL_ROWS'],
        validation_num_crops=g['VALIDATION_NUM_CROPS'],
        quick_run=g.get('QUICK_RUN', False),
        val_subset=g.get('VAL_SUBSET'),
        use_log_mel=g.get('USE_LOG_MEL', False),
        n_mels=g.get('N_MELS', 128),
        stft_fft_length=g.get('STFT_FFT_LENGTH', 1024),
        fast_sweep_max_rows=g.get('FAST_SWEEP_MAX_ROWS', 200),
        fast_sweep_max_crops=g.get('FAST_SWEEP_MAX_CROPS', 2),
        full_sweep_max_rows=g.get('FULL_SWEEP_MAX_ROWS'),
        fast_sweep_threshold_start=g.get('FAST_SWEEP_THRESHOLD_START', 0.10),
        fast_sweep_threshold_stop=g.get('FAST_SWEEP_THRESHOLD_STOP', 0.95),
        fast_sweep_threshold_step=g.get('FAST_SWEEP_THRESHOLD_STEP', 0.10),
        full_sweep_threshold_start=g.get('FULL_SWEEP_THRESHOLD_START', 0.05),
        full_sweep_threshold_stop=g.get('FULL_SWEEP_THRESHOLD_STOP', 0.95),
        full_sweep_threshold_step=g.get('FULL_SWEEP_THRESHOLD_STEP', 0.05),
        min_recall_floor=g.get('MIN_RECALL_FLOOR', 0.10),
        load_once_crops=g.get('LOAD_ONCE_CROPS', True),
        random_crop_samples=g.get('RANDOM_CROP_SAMPLES'),
        sample_rate=g.get('SAMPLE_RATE', 32000),
        inference_batch_size=batch_size * 4,
        crop_seed=g.get('CROP_SEED'),
        prediction_cache_dir=g.get('PREDICTION_CACHE_DIR'),
    )


def run_context_config_from_globals(g):
    """Build RunContextConfig from the notebook globals dict."""
    train_subset = g.get('TRAIN_SUBSET')
    val_subset = g.get('VAL_SUBSET')
    run_tag = (
        f"subset_{train_subset}_{val_subset}"
        if train_subset is not None and val_subset is not None
        else None
    )
    train_fps = g.get('train_file_paths')
    val_fps = g.get('val_file_paths')
    return RunContextConfig(
        run_id=g.get('RUN_ID'),
        run_tag=run_tag,
        train_subset_rows=int(train_subset) if train_subset is not None else None,
        val_subset_rows=int(val_subset) if val_subset is not None else None,
        train_rows_full_split=int(len(train_fps)) if train_fps is not None else None,
        val_rows_full_split=int(len(val_fps)) if val_fps is not None else None,
        train_soundscape_rows=(
            int(g['TRAIN_SOUNDSCAPE_ROWS'])
            if 'TRAIN_SOUNDSCAPE_ROWS' in g and g['TRAIN_SOUNDSCAPE_ROWS'] is not None
            else None
        ),
        use_train_soundscape_labels=(
            bool(g['USE_TRAIN_SOUNDSCAPE_LABELS'])
            if 'USE_TRAIN_SOUNDSCAPE_LABELS' in g
            else None
        ),
        batch_size=int(g['BATCH_SIZE']) if 'BATCH_SIZE' in g else None,
        steps_per_epoch=int(g['STEPS_PER_EPOCH']) if 'STEPS_PER_EPOCH' in g else None,
        validation_steps=int(g['VAL_STEPS']) if 'VAL_STEPS' in g else None,
        max_audio_seconds=g.get('MAX_AUDIO_SECONDS'),
        random_crop_seconds=g.get('RANDOM_CROP_SECONDS'),
        validation_num_crops=int(g['VALIDATION_NUM_CROPS']) if 'VALIDATION_NUM_CROPS' in g else None,
        validation_eval_rows=int(g['VALIDATION_EVAL_ROWS']) if 'VALIDATION_EVAL_ROWS' in g else None,
        min_rating=float(g['MIN_RATING']) if 'MIN_RATING' in g else None,
        min_samples=int(g['MIN_SAMPLES']) if 'MIN_SAMPLES' in g else None,
        num_classes=int(g['num_classes']) if 'num_classes' in g else None,
        hard_negative_enabled=bool(g['HARD_NEGATIVE_ENABLED']) if 'HARD_NEGATIVE_ENABLED' in g else None,
        hard_negative_db_threshold=float(g['HARD_NEGATIVE_DB_THRESHOLD']) if 'HARD_NEGATIVE_DB_THRESHOLD' in g else None,
        hard_negative_sample_prob=float(g['HARD_NEGATIVE_SAMPLE_PROB']) if 'HARD_NEGATIVE_SAMPLE_PROB' in g else None,
        use_log_mel=bool(g['USE_LOG_MEL']) if 'USE_LOG_MEL' in g else None,
        n_mels=int(g['N_MELS']) if 'N_MELS' in g else None,
        mel_fmin=float(g['MEL_FMIN']) if 'MEL_FMIN' in g else None,
        mel_fmax=float(g['MEL_FMAX']) if 'MEL_FMAX' in g else None,
        stft_frame_length=int(g['STFT_FRAME_LENGTH']) if 'STFT_FRAME_LENGTH' in g else None,
        stft_frame_step=int(g['STFT_FRAME_STEP']) if 'STFT_FRAME_STEP' in g else None,

        regularization_profile=g.get('REGULARIZATION_PROFILE', 'spatial_dropout0.2_lstm_dropout0.2_l2_1e-4_dense_dropout0.3'),
        early_stopping_monitor=g.get('EARLY_STOPPING_MONITOR', 'val_pr_auc'),
        checkpoint_best_path=g.get('CHECKPOINT_BEST_PATH', 'checkpoints/best_model.keras'),
        checkpoint_last_path=g.get('CHECKPOINT_LAST_PATH', 'checkpoints/last_epoch.keras'),
    )


@dataclass
class PostTrainRunner:
    """Convenience runner for summary, sweep, and run-log append."""

    data_refs: SweepDataRefs
    sweep_config: SweepConfig
    run_log_path: str = "results/run_comparison.csv"

    def run_all(self, model, hist, run_context=None, verbose=True):
        summary = summarize_training_history(hist=hist, verbose=verbose)
        sweep = run_threshold_sweep(
            model=model,
            data_refs=self.data_refs,
            config=self.sweep_config,
            verbose=verbose,
        )
        run_log_df = append_run_metrics_csv(
            hist=hist,
            context=run_context,
            threshold_summary=sweep,
            run_log_path=self.run_log_path,
            verbose=verbose,
        )
        return {
            "summary": summary,
            "sweep": sweep,
            "run_log_df": run_log_df,
        }


def build_audio_stats(descriptions_df, load_fn, sample_rate=32000):
    """Compute basic clip-length statistics for a descriptions DataFrame."""
    lengths = []

    for _, row in descriptions_df.iterrows():
        fp = row["folder"] + row["filename"] if str(row["folder"]).endswith("/") else row["folder"] + "/" + row["filename"]
        wav = load_fn(fp, sr=sample_rate, mono=True)
        lengths.append(len(wav))

    lengths_arr = np.asarray(lengths, dtype=np.int64)

    stats = {
        "mean_sec": float(lengths_arr.mean() / sample_rate),
        "min_sec": float(lengths_arr.min() / sample_rate),
        "max_sec": float(lengths_arr.max() / sample_rate),
        "over_10": int((lengths_arr > (10 * sample_rate)).sum()),
        "over_20": int((lengths_arr > (20 * sample_rate)).sum()),
        "over_40": int((lengths_arr > (40 * sample_rate)).sum()),
        "over_60": int((lengths_arr > (60 * sample_rate)).sum()),
        "over_120": int((lengths_arr > (120 * sample_rate)).sum()),
        "under_1": int((lengths_arr < sample_rate).sum()),
    }

    return stats


def print_audio_file_stats(descriptions_df, load_fn, sample_rate=32000):
    """Print basic clip-length statistics."""
    s = build_audio_stats(descriptions_df, load_fn=load_fn, sample_rate=sample_rate)
    print(f"Mean: {s['mean_sec']}\\nMin: {s['min_sec']}\\nMax: {s['max_sec']}")
    print(f"Number over 10 seconds: {s['over_10']}")
    print(f"Number over 20 seconds: {s['over_20']}")
    print(f"Number over 40 seconds: {s['over_40']}")
    print(f"Number over 60 seconds: {s['over_60']}")
    print(f"Number over 120 seconds: {s['over_120']}")
    print(f"Number under 1 second: {s['under_1']}")


def load_audio_tensor_from_ogg(filename, sr=32000, mono=True):
    """Load an OGG file and return a float32 Tensor."""
    y, _ = librosa.load(filename, sr=sr, mono=mono)
    return tf.convert_to_tensor(y, dtype=tf.float32)


def build_audio_processors(config):
    """Build train/eval waveform loaders and spectrogram function from config."""
    sample_rate = int(config.get("sample_rate", 32000))
    max_audio_seconds = config.get("max_audio_seconds")
    random_crop_seconds = config.get("random_crop_seconds")

    use_log_mel = bool(config.get("use_log_mel", True))
    stft_frame_length = int(config.get("stft_frame_length", 1024))
    stft_frame_step = int(config.get("stft_frame_step", 320))
    stft_fft_length = int(config.get("stft_fft_length", 1024))

    n_mels = int(config.get("n_mels", 128))
    mel_fmin = float(config.get("mel_fmin", 50.0))
    mel_fmax = float(config.get("mel_fmax", 14000.0))

    max_audio_samples = None if max_audio_seconds is None else int(sample_rate * max_audio_seconds)
    random_crop_samples = None if random_crop_seconds is None else int(sample_rate * random_crop_seconds)
    feature_bins = n_mels if use_log_mel else (stft_fft_length // 2 + 1)

    _SEG_TOKEN = "__SEG__"

    def _load_waveform_for_path(path_str):
        """Load full waveform or an encoded segment path into float32 samples."""
        if _SEG_TOKEN in path_str:
            base_fp, seg_meta = path_str.split(_SEG_TOKEN, 1)
            try:
                start_s, end_s = seg_meta.split("__", 1)
                seg_start = float(start_s)
                seg_end = float(end_s)
                seg_duration = max(0.0, seg_end - seg_start)
                y, _ = librosa.load(base_fp, sr=sample_rate, mono=True, offset=seg_start, duration=seg_duration)
            except ValueError:
                # Fallback for malformed encoded paths.
                y, _ = librosa.load(base_fp, sr=sample_rate, mono=True)
        else:
            y, _ = librosa.load(path_str, sr=sample_rate, mono=True)
        return y

    def _load_waveform(fp, random_crop):
        try:
            y = _load_waveform_for_path(fp)
        except FileNotFoundError:
            # Missing files are treated as empty clips and filtered out downstream.
            return np.zeros((0,), dtype="float32"), 0

        if random_crop_samples is not None and len(y) > random_crop_samples:
            if random_crop:
                start = np.random.randint(0, len(y) - random_crop_samples + 1)
            else:
                start = (len(y) - random_crop_samples) // 2
            y = y[start : start + random_crop_samples]
        elif max_audio_samples is not None and len(y) > max_audio_samples:
            if random_crop:
                start = np.random.randint(0, len(y) - max_audio_samples + 1)
            else:
                start = 0
            y = y[start : start + max_audio_samples]

        return y.astype("float32"), len(y)

    def _load_and_process(file_path, target, random_crop):
        def _load(fp):
            fp = fp.numpy().decode()
            y, length = _load_waveform(fp, random_crop=random_crop)
            return y, length

        wav, length = tf.py_function(_load, [file_path], [tf.float32, tf.int64])
        wav.set_shape([None])
        return wav, target, length

    def load_and_process_train(file_path, target):
        return _load_and_process(file_path, target, random_crop=True)

    def load_and_process_eval(file_path, target):
        return _load_and_process(file_path, target, random_crop=False)

    def compute_spectrogram(wav, target, length):
        stft = tf.signal.stft(
            wav,
            frame_length=stft_frame_length,
            frame_step=stft_frame_step,
            fft_length=stft_fft_length,
        )
        magnitude = tf.abs(stft)

        if use_log_mel:
            power_spec = tf.square(magnitude)
            mel_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=n_mels,
                num_spectrogram_bins=(stft_fft_length // 2) + 1,
                sample_rate=sample_rate,
                lower_edge_hertz=mel_fmin,
                upper_edge_hertz=mel_fmax,
            )
            mel_power = tf.matmul(power_spec, mel_matrix)
            features = tf.math.log(mel_power + 1e-6)
        else:
            features = magnitude

        return features, target

    return {
        "max_audio_samples": max_audio_samples,
        "random_crop_samples": random_crop_samples,
        "feature_bins": feature_bins,
        "load_and_process_train": load_and_process_train,
        "load_and_process_eval": load_and_process_eval,
        "compute_spectrogram": compute_spectrogram,
    }


def summarize_training_history(hist, verbose=True):
    """Print and return a compact metric summary for a Keras History object."""
    h = pd.DataFrame(hist.history)

    if h.empty:
        raise ValueError(
            "Training history is empty (0 epochs recorded). "
            "This usually means training did not run any complete epochs. "
            "Check steps_per_epoch/validation_steps and dataset filters."
        )

    last = h.iloc[-1]
    metric_order = [
        "loss",
        "bin_acc@0.2",
        "precision@0.2",
        "recall@0.2",
        "pr_auc",
        "val_loss",
        "val_bin_acc@0.2",
        "val_precision@0.2",
        "val_recall@0.2",
        "val_pr_auc",
        "val_contest_auc",
    ]

    if "val_contest_auc" in h.columns:
        best_epoch_idx = int(h["val_contest_auc"].idxmax())
    elif "val_pr_auc" in h.columns:
        best_epoch_idx = int(h["val_pr_auc"].idxmax())
    else:
        best_epoch_idx = int(h["loss"].idxmin())

    epoch_numbers = h["epoch_number"] if "epoch_number" in h.columns else None
    best_epoch_value = int(epoch_numbers.iloc[best_epoch_idx]) if epoch_numbers is not None else (best_epoch_idx + 1)
    epoch_start = int(epoch_numbers.iloc[0]) if epoch_numbers is not None else 1
    epoch_end = int(epoch_numbers.iloc[-1]) if epoch_numbers is not None else int(len(h))

    if verbose:
        print("Last epoch metrics:")
        for k in metric_order:
            if k in h.columns:
                print(f"  {k}: {last[k]:.6f}")

        print(f"\nEpochs recorded: {epoch_start} to {epoch_end} ({len(h)} total)")
        print(f"Best epoch: {best_epoch_value}")
        if "val_contest_auc" in h.columns:
            print(f"  val_contest_auc: {h.loc[best_epoch_idx, 'val_contest_auc']:.6f}")
        elif "val_pr_auc" in h.columns:
            print(f"  val_pr_auc: {h.loc[best_epoch_idx, 'val_pr_auc']:.6f}")
        else:
            print(f"  loss: {h.loc[best_epoch_idx, 'loss']:.6f}")
        for k in ["val_precision@0.2", "val_recall@0.2", "val_loss"]:
            if k in h.columns:
                print(f"  {k}: {h.loc[best_epoch_idx, k]:.6f}")

    return {
        "history_df": h,
        "last": last,
        "best_epoch_idx": best_epoch_idx,
        "best_epoch": best_epoch_value,
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "epochs_ran": int(len(h)),
    }


def threshold_sweep_multicrop(
    model,
    val_file_paths,
    val_targets,
    load_and_process_train,
    load_and_process_eval,
    compute_spectrogram,
    batch_size,
    min_samples,
    num_classes,
    validation_eval_rows,
    validation_num_crops,
    quick_run=False,
    val_subset=None,
    use_log_mel=False,
    n_mels=128,
    stft_fft_length=1024,
    fast_sweep_max_rows=200,
    fast_sweep_max_crops=2,
    full_sweep_max_rows=None,
    fast_sweep_threshold_start=0.10,
    fast_sweep_threshold_stop=0.95,
    fast_sweep_threshold_step=0.10,
    full_sweep_threshold_start=0.05,
    full_sweep_threshold_stop=0.95,
    full_sweep_threshold_step=0.05,
    min_recall_floor=0.10,
    load_once_crops=False,
    inference_batch_size=None,
    sample_rate=32000,
    random_crop_samples=None,
    crop_seed=None,
    prediction_cache_dir=None,
    verbose=True,
):
    """Run threshold sweep using multi-crop validation prediction averaging."""
    fast_sweep = bool(quick_run)

    base_eval_rows = min(int(val_subset), int(validation_eval_rows)) if val_subset is not None else int(validation_eval_rows)
    base_num_crops = int(validation_num_crops)

    fast_max_rows = int(fast_sweep_max_rows)
    fast_max_crops = int(fast_sweep_max_crops)
    full_max_rows = base_eval_rows if full_sweep_max_rows is None else int(full_sweep_max_rows)

    if fast_sweep:
        eval_rows = min(base_eval_rows, fast_max_rows)
        num_val_crops = min(base_num_crops, fast_max_crops)
        thresholds = np.arange(
            float(fast_sweep_threshold_start),
            float(fast_sweep_threshold_stop),
            float(fast_sweep_threshold_step),
        )
    else:
        eval_rows = min(base_eval_rows, full_max_rows)
        num_val_crops = base_num_crops
        thresholds = np.arange(
            float(full_sweep_threshold_start),
            float(full_sweep_threshold_stop),
            float(full_sweep_threshold_step),
        )

    eval_feature_bins = int(n_mels) if bool(use_log_mel) else ((int(stft_fft_length) // 2) + 1)
    eval_batches = max(1, eval_rows // int(batch_size))
    _infer_bs = int(inference_batch_size) if inference_batch_size is not None else int(batch_size)

    if verbose:
        print(
            f"Threshold sweep mode: {'FAST' if fast_sweep else 'FULL'} | "
            f"rows={eval_rows}, crops={num_val_crops}, thresholds={len(thresholds)}"
        )
        print(
            f"Sweep feature bins: {eval_feature_bins} "
            f"({'log-mel' if use_log_mel else 'magnitude STFT'})"
        )
        _mode_str = "load-once" if (load_once_crops and random_crop_samples is not None) else "tf.data-per-crop"
        print(f"Inference mode: {_mode_str} | inference batch={_infer_bs}")

    eval_paths = val_file_paths[:eval_rows]
    eval_targets = val_targets[:eval_rows]

    _SEG_TOKEN = "__SEG__"

    def _load_eval_waveform(path_str):
        if _SEG_TOKEN in path_str:
            base_fp, seg_meta = path_str.split(_SEG_TOKEN, 1)
            try:
                start_s, end_s = seg_meta.split("__", 1)
                seg_start = float(start_s)
                seg_end = float(end_s)
                seg_duration = max(0.0, seg_end - seg_start)
                wav, _ = librosa.load(base_fp, sr=int(sample_rate), mono=True, offset=seg_start, duration=seg_duration)
            except ValueError:
                wav, _ = librosa.load(base_fp, sr=int(sample_rate), mono=True)
        else:
            wav, _ = librosa.load(path_str, sr=int(sample_rate), mono=True)
        return wav

    y_true = None
    crop_preds = []

    if load_once_crops and random_crop_samples is not None:
        # Fast path: load each waveform once, apply all crops in-memory.
        # Eliminates (num_val_crops - 1) × file I/O passes and uses a larger inference batch.
        _crop_len = int(random_crop_samples)
        _use_random = (num_val_crops > 1)
        _dummy_target = tf.zeros(int(num_classes), dtype=tf.float32)

        # Build cache path when seed + cache dir are set
        _cache_path = None
        if crop_seed is not None and prediction_cache_dir is not None:
            _feat_str = "log_mel" if bool(use_log_mel) else "stft"
            _cache_fname = f"sweep_{_feat_str}_{eval_rows}r_{num_val_crops}c_seed{crop_seed}.npz"
            _cache_path = Path(prediction_cache_dir) / _cache_fname

        if _cache_path is not None and _cache_path.exists():
            if verbose:
                print(f"Loading cached predictions from {_cache_path}")
            _cached = np.load(_cache_path)
            y_true = _cached["y_true"]
            y_pred = _cached["y_pred"]
        else:
            _rng = np.random.default_rng(crop_seed) if crop_seed is not None else np.random.default_rng()

            raw_waveforms, valid_targets_list = [], []
            skipped_missing = 0
            for path_val, target_val in zip(eval_paths, eval_targets):
                path_str = path_val.decode() if isinstance(path_val, bytes) else str(path_val)
                try:
                    wav = _load_eval_waveform(path_str)
                except FileNotFoundError:
                    skipped_missing += 1
                    continue
                if len(wav) > int(min_samples):
                    raw_waveforms.append(wav.astype(np.float32))
                    valid_targets_list.append(target_val)

            if skipped_missing > 0 and verbose:
                print(f"Skipped {skipped_missing} missing file(s) during threshold sweep.")

            y_true = np.array(valid_targets_list)

            for _ in range(num_val_crops):
                y_pred_all = []
                for b_start in range(0, len(raw_waveforms), _infer_bs):
                    b_wavs = raw_waveforms[b_start: b_start + _infer_bs]
                    batch_specs = []
                    for wav in b_wavs:
                        if len(wav) >= _crop_len:
                            s = _rng.integers(0, len(wav) - _crop_len + 1) if _use_random else (len(wav) - _crop_len) // 2
                            wav_crop = wav[s: s + _crop_len]
                        else:
                            wav_crop = np.zeros(_crop_len, dtype=np.float32)
                            wav_crop[: len(wav)] = wav
                        spec, _ = compute_spectrogram(tf.constant(wav_crop), _dummy_target, _crop_len)
                        batch_specs.append(spec.numpy())
                    batch_arr = np.stack(batch_specs, axis=0)
                    preds = model(batch_arr, training=False).numpy()
                    y_pred_all.append(preds)
                crop_preds.append(np.concatenate(y_pred_all, axis=0))

            y_pred = np.mean(np.stack(crop_preds, axis=0), axis=0)

            if _cache_path is not None:
                _cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(_cache_path, y_true=y_true, y_pred=y_pred)
                if verbose:
                    print(f"Saved prediction cache to {_cache_path}")
    else:
        # Standard path: rebuild tf.data pipeline per crop.
        for _ in range(num_val_crops):
            loader_fn = load_and_process_eval if num_val_crops == 1 else load_and_process_train

            val_eval_ds = (
                tf.data.Dataset.from_tensor_slices((eval_paths, eval_targets))
                .map(loader_fn, num_parallel_calls=tf.data.AUTOTUNE)
                .filter(lambda wav, target, length: length > min_samples)
                .map(lambda wav, target, length: compute_spectrogram(wav, target, length), num_parallel_calls=tf.data.AUTOTUNE)
                .padded_batch(int(batch_size), padded_shapes=([None, eval_feature_bins], [int(num_classes)]))
                .prefetch(tf.data.AUTOTUNE)
            )

            y_true_all, y_pred_all = [], []
            for x_batch, y_batch in val_eval_ds:
                preds = model(x_batch, training=False).numpy()
                y_true_all.append(y_batch.numpy())
                y_pred_all.append(preds)

            y_true_current = np.concatenate(y_true_all, axis=0)
            y_pred_current = np.concatenate(y_pred_all, axis=0)

            if y_true is None:
                y_true = y_true_current
            crop_preds.append(y_pred_current)

        y_pred = np.mean(np.stack(crop_preds, axis=0), axis=0)

    y_pred_t = y_pred[None, :, :]
    y_true_t = y_true[None, :, :]
    threshold_t = thresholds[:, None, None]
    y_bin = (y_pred_t >= threshold_t).astype(np.float32)

    tp = (y_bin * y_true_t).sum(axis=1)
    fp = (y_bin * (1.0 - y_true_t)).sum(axis=1)
    fn = ((1.0 - y_bin) * y_true_t).sum(axis=1)

    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / np.maximum(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-8)

    results_arr = np.stack(
        [
            thresholds,
            precision.mean(axis=1),
            recall.mean(axis=1),
            f1.mean(axis=1),
        ],
        axis=1,
    )

    best_idx = int(results_arr[:, 3].argmax())
    best_t, best_p, best_r, best_f1 = results_arr[best_idx]

    eligible = results_arr[results_arr[:, 2] >= float(min_recall_floor)]
    if len(eligible) > 0:
        precision_best_idx = int(eligible[:, 1].argmax())
        best_t_precision, best_p_precision, best_r_precision, best_f1_precision = eligible[precision_best_idx]
    else:
        best_t_precision, best_p_precision, best_r_precision, best_f1_precision = (np.nan, np.nan, np.nan, np.nan)

    if verbose:
        print(f"Evaluated {y_true.shape[0]} rows across {num_val_crops} crop(s) each")
        print(f"{'Threshold':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
        for t, p, r, f1_val in results_arr:
            marker = " <-- best_f1" if abs(t - best_t) < 1e-6 else ""
            print(f"{t:10.2f}  {p:10.4f}  {r:10.4f}  {f1_val:10.4f}{marker}")

        print(
            f"\nBest threshold by F1: {best_t:.2f} | "
            f"Precision: {best_p:.4f} | Recall: {best_r:.4f} | F1: {best_f1:.4f}"
        )
        if len(eligible) > 0:
            print(
                f"Best precision with recall >= {float(min_recall_floor):.2f}: "
                f"threshold={best_t_precision:.2f}, precision={best_p_precision:.4f}, "
                f"recall={best_r_precision:.4f}, f1={best_f1_precision:.4f}"
            )
        else:
            print(f"No threshold met recall floor >= {float(min_recall_floor):.2f}")

    return {
        "fast_sweep": fast_sweep,
        "eval_rows": int(eval_rows),
        "eval_batches": int(eval_batches),
        "num_val_crops": int(num_val_crops),
        "eval_feature_bins": int(eval_feature_bins),
        "thresholds": thresholds,
        "results_arr": results_arr,
        "best_t": float(best_t),
        "best_p": float(best_p),
        "best_r": float(best_r),
        "best_f1": float(best_f1),
        "best_t_precision": float(best_t_precision),
        "best_p_precision": float(best_p_precision),
        "best_r_precision": float(best_r_precision),
        "best_f1_precision": float(best_f1_precision),
        "min_recall_floor": float(min_recall_floor),
    }


def run_threshold_sweep(model, data_refs, config, verbose=True):
    """Run threshold sweep using grouped data references and config objects."""
    return threshold_sweep_multicrop(
        model=model,
        val_file_paths=data_refs.val_file_paths,
        val_targets=data_refs.val_targets,
        load_and_process_train=data_refs.load_and_process_train,
        load_and_process_eval=data_refs.load_and_process_eval,
        compute_spectrogram=data_refs.compute_spectrogram,
        batch_size=config.batch_size,
        min_samples=config.min_samples,
        num_classes=data_refs.num_classes,
        validation_eval_rows=config.validation_eval_rows,
        validation_num_crops=config.validation_num_crops,
        quick_run=config.quick_run,
        val_subset=config.val_subset,
        use_log_mel=config.use_log_mel,
        n_mels=config.n_mels,
        stft_fft_length=config.stft_fft_length,
        fast_sweep_max_rows=config.fast_sweep_max_rows,
        fast_sweep_max_crops=config.fast_sweep_max_crops,
        full_sweep_max_rows=config.full_sweep_max_rows,
        fast_sweep_threshold_start=config.fast_sweep_threshold_start,
        fast_sweep_threshold_stop=config.fast_sweep_threshold_stop,
        fast_sweep_threshold_step=config.fast_sweep_threshold_step,
        full_sweep_threshold_start=config.full_sweep_threshold_start,
        full_sweep_threshold_stop=config.full_sweep_threshold_stop,
        full_sweep_threshold_step=config.full_sweep_threshold_step,
        min_recall_floor=config.min_recall_floor,
        load_once_crops=config.load_once_crops,
        inference_batch_size=config.inference_batch_size,
        sample_rate=config.sample_rate,
        random_crop_samples=config.random_crop_samples,
        crop_seed=config.crop_seed,
        prediction_cache_dir=config.prediction_cache_dir,
        verbose=verbose,
    )


def append_run_metrics_csv(
    hist,
    context=None,
    threshold_summary=None,
    run_log_path="results/run_comparison.csv",
    verbose=True,
):
    """Append training outcomes and optional threshold sweep metrics to a CSV log."""
    context = {} if context is None else dict(context)
    threshold_summary = {} if threshold_summary is None else dict(threshold_summary)

    output_path = Path(run_log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h = pd.DataFrame(hist.history)

    if h.empty:
        raise ValueError(
            "Training history is empty (0 epochs recorded), so run metrics cannot be appended. "
            "Check steps_per_epoch/validation_steps and dataset filters."
        )

    last = h.iloc[-1]
    if "val_contest_auc" in h.columns:
        best_epoch_idx = int(h["val_contest_auc"].idxmax())
    elif "val_pr_auc" in h.columns:
        best_epoch_idx = int(h["val_pr_auc"].idxmax())
    else:
        best_epoch_idx = int(h["loss"].idxmin())

    epoch_numbers = h["epoch_number"] if "epoch_number" in h.columns else None
    best_epoch_value = int(epoch_numbers.iloc[best_epoch_idx]) if epoch_numbers is not None else (best_epoch_idx + 1)
    epoch_start = int(epoch_numbers.iloc[0]) if epoch_numbers is not None else 1
    epoch_end = int(epoch_numbers.iloc[-1]) if epoch_numbers is not None else int(len(h))

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **context,
        "epochs_ran": int(len(h)),
        "epoch_start": epoch_start,
        "epoch_end": epoch_end,
        "best_epoch": best_epoch_value,
        "last_loss": float(last["loss"]) if "loss" in h.columns else None,
        "last_pr_auc": float(last["pr_auc"]) if "pr_auc" in h.columns else None,
        "last_val_loss": float(last["val_loss"]) if "val_loss" in h.columns else None,
        "last_val_pr_auc": float(last["val_pr_auc"]) if "val_pr_auc" in h.columns else None,
        "last_val_contest_auc": float(last["val_contest_auc"]) if "val_contest_auc" in h.columns else None,
        "last_val_precision_at_0_2": float(last["val_precision@0.2"]) if "val_precision@0.2" in h.columns else None,
        "last_val_recall_at_0_2": float(last["val_recall@0.2"]) if "val_recall@0.2" in h.columns else None,
        "best_val_pr_auc": float(h.loc[best_epoch_idx, "val_pr_auc"]) if "val_pr_auc" in h.columns else None,
        "best_val_contest_auc": float(h.loc[best_epoch_idx, "val_contest_auc"]) if "val_contest_auc" in h.columns else None,
        "best_val_precision_at_0_2": float(h.loc[best_epoch_idx, "val_precision@0.2"])
        if "val_precision@0.2" in h.columns
        else None,
        "best_val_recall_at_0_2": float(h.loc[best_epoch_idx, "val_recall@0.2"]) if "val_recall@0.2" in h.columns else None,
        "threshold_best": threshold_summary.get("best_t"),
        "threshold_precision": threshold_summary.get("best_p"),
        "threshold_recall": threshold_summary.get("best_r"),
        "threshold_f1": threshold_summary.get("best_f1"),
        "threshold_eval_rows": threshold_summary.get("eval_rows"),
        "threshold_eval_batches": threshold_summary.get("eval_batches"),
        "threshold_eval_num_crops": threshold_summary.get("num_val_crops"),
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
        print(run_log_df.tail(5))

    return run_log_df
