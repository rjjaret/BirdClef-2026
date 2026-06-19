"""
PyTorch Dataset classes for BirdCLEF audio data with pure PyTorch preprocessing.
"""
import torch
import numpy as np
import os
import hashlib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from .audio_processing import load_audio_ogg, random_crop_audio, center_crop_audio, compute_log_mel_spectrogram
import soundfile as sf
import math
import random



class AudioSpectrogramDataset(Dataset):
    """
    PyTorch Dataset for audio spectrograms with on-the-fly computation using pure PyTorch.
    
    This version uses pure PyTorch preprocessing (no TensorFlow) which enables
    multiprocessing with NUM_WORKERS > 0.
    """
    
    def __init__(self, file_paths, targets, sample_rate=32000, n_fft=1024, hop_length=320,
                 n_mels=128, fmin=50.0, fmax=14000.0, crop_samples=None,
                 is_train=True, min_samples=32000):
        """
        Args:
            file_paths: Array of file paths to audio files
            targets: Array of multi-hot target vectors
            sample_rate: Audio sample rate
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbank bins
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            crop_samples: Number of samples to crop to (None = no cropping)
            is_train: Whether this is a training dataset (uses random crops)
            min_samples: Minimum number of audio samples required
        """
        self.file_paths = file_paths
        self.targets = targets
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.crop_samples = crop_samples
        self.is_train = is_train
        self.min_samples = min_samples
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        target = self.targets[idx]
        
        # Load audio (handles OGG files and __SEG__ paths)
        try:
            audio, sr = load_audio_ogg(file_path, sample_rate=self.sample_rate)
        except Exception as e:
            # Return None for failed loads - will be filtered in collate_fn
            return None
        
        # Check minimum length
        if len(audio) < self.min_samples:
            return None
        
        # Crop audio
        if self.crop_samples is not None:
            if self.is_train:
                audio, length = random_crop_audio(audio, self.crop_samples, self.min_samples)
            else:
                audio, length = center_crop_audio(audio, self.crop_samples)
        else:
            length = len(audio)
        
        # Compute log-mel spectrogram
        spec = compute_log_mel_spectrogram(
            audio,
            self.sample_rate,
            self.n_fft,
            self.hop_length,
            self.n_mels,
            self.fmin,
            self.fmax
        )
        
        # Transpose from [n_mels, time] to [time, n_mels] for model input
        spec = spec.transpose(0, 1)
        
        # Convert target to tensor
        target_tensor = torch.from_numpy(target).float()
        
        return spec, target_tensor


def collate_fn_pad(batch):
    """
    Custom collate function for DataLoader that pads spectrograms to batch max length.
    
    Filters out None samples (too short audio files) and pads remaining samples
    to the maximum length in the batch.
    """
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None, None
    
    specs, targets = zip(*batch)
    
    # Pad spectrograms to max length in batch
    specs_padded = pad_sequence(specs, batch_first=True, padding_value=0.0)
    targets_stacked = torch.stack(targets)
    
    return specs_padded, targets_stacked


def collate_windowed(batch):
    """
    Collate for windowed datasets. Returns (specs_padded, targets_stacked, rec_idxs, starts)
    or (specs_padded, targets_stacked, None, None) when rec_idxs not provided.
    """
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    # Support both (spec, target) and (spec, target, rec_idx, start)
    if len(batch[0]) == 2:
        specs, targets = zip(*batch)
        rec_idxs, starts = None, None
    else:
        specs, targets, rec_idxs, starts = zip(*batch)

    specs_padded = pad_sequence(specs, batch_first=True, padding_value=0.0)
    targets_stacked = torch.stack(targets)
    return specs_padded, targets_stacked, rec_idxs, starts


def compute_window_weights_from_class_weights(dataset, class_weights):
    """Compute per-window weights from per-class weights.

    dataset: WindowedAudioDataset (must have .index and .targets)
    class_weights: numpy array length C
    Returns: numpy array of length len(dataset)
    """
    if not hasattr(dataset, 'index') or dataset.index is None:
        raise ValueError('Dataset must be windowed (have dataset.index)')
    w = np.zeros(len(dataset), dtype=float)
    for i, (rec_idx, start) in enumerate(dataset.index):
        t = np.array(dataset.targets[rec_idx], dtype=float)
        w[i] = float(np.dot(t, class_weights))
    w = w + 1e-6
    return w


def log_epoch_coverage(targets_seen, labels=None, top_n=10):
    """Simple coverage logger: prints min/max/total and rarest classes."""
    total_seen = int(targets_seen.sum())
    per_class = np.array(targets_seen, dtype=int)
    print(f'Per-class seen min/max/total: {per_class.min()}/{per_class.max()}/{total_seen}')
    if labels is not None:
        order = np.argsort(per_class)[:top_n]
        print('Rarest classes:')
        for k in order:
            print(k, labels[k], int(per_class[k]))


def build_windowed_loader(
    file_paths,
    targets,
    window_samples,
    stride,
    batch_size,
    num_workers=4,
    mode='windowed-shuffled',
    use_weighted_sampler=False,
    class_weights=None,
    cache_dir='cache/spectrograms',
):
    """Helper to build WindowedAudioDataset + DataLoader + samplers.

    Adds `pin_memory`, `persistent_workers`, and `prefetch_factor` options so callers
    (notebooks) can tune DataLoader for their device (e.g., MPS requires pin_memory=False).

    Returns: (dataset, loader, sampler, weighted_sampler)
    """
    ds = WindowedAudioDataset(
        file_paths,
        targets,
        window_samples=window_samples,
        stride=stride,
        mode=mode,
        cache_dir=cache_dir,
    )
    sampler = None
    if mode.startswith('windowed'):
        sampler = EpochShuffledSampler(ds, seed=42)
    weighted_sampler = None
    if use_weighted_sampler and class_weights is not None:
        w = compute_window_weights_from_class_weights(ds, class_weights)
        weighted_sampler = WeightedRandomSampler(weights=torch.from_numpy(w).double(), num_samples=len(w), replacement=True)

    # Choose sampler: weighted sampler takes precedence if provided, otherwise epoch-shuffled sampler
    chosen_sampler = weighted_sampler if weighted_sampler is not None else sampler

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        sampler=chosen_sampler,
        collate_fn=collate_windowed,
        num_workers=num_workers,
        pin_memory=True,
    )
    return ds, loader, sampler, weighted_sampler


class CachedAudioSpectrogramDataset(Dataset):
    def __init__(self, file_paths, targets, cache_dir='cache/spectrograms',
                 sample_rate=32000, n_fft=1024, hop_length=320,
                 n_mels=128, fmin=50.0, fmax=14000.0, crop_samples=None,
                 is_train=True, min_samples=32000):
        """
        Args:
            file_paths: Array of file paths to audio files
            targets: Array of multi-hot target vectors
            cache_dir: Directory to store cached spectrograms
            sample_rate: Audio sample rate
            n_fft: FFT size for STFT
            hop_length: Hop length for STFT
            n_mels: Number of mel filterbank bins
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            crop_samples: Number of samples to crop to (None = no cropping)
            is_train: Whether this is a training dataset (uses random crops)
            min_samples: Minimum number of audio samples required
        """
        self.file_paths = file_paths
        self.targets = targets
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.crop_samples = crop_samples
        self.is_train = is_train
        self.min_samples = min_samples
        
        # Compute expected crop length in frames
        if crop_samples is not None:
            self.crop_frames = (crop_samples - n_fft) // hop_length + 1
        else:
            self.crop_frames = None
        
        # Cache statistics for debugging
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _get_cache_path(self, file_path):
        """Generate cache file path from audio file path."""
        # Create a hash of the file path + audio params for unique cache key
        cache_key = f"{file_path}_{self.sample_rate}_{self.n_fft}_{self.hop_length}_{self.n_mels}_{self.fmin}_{self.fmax}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{cache_hash}.npy"
        
    def __len__(self):
        return len(self.file_paths)
    
    def get_cache_stats(self):
        """Return cache hit/miss statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = 100.0 * self.cache_hits / total if total > 0 else 0.0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_accesses': total,
            'hit_rate_percent': hit_rate,
        }
    
    def reset_cache_stats(self):
        """Reset cache statistics counters."""
        self.cache_hits = 0
        self.cache_misses = 0
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        target = self.targets[idx]
        cache_path = self._get_cache_path(file_path)
        
        # Try to load from cache
        if cache_path.exists():
            try:
                spec = np.load(cache_path)
                spec = torch.from_numpy(spec).float()
                self.cache_hits += 1
            except Exception as e:
                # Cache corrupted, recompute
                spec = None
                self.cache_misses += 1
        else:
            spec = None
            self.cache_misses += 1
        
        # Compute and cache if not in cache
        if spec is None:
            try:
                # Load audio
                audio, sr = load_audio_ogg(file_path, sample_rate=self.sample_rate)
                
                # Check minimum length
                if len(audio) < self.min_samples:
                    return None
                
                # Compute full spectrogram
                spec = compute_log_mel_spectrogram(
                    audio,
                    self.sample_rate,
                    self.n_fft,
                    self.hop_length,
                    self.n_mels,
                    self.fmin,
                    self.fmax
                )
                
                # Transpose: [n_mels, time] -> [time, n_mels]
                if torch.is_tensor(spec):
                    spec = spec.transpose(0, 1)
                    spec_np = spec.numpy()
                else:
                    spec = spec.T
                    spec_np = spec
                
                # Save to cache
                np.save(cache_path, spec_np)
                
                # Convert to tensor if needed
                if not torch.is_tensor(spec):
                    spec = torch.from_numpy(spec).float()
                
            except Exception as e:
                # Failed to load/compute - print for debugging
                print(f"Cache write failed for {file_path}: {e}")
                return None
        
        # Apply random/center cropping in time dimension
        if self.crop_frames is not None and spec.shape[0] > self.crop_frames:
            if self.is_train:
                # Random crop
                max_start = spec.shape[0] - self.crop_frames
                start = np.random.randint(0, max_start + 1)
                spec = spec[start:start + self.crop_frames, :]
            else:
                # Center crop
                start = (spec.shape[0] - self.crop_frames) // 2
                spec = spec[start:start + self.crop_frames, :]
        
        # Convert target to tensor
        target_tensor = torch.from_numpy(target).float()
        
        return spec, target_tensor


class WindowedAudioDataset(Dataset):
    """
    Dataset that enumerates fixed-length windows from audio files so every
    region can be trained on. Supports overlapping windows via `stride` and
    modes: 'windowed-exhaustive', 'windowed-shuffled', and 'random-crop'.

    Returns: (spec, target_tensor, record_idx, start_sample)
    """
    def __init__(self, file_paths, targets, window_samples=160000, stride=None,
                 mode='windowed-exhaustive', max_windows_per_file=None,
                 sample_rate=32000, n_fft=1024, hop_length=320, n_mels=128,
                 fmin=50.0, fmax=14000.0, is_train=True, min_samples=32000,
                 precompute_lengths=True, cache_dir='cache/spectrograms'):
        self.file_paths = list(file_paths)
        self.targets = list(targets)
        self.window_samples = int(window_samples)
        self.stride = int(stride) if stride is not None else int(window_samples)
        self.mode = mode
        self.max_windows_per_file = max_windows_per_file
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.is_train = is_train
        self.min_samples = min_samples
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Precompute lengths (in samples) if requested to avoid loading audio now
        self.lengths = [None] * len(self.file_paths)
        if precompute_lengths:
            for i, p in enumerate(self.file_paths):
                try:
                    info = sf.info(str(p))
                    self.lengths[i] = int(info.frames)
                except Exception:
                    self.lengths[i] = None

        # Precompute frames needed for a window (used when slicing cached spectrograms)
        # Number of frames for a window of W samples: floor((W - n_fft)/hop) + 1
        if self.window_samples is not None:
            self.crop_frames = max(1, (self.window_samples - self.n_fft) // self.hop_length + 1)
        else:
            self.crop_frames = None

        # Build index for windowed modes
        self.index = None
        if self.mode.startswith('windowed'):
            self._build_index()

    def _get_cache_path(self, file_path):
        cache_key = f"{file_path}_{self.sample_rate}_{self.n_fft}_{self.hop_length}_{self.n_mels}_{self.fmin}_{self.fmax}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        return self.cache_dir / f"{cache_hash}.npy"

    def _build_index(self):
        idx = []
        for i, p in enumerate(self.file_paths):
            length = self.lengths[i]
            if length is None:
                # try to obtain length via soundfile.info; if fails, load file
                try:
                    info = sf.info(str(p))
                    length = int(info.frames)
                    self.lengths[i] = length
                except Exception:
                    # fallback: load file to inspect
                    try:
                        audio, sr = load_audio_ogg(p, sample_rate=self.sample_rate)
                        length = len(audio)
                        self.lengths[i] = length
                    except Exception:
                        length = 0

            if length <= 0:
                continue

            if length <= self.window_samples:
                idx.append((i, 0))
                continue

            # Compute start positions with given stride
            starts = list(range(0, length - self.window_samples + 1, self.stride))

            # Ensure last tail is included if it doesn't align
            last_start = length - self.window_samples
            if len(starts) == 0 or starts[-1] != last_start:
                starts.append(last_start)

            # If max_windows_per_file is set and fewer windows desired, sample evenly
            if self.max_windows_per_file and len(starts) > self.max_windows_per_file:
                idxs = [int(round(x)) for x in np.linspace(0, len(starts)-1, self.max_windows_per_file)]
                starts = [starts[k] for k in idxs]

            for s in starts:
                idx.append((i, s))

        self.index = idx

    def __len__(self):
        if self.mode.startswith('windowed'):
            return len(self.index)
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            if self.mode.startswith('windowed'):
                rec_idx, start = self.index[idx]
                file_path = self.file_paths[rec_idx]
                target = self.targets[rec_idx]

                # Prefer cached spectrograms when available to avoid recomputing STFTs
                cache_path = self._get_cache_path(file_path)
                spec_np = None
                if cache_path.exists():
                    try:
                        spec_np = np.load(cache_path)
                    except Exception:
                        spec_np = None

                if spec_np is not None:
                    # spec_np expected shape: [time_frames, n_mels]
                    # Compute start frame corresponding to `start` samples
                    start_frame = int(math.floor(start / float(self.hop_length)))
                    end_frame = start_frame + self.crop_frames
                    if start_frame < 0:
                        start_frame = 0
                    if end_frame > spec_np.shape[0]:
                        # slice available and pad remaining frames with zeros
                        available = spec_np.shape[0] - start_frame
                        if available <= 0:
                            slice_np = np.zeros((self.crop_frames, self.n_mels), dtype=np.float32)
                        else:
                            slice_np = spec_np[start_frame:start_frame + available]
                            pad_frames = self.crop_frames - available
                            if pad_frames > 0:
                                pad_arr = np.zeros((pad_frames, spec_np.shape[1]), dtype=spec_np.dtype)
                                slice_np = np.concatenate([slice_np, pad_arr], axis=0)
                    else:
                        slice_np = spec_np[start_frame:end_frame]

                    # Convert to torch tensor and ensure shape [time, n_mels]
                    spec = torch.from_numpy(slice_np.astype(np.float32))
                    target_tensor = torch.from_numpy(target).float()
                    return spec, target_tensor, rec_idx, start

                # Fallback: load audio and compute spectrogram for the window
                try:
                    audio, sr = load_audio_ogg(file_path, sample_rate=self.sample_rate)
                except Exception:
                    return None

                # Convert to mono if needed (load_audio_ogg does this)
                length = len(audio)
                end = start + self.window_samples
                if end > length:
                    pad = end - length
                    seg = np.pad(audio[start:length], (0, pad), mode='constant')
                else:
                    seg = audio[start:end]

                spec = compute_log_mel_spectrogram(
                    seg, self.sample_rate, self.n_fft, self.hop_length, self.n_mels, self.fmin, self.fmax
                )
                # transpose to [time, n_mels]
                spec = spec.transpose(0, 1)
                target_tensor = torch.from_numpy(target).float()
                return spec, target_tensor, rec_idx, start

            # fallback: random-crop behavior per-file
            file_path = self.file_paths[idx]
            target = self.targets[idx]
            try:
                audio, sr = load_audio_ogg(file_path, sample_rate=self.sample_rate)
            except Exception:
                return None
            if len(audio) < self.min_samples:
                return None
            if len(audio) <= self.window_samples:
                start = 0
            else:
                start = np.random.randint(0, len(audio) - self.window_samples + 1)
            seg = audio[start:start + self.window_samples]
            spec = compute_log_mel_spectrogram(
                seg, self.sample_rate, self.n_fft, self.hop_length, self.n_mels, self.fmin, self.fmax
            )
            spec = spec.transpose(0, 1)
            target_tensor = torch.from_numpy(target).float()
            return spec, target_tensor, idx, start
        except Exception as e:
            # Log full traceback and process info to help debug worker exits
            try:
                tb = traceback.format_exc()
                pid = os.getpid()
                with open(NOTEBOOK_ERROR_LOG, 'a') as fh:
                    fh.write(f'--- Worker exception (pid={pid}) idx={idx} ---\n')
                    fh.write(tb)
                    fh.write('\n')
            except Exception:
                pass
            # Re-raise so DataLoader shows the original exception behavior
            raise


class EpochShuffledSampler(torch.utils.data.Sampler):
    """Sampler that shuffles indices each epoch and allows deterministic seeding.

    Call `set_epoch(epoch)` each epoch to reseed shuffling.
    """
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self):
        n = len(self.data_source)
        rng = random.Random(self.seed + self.epoch)
        idxs = list(range(n))
        rng.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

