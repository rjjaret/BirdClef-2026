# audio_processing.py
# Pure PyTorch audio loading and processing functions
from os.path import isfile
import matplotlib
import torch
import librosa
import numpy as np
import soundfile as sf

# Helper Functions
import torchaudio
from pathlib import Path

# Global cache for mel filterbanks to avoid repeated TensorFlow calls
_MEL_FILTERBANK_CACHE = {}

def plot_spectogram(spec):
    import matplotlib.pyplot as plt

    # Let's plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    plt.imshow(spec[0].detach().numpy(), aspect="auto", origin="lower")
    plt.title("Mel spectrogram")
    plt.xlabel("Frames")
    plt.ylabel("Frequency")
    # plt.colorbar()
    plt.show()

def display_audio(wav, h):
    from IPython.display import Audio, display


def compute_log_mel_spectrogram(audio, sample_rate, n_fft, hop_length, n_mels, fmin, fmax):
    """
    Compute log-mel spectrogram from audio using pure PyTorch.
    
    Uses power spectrogram (magnitude squared) before mel filterbank.
    
    Args:
        audio: Numpy or torch tensor of audio samples
        sample_rate: Sample rate of audio
        n_fft: FFT size
        hop_length: Hop length for STFT
        n_mels: Number of mel filterbank bins
        fmin: Minimum frequency
        fmax: Maximum frequency
    
    Returns:
        log_mel: Log-mel spectrogram as torch tensor [n_mels, time]
    """
    # Convert to torch if needed
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    
    # Ensure 2D: [1, samples]
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Create or retrieve cached mel filterbank
    # Cache key based on mel filterbank parameters
    cache_key = (sample_rate, n_fft, n_mels, fmin, fmax)
    
    if cache_key not in _MEL_FILTERBANK_CACHE:
        # Create librosa mel filterbank (pure PyTorch dependencies)
        mel_fb_np = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            norm=None
        )
        mel_fb = torch.from_numpy(mel_fb_np).float()
        _MEL_FILTERBANK_CACHE[cache_key] = mel_fb
    else:
        mel_fb = _MEL_FILTERBANK_CACHE[cache_key]
    
    # Create Hann window
    window = torch.hann_window(n_fft)
    
    # Compute STFT with center=False
    stft = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=False,
        pad_mode='reflect',
        normalized=False,
        onesided=True,
        return_complex=True
    )
    
    # Compute magnitude then square to get power
    magnitude = torch.abs(stft)
    power_spec = magnitude ** 2
    
    # Apply mel filterbank to power: [n_mels, n_fft/2+1] @ [n_fft/2+1, time] = [n_mels, time]
    mel_power = torch.matmul(mel_fb, power_spec.squeeze(0))
    
    # Apply log scaling
    log_mel = torch.log(mel_power + 1e-6)
    
    return log_mel


def compute_spectrogram(wav, h, sample_rate=None, use_log_mel_pcen=False):
    """
    Compute spectrogram (wrapper for backwards compatibility).
    
    Args:
        wav: Audio tensor
        h: Config dict with audio parameters
        sample_rate: Optional sample rate override
        use_log_mel_pcen: Ignored (kept for compatibility)
    
    Returns:
        spec: Spectrogram tensor
    """
    sr = sample_rate if sample_rate is not None else h["sample_rate"]
    
    return compute_log_mel_spectrogram(
        wav,
        sr,
        h["stft_frame_length"],
        h["stft_frame_step"],
        h["n_mels"],
        h["mel_fmin"],
        h["mel_fmax"]
    )


def load_audio_ogg(file_path, sample_rate=32000, segment_info=None):
    """
    Load audio from OGG file using soundfile.
    
    Args:
        file_path: Path to OGG file (may include __SEG__ token for segments)
        sample_rate: Target sample rate
        segment_info: Tuple of (start_seconds, duration_seconds) for segment loading
    
    Returns:
        audio: Numpy array of audio samples
        sr: Sample rate
    """
    # Handle __SEG__ token in path (train_soundscapes format)
    if '__SEG__' in str(file_path):
        # Parse segment info from path
        # Format: file.ogg__SEG__5.000__10.000 (start_sec, end_sec)
        parts = str(file_path).split('__SEG__')
        file_path = parts[0]  # Already has .ogg extension
        seg_parts = parts[1].split('__')
        start_sec = float(seg_parts[0])
        end_sec = float(seg_parts[1])
        duration_sec = end_sec - start_sec
        segment_info = (start_sec, duration_sec)
    
    # Load audio (optionally just a segment)
    if segment_info is not None:
        start_sec, duration_sec = segment_info
        start_frame = int(start_sec * sample_rate)
        num_frames = int(duration_sec * sample_rate)
        audio, sr = sf.read(file_path, start=start_frame, frames=num_frames, dtype='float32')
    else:
        audio, sr = sf.read(file_path, dtype='float32')
    
    # Resample if needed
    if sr != sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    return audio, sr


def random_crop_audio(audio, crop_samples, min_samples=32000):
    """
    Randomly crop audio to specified length.
    
    Args:
        audio: Numpy array of audio samples
        crop_samples: Number of samples to crop to
        min_samples: Minimum required samples
    
    Returns:
        cropped_audio: Numpy array of cropped audio
        length: Actual length of audio
    """
    length = len(audio)
    
    if length < min_samples:
        # Pad if too short
        audio = np.pad(audio, (0, min_samples - length), mode='constant')
        length = len(audio)
    
    if length <= crop_samples:
        # Pad to crop_samples if needed
        audio = np.pad(audio, (0, crop_samples - length), mode='constant')
        return audio, length
    
    # Random crop
    max_start = length - crop_samples
    start = np.random.randint(0, max_start + 1)
    return audio[start:start + crop_samples], crop_samples


def center_crop_audio(audio, crop_samples):
    """
    Center crop audio to specified length (deterministic for validation).
    
    Args:
        audio: Numpy array of audio samples
        crop_samples: Number of samples to crop to
    
    Returns:
        cropped_audio: Numpy array of cropped audio
        length: Actual length of audio
    """
    length = len(audio)
    
    if length <= crop_samples:
        # Pad to crop_samples if needed
        audio = np.pad(audio, (0, crop_samples - length), mode='constant')
        return audio, length
    
    # Center crop
    start = (length - crop_samples) // 2
    return audio[start:start + crop_samples], crop_samples


def load_audio_tensor_from_wav(filename, h, sr=None, mono=True, padding=False):
    # sr parameter kept for backwards compatibility but h["sample_rate"] is used if sr is None
    if sr is None:
        sr = h["sample_rate"]
    waveform, file_sr= torchaudio.load(filename)

    if sr!= file_sr:
        waveform = torchaudio.functional.resample(waveform, file_sr, sr)
        print(f"Resampling {filename} from {file_sr} to {sr}")  
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_length = round(h["max_audio_seconds"]* h["sample_rate"])

    # Ensure waveform is at least target_length by padding
    if padding and waveform.shape[1] < target_length:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))

    # Ensure waveform is exactly target_length by truncating
    # This step is crucial if, for example, resampling slightly overshot and made it longer
    waveform = waveform[:, :target_length]

    return waveform.float()

def load_audio_tensors(root_dir, h, max_count=0, padding=True):
    audio_tensors = []
    lengths = []
    current_count = 0
    for file_path in Path(root_dir).rglob('*'):
      if str(file_path).lower().endswith(('.wav')):
        waveform = load_audio_tensor_from_wav(file_path, h, sr=None, mono=True, padding=padding)

        if len(waveform) > 1:
            for wav in waveform:
                audio_tensors.append((wav, h["sample_rate"], str(file_path)))
        else:
            audio_tensors.append((waveform, h["sample_rate"], str(file_path)))
            lengths.append(len(waveform[0]))
            current_count += 1
    
        if max_count > 0 and current_count >= max_count:
          break

    return audio_tensors, lengths