import torchaudio
import librosa # type: ignore
import torch
import numpy as np
from math import floor


# Function to load an audio file using torchaudio and resample if necessary
def load_audio_with_torchaudio(audio_path, sample_rate=12000):
    """
    Load audio using torchaudio. If the sample rate does not match the specified 
    sample_rate, resample the audio.
    
    Args:
    audio_path (str): Path to the audio file.
    sample_rate (int): Target sample rate. Default is 12000 Hz.
    
    Returns:
    np.ndarray: Loaded waveform as a numpy array.
    """
    try:
        waveform, sr = torchaudio.load(audio_path)  # Load the audio file
        if sr != sample_rate:
            # Resample if sample rate does not match
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
            waveform = resampler(waveform)
        return waveform.numpy()[0]  # Return as numpy array
    except Exception as e:
        print(f"Error loading {audio_path} with torchaudio: {e}")
        return None


# Function to compute mel-spectrogram of a single audio file
def compute_melgram(audio_path):
    """
    Compute a mel-spectrogram of an audio file and return a tensor with shape (1, 1, 96, 1366),
    where 96 is the number of mel bins and 1366 is the number of time frames.
    
    Args:
    audio_path (str): Path to the audio file.
    
    Returns:
    torch.Tensor: Mel-spectrogram in the shape (1, 1, 96, 1366).
    """
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12  # Ensure 1366 frames

    # Load audio waveform (torchaudio)
    waveform = load_audio_with_torchaudio(audio_path, sample_rate=SR)
    if waveform is None:
        return None  # Skip if the file couldn't be loaded

    # Adjust waveform length (must match duration)
    n_sample = len(waveform)
    n_sample_fit = int(DURA * SR)

    if n_sample < n_sample_fit:
        # If too short, pad with zeros
        waveform = np.pad(waveform, (0, n_sample_fit - n_sample), mode='constant')
    elif n_sample > n_sample_fit:
        # If too long, center crop
        start_idx = (n_sample - n_sample_fit) // 2
        waveform = waveform[start_idx:start_idx + n_sample_fit]

    # Compute mel-spectrogram (librosa)
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)

    # Convert to log scale
    mel_spec_log = np.log(np.maximum(mel_spec, 1e-9))

    return torch.tensor(mel_spec_log).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, 96, 1366)


# Function to compute mel-spectrograms for multiple frames in an audio file
def compute_melgram_multiframe(audio_path, all_song=True):
    """
    Compute a multi-frame mel-spectrogram of an audio file, returning a tensor 
    with shape (N, 1, 96, 1366), where N is the number of frames.
    
    Args:
    audio_path (str): Path to the audio file.
    all_song (bool): Whether to process the entire song or discard the first and last 20 seconds.
    
    Returns:
    torch.Tensor: Mel-spectrogram of multiple frames.
    """
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12
    DURA_TRASH = 0 if all_song else 20  # Optionally discard first and last 20 seconds

    # Load audio waveform (torchaudio)
    waveform = load_audio_with_torchaudio(audio_path, sample_rate=SR)
    if waveform is None:
        return None  # Skip if the file couldn't be loaded

    # Remove trash from the beginning and end
    n_sample = len(waveform)
    n_sample_trash = int(DURA_TRASH * SR)
    waveform = waveform[n_sample_trash:n_sample - n_sample_trash]

    # Split the waveform into multiple frames
    n_sample = len(waveform)
    n_sample_fit = int(DURA * SR)
    ret = []

    if n_sample < n_sample_fit:
        # Pad if audio too short 
        waveform = np.pad(waveform, (0, n_sample_fit - n_sample), mode='constant')
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
        mel_spec_log = np.log(np.maximum(mel_spec, 1e-9))
        ret.append(torch.tensor(mel_spec_log).unsqueeze(0).unsqueeze(0))
    else:
        # Split into frames
        N = floor(n_sample / n_sample_fit)
        for i in range(N):
            frame = waveform[i * n_sample_fit:(i + 1) * n_sample_fit]
            mel_spec = librosa.feature.melspectrogram(y=frame, sr=SR, n_fft=N_FFT, hop_length=HOP_LEN, n_mels=N_MELS)
            mel_spec_log = np.log(np.maximum(mel_spec, 1e-9))
            ret.append(torch.tensor(mel_spec_log).unsqueeze(0).unsqueeze(0))

    return torch.cat(ret, dim=0)  # Concatenate into single tensor; Shape (N, 1, 96, 1366)
