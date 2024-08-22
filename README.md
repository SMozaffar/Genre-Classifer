# Genre-Classifer

### audio_processor.py


**compute_melgram(audio_path)**

*What it does:*
1. Input: Takes the path of an audio file.
2. Output: Returns a mel-spectrogram with dimensions (1, 1, 96, 1366), where 96 is the number of mel-bins, and 1366 is the number of time frames.

*Key Steps:*
Set parameters:
1. SR (sampling rate): 12000 Hz.
2. N_FFT: 512 (the window size for the FFT).
3. N_MELS: 96 (number of mel-bins).
4. HOP_LEN: 256 (hop length between frames).
5. DURA: 29.12 seconds (duration to ensure 1366 time frames).

Load the audio:
1. The audio file is loaded at a sample rate of 12000 Hz using librosa.load().
2. src is the signal (audio data), and sr is the sample rate.

Padding or trimming:
1. If the audio signal is shorter than the required duration (DURA), it's padded with zeros.
2. If it's too long, it's trimmed to match the required number of samples (n_sample_fit).

Compute mel-spectrogram:
1. A mel-spectrogram is calculated using librosa.feature.melspectrogram().
2. The spectrogram is converted to a log scale using librosa.logamplitude().

Reshape: The result is reshaped to the format (1, 1, 96, 1366).


**compute_melgram_multiframe(audio_path, all_song=True)**

*What it does:*
1. Input: Takes the path of an audio file and a flag (all_song) to determine if the entire song should be processed.
2. Output: Returns a mel-spectrogram of multiple frames in the shape (N, 1, 96, 1366), where N is the number of frames.

*Key Steps:*
Parameters:
1. Similar parameters as the compute_melgram function, but with an additional DURA_TRASH variable to remove unwanted parts of the song.

Trim the song:
1. If all_song=False, it removes some initial and ending parts of the song.

Multi-frame spectrogram:
1. If the song is longer than the required duration (DURA), the code splits the audio into multiple frames and computes a mel-spectrogram for each frame.

Concatenation:
1. The mel-spectrograms of each frame are concatenated along the first dimension to get a result of shape (N, 1, 96, 1366).