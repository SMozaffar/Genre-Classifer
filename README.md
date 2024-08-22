# 🎶 Music Genre Classification using CNN + RNN 🎵

This project implements a **Music Genre Classification** system using a **Convolutional Neural Network (CNN)** combined with a **Recurrent Neural Network (RNN)**. The model is trained on the [GTZAN dataset](http://marsyas.info/downloads/datasets.html), a well-known dataset in the field of music information retrieval. The primary goal of this project is to classify audio tracks into one of 10 music genres based on their acoustic characteristics.

## 🚀 Project Overview

The main objective of this project is to develop a deep learning model that can classify audio samples into 10 different genres by analyzing their **mel-spectrograms**. The approach combines the strengths of **CNNs** for feature extraction and **RNNs** for capturing temporal dependencies in audio data.

The project includes:
- **Audio preprocessing** using mel-spectrogram extraction.
- **CNN + RNN architecture** for learning both spatial (CNN) and temporal (RNN) patterns.
- **Training** and **testing** routines with model evaluation.
- **Batch processing** to handle large datasets efficiently on GPU (or CPU if necessary).

### 🎯 Target Music Genres:
The model classifies audio into one of the following 10 genres:
- **Blues**
- **Classical**
- **Country**
- **Disco**
- **HipHop**
- **Jazz**
- **Metal**
- **Pop**
- **Reggae**
- **Rock**

## 📁 Dataset: GTZAN

The project utilizes the **GTZAN dataset**, which contains 1,000 audio tracks evenly distributed across 10 music genres (100 tracks per genre). Each track is a 30-second excerpt and is provided as a `.wav` file.

**Dataset Details**:
- **Number of tracks**: 1000
- **Duration of each track**: 30 seconds
- **Sampling rate**: 22.05 kHz
- **Genres**: Blues, Classical, Country, Disco, HipHop, Jazz, Metal, Pop, Reggae, Rock

You can download the dataset from the [official GTZAN dataset page](http://marsyas.info/downloads/datasets.html).

### 🎵 Audio Preprocessing

Before feeding the audio data into the model, we extract **mel-spectrograms** from each track. The mel-spectrogram provides a time-frequency representation of the audio signal, making it suitable for processing by convolutional layers.

#### Steps for Audio Preprocessing:
1. **Resampling**: All audio tracks are resampled to a consistent sample rate of 12 kHz.
2. **Mel-Spectrogram Extraction**: Using **Librosa**, mel-spectrograms are generated with the following parameters:
   - **Number of mel bins**: 96
   - **FFT window size**: 512
   - **Hop length**: 256
   - **Duration of each frame**: 29.12 seconds
3. **Log-Scaling**: The mel-spectrograms are log-scaled to improve model performance by stabilizing variations in amplitude.

## 🛠️ Techniques Used

### 📊 Mel-Spectrograms

Mel-spectrograms are 2D time-frequency representations of audio signals. We treat them like images, allowing us to apply convolutional neural networks (CNNs) for feature extraction.

### 🧠 CNN + RNN Architecture

We use a combination of CNN and RNN layers to process the mel-spectrograms:

- **CNN Layers**:
  - The **CNN** part extracts local spatial features from the mel-spectrogram. This helps capture the structure of audio signals in both time and frequency domains.
  - Five convolutional layers followed by **max-pooling** and **batch normalization** are used to reduce dimensionality and improve generalization.
  
- **RNN Layers**:
  - The **RNN** (specifically, GRU) captures sequential dependencies in the mel-spectrograms. Since music signals are time-dependent, RNNs are well-suited for learning the temporal aspects of the data.
  - Two **GRU** layers are used after the CNN layers to capture the sequential patterns.

- **Output Layer**:
  - A **fully connected (Dense) layer** followed by a **sigmoid activation** is used to make genre predictions. The sigmoid function is used since this is a multi-label classification problem.

### 🎓 Training Process

- **Loss function**: `CrossEntropyLoss` is used for multi-class classification.
- **Optimizer**: The model is trained using the `Adam` optimizer with a learning rate of 0.001.
- **Batch size**: Default batch size of 32 is used for training, but it can be customized via the command line.
- **GPU support**: The model is trained on **CUDA** (if available) or falls back to **CPU**.

## 📜 Relevant Research Papers

Here are some key papers that inspired the architecture and techniques used in this project:

1. **[Automatic Tagging Using Deep Convolutional Neural Networks](https://arxiv.org/abs/1606.00298)**  
   *Keunwoo Choi et al., 2016*  
   This paper introduces the use of deep CNNs for automatic music tagging and genre classification using mel-spectrograms.

2. **[CRNN: Convolutional Recurrent Neural Networks for Music Classification](https://arxiv.org/abs/1609.04243)**  
   *Keunwoo Choi et al., 2016*  
   This paper details the benefits of combining CNNs and RNNs for audio classification tasks. It demonstrates how RNNs can capture temporal dependencies in audio data.

3. **[Music-AutoTagging-Keras](https://github.com/keunwoochoi/music-auto_tagging-keras)**  
   *Keunwoo Choi’s GitHub repository*  
   A popular repository that implements automatic music tagging using Keras, providing inspiration for many implementations in this field.

## 📦 Installation and Running the Model

### 1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/music-genre-classification.git
   cd music-genre-classification
