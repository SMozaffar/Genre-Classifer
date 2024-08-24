# 🎶 Music Genre Classification using CNN + RNN 🎵

This project implements a PyTorch adaptation of the model from [Choi et al.](https://github.com/keunwoochoi/music-auto_tagging-keras) to train a **Genre Classification** system using a **Convolutional Neural Network (CNN)** combined with a **Recurrent Neural Network (RNN)**. The model is trained on the [GTZAN dataset](http://marsyas.info/downloads/datasets.html). The primary goal of this project is to classify audio tracks into one of 10 music genres based on their acoustic characteristics.

## 🚀 Project Overview

The main objective of this project is to develop a deep learning model that can classify audio samples into 10 different genres by analyzing their **mel-spectrograms**. The approach combines the strengths of **CNNs** for feature extraction and **RNNs** for capturing temporal dependencies in audio data.

![Mel-Spectrogram](https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Mel_spectrogram_example.png/400px-Mel_spectrogram_example.png)
*Example of a mel-spectrogram.*

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
   ```

### 2. **Install Dependencies:**:

   

## Results/Analysis
# Model Evaluation Report

## Overview:
- **Accuracy:** 77% (The model correctly classified 77% of the test samples.)
- **Macro avg:** 0.80 precision, 0.77 recall, 0.77 F1-score (Average performance across all classes, treating each class equally.)
- **Weighted avg:** 0.80 precision, 0.77 recall, 0.77 F1-score (Performance averages, weighted by the number of instances in each class.)

## Class-wise Analysis:

### 1. **Blues:**
   - **Precision:** 0.68 — Out of all the predictions made as "blues," 68% were correct.
   - **Recall:** 0.65 — Out of all the true "blues" samples, 65% were correctly predicted.
   - **F1-Score:** 0.67 — This is the harmonic mean of precision and recall, indicating a moderate performance. 
   - **Analysis:** The model confuses blues with other genres, especially rock (20%), suggesting a similarity in features between these two genres.

### 2. **Classical:**
   - **Precision:** 0.95 — The model is highly confident in predicting classical music.
   - **Recall:** 0.95 — It also detects almost all classical music samples.
   - **F1-Score:** 0.95 — Very high performance. 
   - **Analysis:** Classical music is easily distinguishable by the model, likely due to its unique features compared to other genres.

### 3. **Country:**
   - **Precision:** 0.92 — High precision, meaning when the model predicts "country," it is usually correct.
   - **Recall:** 0.55 — However, the recall is quite low, indicating that many actual "country" samples are misclassified (only 55% correctly identified).
   - **F1-Score:** 0.69 — Reflects the imbalance between precision and recall.
   - **Analysis:** The model confuses "country" with "rock" and "blues" (35% combined confusion with other genres), which may suggest similar audio patterns or features between these genres.

### 4. **Disco:**
   - **Precision:** 0.73 — Lower precision; it often predicts disco incorrectly.
   - **Recall:** 0.80 — Fairly good at detecting disco samples.
   - **F1-Score:** 0.76 — Overall decent performance.
   - **Analysis:** Disco might be mistaken for "pop" (10%), which is understandable as both genres share upbeat rhythms and instrumentation.

### 5. **Hiphop:**
   - **Precision:** 0.94 — Excellent precision, meaning the model predicts hip-hop very accurately.
   - **Recall:** 0.80 — However, some hip-hop samples are misclassified (likely with reggae and disco).
   - **F1-Score:** 0.86 — Overall, strong performance.
   - **Analysis:** There is a slight confusion with genres like disco and reggae, which share rhythmic similarities with hip-hop.

### 6. **Jazz:**
   - **Precision:** 0.86 — Strong precision.
   - **Recall:** 0.90 — High recall; the model detects most jazz samples.
   - **F1-Score:** 0.88 — Excellent overall performance.
   - **Analysis:** Jazz is fairly well predicted, but there are slight confusions with blues and classical music.

### 7. **Metal:**
   - **Precision:** 0.90 — Very high precision.
   - **Recall:** 0.95 — The model detects almost all metal samples.
   - **F1-Score:** 0.93 — Excellent overall performance.
   - **Analysis:** Metal, with its distinct heavy instrumentation, is well classified by the model.

### 8. **Pop:**
   - **Precision:** 0.81 — Decent precision.
   - **Recall:** 0.65 — Many pop samples are misclassified (with disco, hip-hop, and reggae).
   - **F1-Score:** 0.72 — Overall, performance is moderate.
   - **Analysis:** Pop is often confused with similar genres like disco and hip-hop, which have overlapping musical elements.

### 9. **Reggae:**
   - **Precision:** 0.82 — Strong precision.
   - **Recall:** 0.70 — Some reggae samples are misclassified, possibly with rock and blues.
   - **F1-Score:** 0.76 — Good overall performance.
   - **Analysis:** Reggae shares rhythmic patterns with rock and hip-hop, leading to occasional misclassifications.

### 10. **Rock:**
   - **Precision:** 0.40 — Low precision, meaning when the model predicts "rock," it often gets it wrong.
   - **Recall:** 0.70 — However, it detects most rock samples, but misclassifies them with other genres (blues and country).
   - **F1-Score:** 0.51 — Low F1-score, indicating poor performance compared to other genres.
   - **Analysis:** Rock is often confused with blues, country, and metal, all of which share certain musical elements like electric guitars and similar tempo.

## Key Insights:
- **Classical and metal** have the best performance, as these genres have very distinct features (e.g., orchestration in classical, distorted guitars in metal).
- **Rock, blues, and country** show more confusion with each other, likely because they share certain instrumentation and rhythm patterns. The model struggles to distinguish these genres.
- **Pop and disco** also have some overlap, which makes sense since they both feature upbeat, danceable rhythms.

## Next Steps to Improve Performance:
1. **Increase Data:** Add more training data for confusing genres like rock, country, and blues to help the model learn better distinctions between them.
2. **Fine-tune Model:** Adjust model hyperparameters like learning rate, batch size, and number of epochs to improve training.
3. **Data Augmentation:** Apply audio data augmentation techniques like time stretching, pitch shifting, or noise injection to make the model more robust.
4. **Model Enhancements:** Experiment with deeper or more complex models, or try adding additional features like tempo, beat tracking, or harmonic components.
5. **Confusion Matrix Analysis:** Focus on the pairs of genres that show confusion and try to understand if there are certain overlapping features causing the issue (e.g., overlapping frequencies in mel-spectrograms). 

This analysis indicates that while the model performs well on certain distinct genres, it struggles with genres that have similar audio characteristics.

