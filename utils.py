import os
import time
import h5py
import sys
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
from math import floor
from sklearn.metrics import precision_recall_curve, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from audio_processor import compute_melgram, compute_melgram_multiframe


# Utility functions for data loading, saving, processing


# Save dataset to an HDF5 file
def save_data(path, data, name):
    with h5py.File(path + name, 'w') as hf:
        hf.create_dataset('data', data=data)


# Load dataset from an HDF5 file
def load_dataset(dataset_path):
    with h5py.File(dataset_path, 'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('labels'))
        num_frames = np.array(hf.get('num_frames'))
    return data, labels, num_frames


# Save data, labels, and frame numbers to an HDF5 file.
def save_dataset(path, data, labels, num_frames):
    with h5py.File(path, 'w') as hf:
        hf.create_dataset('data', data=data)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('num_frames', data=num_frames)


# Sort and print the genre predictions in descending order of confidence scores
def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)

    for name, score in sorted_result:
        score = np.array(score)
        score *= 100
        print(f'{name} : {score:.3f}   ', end=' ')
    print()


# Predict the label with the highest score
def predict_label(preds):
    labels = preds.argsort()[::-1]
    return labels[0]


# Load ground truth labels from a text file
def load_gt(path):
    with open(path, "r") as insTest:
        gt_total = []
        for lineTest in insTest:
            gt_total.append(int(lineTest))
        gt_total = np.array(gt_total)
    return gt_total


# Plot and save a confusion matrix
def plot_confusion_matrix(cnf_matrix, classes, title):
    cnfm_suma = cnf_matrix.sum(1)
    cnfm_suma_matrix = np.repeat(cnfm_suma[:, None], cnf_matrix.shape[1], axis=1)

    cnf_matrix = 10000 * cnf_matrix / cnfm_suma_matrix
    cnf_matrix = cnf_matrix / (100 * 1.0)
    print(cnf_matrix)

    fig = plt.figure()
    cmap = plt.cm.Blues
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, f'{cnf_matrix[i, j]:.2f}',
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    fig.savefig(title)


# Mel-spectrogram extraction function
def extract_melgrams(list_path, MULTIFRAMES, process_all_song, genre_mapping=None):
    if genre_mapping is None:
        # Define a mapping of genre folder names to numerical labels (0 to 9)
        genre_mapping = {
            'blues': 0, 'classical': 1, 'country': 2, 'disco': 3,
            'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9
        }
    
    melgrams = np.zeros((0, 1, 96, 1366), dtype=np.float32)
    song_paths = open(list_path, 'r').read().splitlines()
    labels = []
    num_frames_total = []

    for song_path in song_paths:
        print(song_path)

        # Get the genre name from the folder structure and map to label
        genre_name = os.path.basename(os.path.dirname(song_path))
        if genre_name in genre_mapping:
            label = genre_mapping[genre_name]
        else:
            print(f"Warning: Genre {genre_name} not found in genre mapping, skipping {song_path}")
            continue  # Skip this file if the genre is not in the mapping

        # Compute mel-spectrogram
        if MULTIFRAMES:
            melgram = compute_melgram_multiframe(song_path, process_all_song)
        else:
            melgram = compute_melgram(song_path)

        if melgram is None:  # Skip this file if melgram is None
            print(f"Skipping {song_path} due to loading error.")
            continue

        num_frames = melgram.shape[0]
        num_frames_total.append(num_frames)
        print(f'num frames: {num_frames}')

        # Append the label for each frame in the mel-spectrogram
        for _ in range(num_frames):
            labels.append(label)

        melgrams = np.concatenate((melgrams, melgram), axis=0)

    print("Melgram Extraction Complete")
    return melgrams, labels, num_frames_total


def plot_precision_recall_curve(Y_test, Y_pred, genres):
    """
    Plot precision-recall curves for each genre. Converts 1D arrays of predictions/labels
    into multi-label binary arrays (one-hot encoded). Enhanced for better visualization.
    """
    num_classes = len(genres)

    # Convert 1D arrays to 2D binary arrays using one-hot encoding (label binarization)
    Y_test_binarized = label_binarize(Y_test, classes=np.arange(num_classes))
    Y_pred_binarized = label_binarize(Y_pred, classes=np.arange(num_classes))

    plt.figure(figsize=(12, 10))

    colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for distinct genre colors

    # Iterate over each genre and plot precision-recall curve
    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(Y_test_binarized[:, i], Y_pred_binarized[:, i])
        plt.plot(recall, precision, lw=2, color=colors[i], label=genres[i])

    # Set labels and title
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curves for Each Genre", fontsize=16, fontweight='bold')

    # Customize legend, grid, and layout
    plt.legend(loc="lower left", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save and display the figure
    plt.savefig("precision_recall_curve.png")
    plt.show()


def plot_roc_curve(Y_test, Y_pred, genres):
    """
    Plot ROC curves for each genre. Converts 1D arrays of predictions/labels
    into multi-label binary arrays (one-hot encoded). Enhanced for better visualization.
    """
    num_classes = len(genres)

    # Convert 1D arrays to 2D binary arrays using one-hot encoding (label binarization)
    Y_test_binarized = label_binarize(Y_test, classes=np.arange(num_classes))
    Y_pred_binarized = label_binarize(Y_pred, classes=np.arange(num_classes))

    plt.figure(figsize=(12, 10))

    colors = plt.get_cmap('tab10').colors  # Use 'tab10' colormap for distinct genre colors

    # Iterate over each genre and plot ROC curve
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(Y_test_binarized[:, i], Y_pred_binarized[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors[i], label=f'{genres[i]} (AUC = {roc_auc:.2f})')

    # Plot the baseline for random guessing
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Random Guessing')

    # Set labels and title
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curves for Each Genre", fontsize=16, fontweight='bold')

    # Customize legend, grid, and layout
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # Save and display the figure
    plt.savefig("roc_curve.png")
    plt.show()


def print_classification_report(Y_test, Y_pred, genres):
    """
    Prints the classification report using true and predicted labels.
    Adjusts for Y_test being a 1D array of labels instead of multi-label format.
    """
    # Y_test and Y_pred are already 1D arrays of class indices
    Y_test_labels = Y_test  # True labels
    Y_pred_labels = Y_pred  # Predicted labels

    # Print classification report
    print("Classification Report:")
    print(classification_report(Y_test_labels, Y_pred_labels, target_names=genres))


def plot_confusion_matrix_heatmap(cnf_matrix, genres):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cnf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.title("Confusion Matrix Heatmap")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig('confusion_matrix_heatmap.png')
    plt.show()