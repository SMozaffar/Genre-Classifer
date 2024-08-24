import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from genre_net import Genrefier_CRNN
from utils import save_data, load_dataset, save_dataset, sort_result, predict_label, extract_melgrams


def test_model(args):
    """
    Steps:
    1. Load the pre-trained model weights.
    2. Extract mel-spectrograms for the test data.
    3. Run the model to predict genres for the test songs.
    4. Visualize and output the predictions.

    Args:
    - args: Command line arguments provided by the user.
    """
    # Set device to MPS or CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load tags and model
    tags = np.array(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock'])
    model = Genrefier_CRNN()
    if args.load_weights:
        model.load_state_dict(torch.load(args.weights_path))

    model.to(device)
    model.eval()

    # Extract mel-spectrograms
    X_test, Y_test, num_frames_test = extract_melgrams(args.test_songs_list, args.multiframes, process_all_song=True)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)  # Convert to torch tensor

    num_frames_test = np.array(num_frames_test)

    # Initialize results
    results = np.zeros((X_test.shape[0], len(tags)))  # Model's genre predictions
    predicted_labels_mean = np.zeros((num_frames_test.shape[0], 1))  # Mean prediction per song
    predicted_labels_frames = np.zeros((X_test.shape[0], 1))  # Frame-by-frame predictions

    # Perform prediction
    with torch.no_grad():
        previous_numFrames = 0  # Track frame position 
        for i in range(num_frames_test.shape[0]):
            num_frames = num_frames_test[i]  # Number of frames
            print(f'Song {i}: {os.path.basename(args.test_songs_list)}')

            # Get predictions for each frame in song
            outputs = model(X_test[previous_numFrames:previous_numFrames + num_frames])
            outputs = F.softmax(outputs, dim=1)
            results[previous_numFrames:previous_numFrames + num_frames] = outputs.cpu().numpy()

            # Normalization and frame-by-frame label prediction
            for j in range(previous_numFrames, previous_numFrames + num_frames):
                total = results[j, :].sum()
                results[j, :] = results[j, :] / total  # Normalize probabilities
                predicted_label_frames = predict_label(results[j, :])  # Get predicted genre label
                predicted_labels_frames[j] = predicted_label_frames
            previous_numFrames += num_frames

        # Compute mean predictions (entire song)
        mean = results.mean(0)
        predicted_label_mean = predict_label(mean)
        predicted_labels_mean[i] = predicted_label_mean

        # Visualization
        plot_genre_distribution(mean, tags, args.output_file)


def plot_genre_distribution(mean, tags, output_file):
    colors = ['b', 'g', 'c', 'r', 'm', 'k', 'y', '#ff1122', '#5511ff', '#44ff22']
    fig, ax = plt.subplots()
    index = np.arange(len(tags))
    bar_width = 0.2
    plt.bar(x=index, height=mean, width=bar_width, alpha=1, color=colors)
    plt.xlabel('Genres')
    plt.ylabel('Percentage')
    plt.title('Scores by genre')
    plt.xticks(index + bar_width / 2, tags)
    plt.tight_layout()
    fig.autofmt_xdate()
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the CRNN model")
    parser.add_argument('--test_songs_list', type=str, required=True, help='Path to the test song list')
    parser.add_argument('--weights_path', type=str, default='', help='Path to model weights')
    parser.add_argument('--load_weights', type=bool, default=False, help='Load model weights')
    parser.add_argument('--multiframes', type=int, default=1, help='Use multiple frames per song')
    parser.add_argument('--output_file', type=str, default='genres_prediction.png', help='Output file for genre distribution chart')
    
    args = parser.parse_args()
    test_model(args)
