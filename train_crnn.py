import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from genre_net import Genrefier_CRNN  
from utils import extract_melgrams, load_dataset, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve, print_classification_report, plot_confusion_matrix_heatmap


# List of genres
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def train_model(args):
    """
    Steps:
    1. Load the mel-spectrogram dataset or extract mel-spectrograms from audio.
    2. Train the model on the training data.
    3. Test the model on the test data (if specified).
    4. Save model weights periodically.

    Args:
    - args: Command line arguments provided by the user.
    """
    # Set device to MPS or CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load or extract mel-spectrograms
    if args.load_db:
        X_train, y_train, num_frames_train = load_dataset(args.train_dataset)
        X_test, y_test, num_frames_test = load_dataset(args.test_dataset)
    else:
        X_train, y_train, num_frames_train = extract_melgrams(args.train_songs_list, args.multiframes, process_all_song=True)
        X_test, y_test, num_frames_test = extract_melgrams(args.test_songs_list, args.multiframes, process_all_song=True)
        

    # Debug: ensure 10 unique labels in train and test
    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Unique labels in y_test: {np.unique(y_test)}")

    # Convert numpy arrays to PyTorch tensors, ensure correct shape
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 96, 1366).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 96, 1366).to(device)
    Y_train = torch.tensor(np.eye(args.nb_classes)[y_train], dtype=torch.float32).to(device)
    Y_test = torch.tensor(np.eye(args.nb_classes)[y_test], dtype=torch.float32).to(device)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, Y_train, torch.tensor(num_frames_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    model = Genrefier_CRNN().to(device)
    if args.load_weights:
        model.load_state_dict(torch.load(args.weights_path))

    # Optimizer and loss 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    if args.train:
        model.train()
        for epoch in range(1, args.epochs + 1):
            epoch_loss = 0.0
            for i, (batch_X, batch_Y, batch_num_frames) in enumerate(train_loader):
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(batch_X)

                # Compute loss
                loss = criterion(outputs, batch_Y)

                # Backward pass and optimization
                loss.backward()
                optimizer.step()

                # Accumulate loss
                epoch_loss += loss.item()

                # DEBUGGING: Print (path, label, genre) for one song in batch (to make sure data is being loaded correctly)
                # song_index = i * args.batch_size  # Index of first song
                # song_path = song_paths[song_index]  # Path to song
                # song_label = y_train[song_index]  # Integer label of song
                # song_genre = genres[song_label]  # Corresponding genre
                # print(f"Song Path: {song_path}, Label: {song_label}, Genre: {song_genre}")

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {epoch_loss/len(train_loader)}")

            # Save model weights every 10 epochs
            if epoch % 10 == 0:
                torch.save(model.state_dict(), args.weights_path + args.model_name + f"_epoch_{epoch}.pth")
                print(f"Saved model at epoch {epoch} to {args.weights_path}")

    # Testing loop
    if args.test:
        print("Testing the model...")
        model.eval()

        # Use DataLoader for batch processing TODO: fix memory issues
        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        predictions = []
        true_labels = []

        # Disable gradient calculations
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                # Move batch data to the correct device
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                # Forward pass 
                outputs = model(batch_X)

                # Get predicted labels (select class with highest probability)
                _, predicted = torch.max(outputs, dim=1)

                # Collect predictions and true labels
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_Y.argmax(dim=1).cpu().numpy())

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(true_labels, predictions)

        # Plot confusion matrix and heatmap
        plot_confusion_matrix(cnf_matrix, classes=np.arange(args.nb_classes), title='Confusion Matrix')

        # Plot Precision-Recall Curve
        plot_precision_recall_curve(true_labels, predictions, genres)

        # Plot ROC curves and AUC
        plot_roc_curve(true_labels, predictions, genres)

        # Class-wise F1 scores
        print_classification_report(true_labels, predictions, genres)


# Command line arguments
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Music Tagger CRNN")
    parser.add_argument('--train_songs_list', type=str, required=True, help='Path to train songs list')
    parser.add_argument('--test_songs_list', type=str, required=True, help='Path to test songs list')
    parser.add_argument('--multiframes', type=int, default=1, help='Use multiple frames per song')
    parser.add_argument('--load_db', type=int, default=0, help='Load pre-extracted mel-spectrogram dataset')
    parser.add_argument('--train_dataset', type=str, help='Path to pre-extracted training dataset')
    parser.add_argument('--test_dataset', type=str, help='Path to pre-extracted testing dataset')
    parser.add_argument('--nb_classes', type=int, default=10, help='Number of classes (genres)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--model_name', type=str, default='crnn_classifier', help='Model name for saving')
    parser.add_argument('--weights_path', type=str, default='models_trained/', help='Path to save model weights')
    parser.add_argument('--test', type=int, default=1, help='Test the model after training')
    parser.add_argument('--train', type=int, default=1, help='Test the model after training')
    parser.add_argument('--load_weights', type=bool, default=False, help='Load model weights')


    args = parser.parse_args()

    # song_paths is a list of the file paths of the songs
    song_paths = open(args.train_songs_list).read().splitlines()


    train_model(args)
