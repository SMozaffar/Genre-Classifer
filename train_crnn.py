import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from genre_net import MusicTaggerCRNN  # Assuming your CRNN model in PyTorch
from utils import extract_melgrams, load_dataset, plot_confusion_matrix

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def train_model(args):
    # Set device to MPS if available, otherwise CPU
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Load or extract mel-spectrogram dataset
    if args.load_db:
        X_train, y_train, num_frames_train = load_dataset(args.train_dataset)
        X_test, y_test, num_frames_test = load_dataset(args.test_dataset)
    else:
        X_train, y_train, num_frames_train = extract_melgrams(args.train_songs_list, args.multiframes, process_all_song=False)
        X_test, y_test, num_frames_test = extract_melgrams(args.test_songs_list, args.multiframes, process_all_song=False)

    print(f"Unique labels in y_train: {np.unique(y_train)}")
    print(f"Unique labels in y_test: {np.unique(y_test)}")

    # Convert numpy arrays to PyTorch tensors and ensure correct shape
    X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 96, 1366).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 96, 1366).to(device)
    Y_train = torch.tensor(np.eye(args.nb_classes)[y_train], dtype=torch.float32).to(device)
    Y_test = torch.tensor(np.eye(args.nb_classes)[y_test], dtype=torch.float32).to(device)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train, Y_train, torch.tensor(num_frames_train))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize the model
    model = MusicTaggerCRNN().to(device)

    # Define optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.train:
        # Training loop
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

                # Print information for one song in this batch
                song_index = i * args.batch_size  # Get the index of the first song in this batch
                song_path = song_paths[song_index]  # Path to the song
                song_label = y_train[song_index]  # Integer label of the song
                song_genre = genres[song_label]  # Genre that corresponds with the label

                print(f"Song Path: {song_path}, Label: {song_label}, Genre: {song_genre}")

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {epoch_loss/len(train_loader)}")

            # Save model weights every 5 epochs
            if epoch % 5 == 0:
                torch.save(model.state_dict(), args.weights_path + args.model_name + f"_epoch_{epoch}.pth")
                print(f"Saved model at epoch {epoch} to {args.weights_path}")

    # Testing loop
    if args.test:
        print("Testing the model...")
        model.eval()

        # Use DataLoader for batch processing to prevent memory issues
        test_dataset = TensorDataset(X_test, Y_test)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Initialize lists to collect results
        predictions = []
        true_labels = []

        # Disable gradient calculations during testing for efficiency
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                # Move batch data to the correct device
                batch_X = batch_X.to(device)
                batch_Y = batch_Y.to(device)

                # Forward pass to get model predictions
                outputs = model(batch_X)

                # Get predicted labels (taking the class with the highest score)
                _, predicted = torch.max(outputs, dim=1)

                # Collect predictions and true labels
                predictions.extend(predicted.cpu().numpy())
                true_labels.extend(batch_Y.argmax(dim=1).cpu().numpy())

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(true_labels, predictions)

        # Plot confusion matrix (assuming you have a plot_confusion_matrix utility function)
        plot_confusion_matrix(cnf_matrix, classes=np.arange(args.nb_classes), title='Confusion Matrix')


# Command line arguments (simplified for example)
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

    args = parser.parse_args()

     # Assume song_paths is a list of the file paths of the songs
    song_paths = open(args.train_songs_list).read().splitlines()


    train_model(args)
