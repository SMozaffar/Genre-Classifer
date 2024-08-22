import os
import random

# Path to the dataset
dataset_path = "/Users/shawyan/Downloads/Data/genres_original"
train_file = "train_songs_list.txt"
test_file = "test_songs_list.txt"

# Ratio of training and testing data (e.g., 80% train, 20% test)
train_ratio = 0.8

# Initialize lists for train and test songs
train_songs = []
test_songs = []

# Process each genre folder separately
for genre_folder in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre_folder)
    if os.path.isdir(genre_path):  # Check if it is a directory
        all_songs_in_genre = [os.path.join(genre_path, file) for file in os.listdir(genre_path) if file.endswith(".wav")]

        # Shuffle the songs in this genre
        random.shuffle(all_songs_in_genre)

        # Split into train and test for this genre
        train_size = int(len(all_songs_in_genre) * train_ratio)
        genre_train_songs = all_songs_in_genre[:train_size]
        genre_test_songs = all_songs_in_genre[train_size:]

        # Add to the main train and test lists
        train_songs.extend(genre_train_songs)
        test_songs.extend(genre_test_songs)

# Write train_songs_list.txt
with open(train_file, 'w') as f:
    for song in train_songs:
        f.write(f"{song}\n")

# Write test_songs_list.txt
with open(test_file, 'w') as f:
    for song in test_songs:
        f.write(f"{song}\n")

print(f"Train list saved to {train_file}")
print(f"Test list saved to {test_file}")
