import os
import random

"""
This script processes each genre folder within the dataset, 
shuffles the songs within each genre, and splits them into 
training and test sets based on the specified ratio.
"""

# Path to the dataset
dataset_path = "/Users/shawyan/Downloads/Data/genres_original"

# Paths for saving the train and test song lists
train_file = "train_songs_list.txt"
test_file = "test_songs_list.txt"

# Train/test split
train_ratio = 0.8

train_songs = []
test_songs = []

# Process each genre folder separately
for genre_folder in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre_folder)
    if os.path.isdir(genre_path):  # Check if directory
        all_songs_in_genre = [os.path.join(genre_path, file) for file in os.listdir(genre_path) if file.endswith(".wav")]

        # Shuffle the songs 
        random.shuffle(all_songs_in_genre)

        # Split into train and test 
        train_size = int(len(all_songs_in_genre) * train_ratio)
        genre_train_songs = all_songs_in_genre[:train_size]
        genre_test_songs = all_songs_in_genre[train_size:]

        # Add to main train and test lists
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
