"""
Preprocessing module for Essen music dataset (Folk songs).
========================================================================

This module contains functions to preprocess the Essen music dataset for training an LSTM model.
(Special thanks to Valerio Velardo)
"""

import json
import music21 as m21
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.logging_setup import setup_logging

logger = setup_logging("INFO", name="Models.Music.LSTM_OnEssen.preprocess")

KERN_DATASET_PATH = "Dataset/KernScores/essen/europa/deutschl"
SAVE_DIR = "modules/Models/Music/LSTM_OnEssen/dataset"
SINGLE_FILE_DATASET_PATH = "modules/Models/Music/LSTM_OnEssen/single_file_dataset"
MAPPING_PATH = "modules/Models/Music/LSTM_OnEssen/mapping.json"
ACCEPTABLE_DURATIONS = [ # durations are expressed in quarter length
    0.25,   # 16th note
    0.5,    # 8th note
    0.75,       # dotted 8th note
    1.0,    # quarter note
    1.5,        # dotted quarter note
    2,      # half note
    3,          # dotted half note 
    4       # whole note
]
SEQUENCE_LENGTH = 64

def load_songs_in_kern(dataset_path: str) -> list:
    """Loads all kern pieces in dataset using music21.

    Args:
        dataset_path (str): Path to dataset

    Returns:
        songs (list of m21 streams): List containing all pieces
    """

    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path '{dataset_path}' does not exist.")
        return []

    songs = []

    # Go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # Consider only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs

def has_acceptable_durations(song: m21.stream.Stream, acceptable_durations: list) -> bool:
    """Boolean routine that returns True if piece has all acceptable duration, False otherwise.

    Args:
        song (m21 stream): The music21 stream to check
        acceptable_durations (list): List of acceptable duration in quarter length

    Returns:
        bool: True if all durations are acceptable, False otherwise
    """

    for note in song.flatten().notesAndRests:
        if note.duration.quarterLength not in acceptable_durations:
            return False
    return True

def transpose(song: m21.stream.Stream) -> m21.stream.Stream:
    """Transposes song to C maj/A min

    Args:
        piece: Piece to transpose

    Returns:
        transposed_song: Transposed version of the input piece
    """

    # Get key from the song
    parts = song.getElementsByClass(m21.stream.Part)
    key = parts[0].getElementsByClass(m21.stream.Measure)[0][4]

    # Estimate key using music21
    if not isinstance(key, m21.key.Key):
        key = song.analyze("key")

    # Get interval for transposition
    if key.mode == "major":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
    elif key.mode == "minor":
        interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

    # Transpose song by calculated interval
    transposed_song = song.transpose(interval)
    return transposed_song

def encode_song(song: m21.stream.Stream, time_step: float=0.25) -> str:
    """Converts a score into a time-series-like music representation. Each item in the encoded list represents 'min_duration'
    quarter lengths. The symbols used at each step are: integers for MIDI notes, 'r' for representing a rest, and '_'
    for representing notes/rests that are carried over into a new time step. Here's a sample encoding:

        ["r", "_", "60", "_", "_", "_", "72" "_"]

    Args:
        song (m21 stream): Piece to encode
        time_step (float): Duration of each time step in quarter length
    Returns:
        encoded_song (str): Encoded song as a string
    """

    encoded_song = []

    for event in song.flatten().notesAndRests:

        # Handle notes
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        # Handle rests
        elif isinstance(event, m21.note.Rest):
            symbol = "r"

        # Convert the note/rest into time series notation
        steps = int(event.duration.quarterLength / time_step)
        for step in range(steps):

            # If it's the first time we see a note/rest, let's encode it.
            # Otherwise, it means we're carrying the same symbol in a new time step
            if step == 0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    # Cast encoded song to str
    encoded_song = " ".join(map(str, encoded_song))

    return encoded_song

def preprocess(dataset_path: str):
    """
    Preprocess the Essen music dataset.

    Args:
        dataset_path (str): Path to the raw Essen music dataset.

    Returns:
        preprocessed_data (list): A list of preprocessed sequences ready for LSTM training.
    """

    # Load the raw dataset
    songs = load_songs_in_kern(dataset_path)
    logger.info(f"Loaded {len(songs)} songs.")

    for i, song in enumerate(songs):

        # Filter the songs that have non-acceptable duration
        if not has_acceptable_durations(song, ACCEPTABLE_DURATIONS):
            logger.info(f"Song {i} has non-acceptable durations. Skipping.")
            continue

        # Transpose the songs to C maj/A min
        song = transpose(song)
        
        # Encode the songs into a suitable format for LSTM (time-series sequences)
        encoded_song = encode_song(song)

        # Save the songs to a text file
        save_path = os.path.join(SAVE_DIR, "song "+str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)

        if i % 10 == 0:
            logger.info(f"Song {i} out of {len(songs)} processed")

    logger.info("Preprocessing completed.")

def load(file_path: str) -> list:
    """Loads preprocessed data from a text file.

    Args:
        file_path (str): Path to the preprocessed data file.
    Returns:
        song (str): The preprocessed song as a string.
    """
    
    with open(file_path, "r") as fp:
        song = fp.read()
    return song

def create_single_file_dataset(dataset_path: str, file_dataset_path: str, sequence_length: int) -> str:
    """Generates a file collating all the encoded songs and adding new piece delimiters.

    Args:
        dataset_path (str): Path to folder containing the encoded songs
        file_dataset_path (str): Path to file for saving songs in single file
        sequence_length (int):  # of time steps to be considered for training

    Returns:
        songs (str): String containing all songs in dataset + delimiters
    """

    new_song_delimiter = "/ " * sequence_length
    songs = ""

    # Load encoded songs and add delimiters
    for path, _, files in os.walk(dataset_path):
        for file in files:
            file_path = os.path.join(path, file)
            song = load(file_path)
            songs = songs + song + " " + new_song_delimiter

    # Remove empty space from last character of string
    songs = songs[:-1]

    # Save string that contains all the dataset
    with open(file_dataset_path, "w") as fp:
        fp.write(songs)

    return songs

def create_mapping(songs: str, mapping_path: str):
    """Creates a json file that maps the symbols in the song dataset onto integers

    Args:
        songs (str): String with all songs
        mapping_path (str): Path where to save mapping
    
    Returns:
        None
    """
    mappings = {}

    # Identify the vocabulary
    songs = songs.split()
    vocabulary = list(set(songs))

    # Create mappings
    for i, symbol in enumerate(vocabulary):
        mappings[symbol] = i

    # Save vocabulary to a json file
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)

def map_songs_to_int(songs: str) -> list:
    """Maps the songs from string format to int format using the mapping created by create_mapping.

    Args:
        songs (str): String with all songs
    Returns:
        int_songs (list): List of integers representing the songs
    """

    int_songs = []

    # Load mappings
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)

    # Transform songs string to list
    songs = songs.split()

    # Map songs to int
    for symbol in songs:
        int_songs.append(mappings[symbol])

    return int_songs

def generate_training_sequences(sequence_length):
    """Create input and output data samples for training. Each sample is a sequence.

    Args:
        sequence_length (int): Length of each sequence. With a quantisation at 16th notes, 64 notes equates to 4 bars
    Returns:
        inputs (ndarray): Training inputs
        targets (ndarray): Training targets
    """

    # Load songs and map them to int
    songs = load(SINGLE_FILE_DATASET_PATH)
    int_songs = map_songs_to_int(songs)

    inputs = []
    targets = []

    # Generate the training sequences
    num_sequences = len(int_songs) - sequence_length
    for i in range(num_sequences):
        inputs.append(int_songs[i:i+sequence_length])
        targets.append(int_songs[i+sequence_length])

    # One-hot encode the sequences
    vocabulary_size = len(set(int_songs))
    # Inputs size: (# of sequences, sequence length, vocabulary size)
    inputs = tf.keras.utils.to_categorical(inputs, num_classes=vocabulary_size, dtype=np.uint8)
    targets = np.array(targets)

    print(f"There are {len(inputs)} sequences.")

    return inputs, targets


if __name__ == "__main__":

    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET_PATH, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
