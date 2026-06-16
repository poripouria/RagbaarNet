"""
Preprocessing module for Essen music dataset (Folk songs).
========================================================================

This module contains functions to preprocess the Essen music dataset for training an LSTM model. 
"""

import json
import music21 as m21
import numpy as np
# from tensorflow import keras
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.logging_setup import setup_logging

logger = setup_logging("INFO", name="Models.Music.LSTM-OnEssen.preprocess")

KERN_DATASET_PATH = "Dataset/KernScores/essen/europa/deutschl"

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

    # go through all the files in dataset and load them with music21
    for path, subdirs, files in os.walk(dataset_path):
        for file in files:

            # consider only kern files
            if file[-3:] == "krn":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)

    return songs

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

    # Filter the songs that have non-acceptable duration
    
    # Encode the songs into a suitable format for LSTM (time-series sequences)

    # Save the songs to a text file

if __name__ == "__main__":

    songs = load_songs_in_kern(KERN_DATASET_PATH)
    logger.info(f"Loaded {len(songs)} songs from the dataset.")

    test_song = songs[0]
    test_song.show()

