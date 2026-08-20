"""
Preprocessing module for Lakh MIDI (LMD) dataset.
========================================================================

This module contains functions to preprocess the Lakh MIDI dataset for training an Transformer model.
"""

from modules import config

class LakhPreprocessor:
    """
    A class for preprocessing the Lakh MIDI dataset for training a Transformer model.

    This class takes melodies, tokenizes and encodes them, and prepares Torch friendly 
    datasets for training sequence-to-sequence models.
    """

    def __init__(self, dataset_path: str = str(config.LMD_DATASET_PATH), batch_size: int = 32):
        """
        Initializes the LakhPreprocessor with the dataset path and output path.

        Parameters:
            dataset_path (str): Path to the dataset file.
            max_melody_length (int): Maximum length of the sequences.
            batch_size (int): Size of each batch in the dataset.
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.max_melody_length = None
        self.number_of_tokens = None
