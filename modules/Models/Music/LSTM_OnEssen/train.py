import tensorflow as tf
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.logging_setup import setup_logging
from modules.Models.Music.LSTM_OnEssen.preprocess import generate_training_sequences, SEQUENCE_LENGTH

