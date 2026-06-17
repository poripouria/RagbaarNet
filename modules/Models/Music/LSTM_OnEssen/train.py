"""
Train and save TF model for LSTM_OnEssen.
========================================================================

This module contains functions to build, train, and save a TensorFlow model for the LSTM_OnEssen architecture. 
The model is designed for music generation tasks.
(Special thanks to Valerio Velardo)
"""

import tensorflow as tf
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.logging_setup import setup_logging
from Models.Music.LSTM_OnEssen.preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH

logger = setup_logging("INFO", name="Models.Music.LSTM_OnEssen.train")

INTERNAL_UNIT_SIZE = [256, 256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 90
BATCH_SIZE = 64
MODEL_SAVE_PATH = "modules/Models/Music/LSTM-OnEssen/LSTM_OnEssen.h5"


def build_model(output_units: int, num_units: list, loss: str, learning_rate: float):
    """Builds and compiles model

    Args:
        output_units (int): Number of output units
        num_units (list of int): Number of units in hidden layers
        loss (str): Type of loss function to use
        learning_rate (float): Learning rate to apply

    Returns:
        model (tf model): Where the magic happens :D
    """

    # create the model architecture
    model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, output_units)),
            tf.keras.layers.LSTM(
                num_units[0],
                return_sequences=True,
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            tf.keras.layers.LSTM(
                num_units[1],
                dropout=0.1,
                recurrent_dropout=0.1
            ),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(output_units, activation="softmax")
    ])
    logger.info(f"Model architecture built with {num_units} hidden units and {output_units} output units.")

    # compile model
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=["accuracy"])
    logger.info(f"Model compiled with loss '{loss}' and learning rate {learning_rate}.")

    model.summary()

    return model

def train(num_units: list = INTERNAL_UNIT_SIZE, loss: str = LOSS, learning_rate: float = LEARNING_RATE):
    """Train and save TF model.

    Args:
        output_units (int): Number of output units
        num_units (list of int): Number of units in hidden layers
        loss (str): Type of loss function to use
        learning_rate (float): Learning rate to apply
    """

    # generate the training sequences
    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)  # X, y for training

    # build the network
    with open(MAPPING_PATH, "r") as fp:
        data = json.load(fp)
    output_units = len(data.keys())

    model = build_model(output_units, num_units, loss, learning_rate)

    # train the model
    logger.info("Starting model training...")
    model.fit(inputs, targets, epochs=EPOCHS, batch_size=BATCH_SIZE)

    # save the model
    model.save(MODEL_SAVE_PATH)


if __name__ == "__main__":

    train()
