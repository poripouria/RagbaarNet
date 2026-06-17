"""
Train and save PyTorch model for LSTM_OnEssen.
========================================================================

This module contains functions to build, train, and save a PyTorch LSTM model
for music generation tasks.
(Special thanks to Valerio Velardo)
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.logging_setup import setup_logging
from Models.Music.LSTM_OnEssen.preprocess import generate_training_sequences, SEQUENCE_LENGTH, MAPPING_PATH

logger = setup_logging("INFO", name="Models.Music.LSTM_OnEssen.train")

INTERNAL_UNIT_SIZE = [256, 256]
LEARNING_RATE      = 0.001
EPOCHS             = 60
BATCH_SIZE         = 128
DROPOUT            = 0.1
MODEL_SAVE_PATH    = "modules/Models/Music/LSTM_OnEssen/LSTM_OnEssen.pt"

class LSTM_OnEssen(nn.Module):
    """Two-layer LSTM for symbolic music generation.

    Architecture mirrors the original Keras model:
        Input  → LSTM(256, return_seq=True) → LSTM(256) → Dropout → Linear(vocab)

    Args:
        input_size  (int):  Vocabulary size (= one-hot vector length).
        hidden_sizes (list): Hidden units for each LSTM layer.
        dropout     (float): Dropout probability applied after the second LSTM.
    """

    def __init__(self, input_size: int, hidden_sizes: list, dropout: float = 0.1):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_sizes[0],
            batch_first = True,    # input shape: (batch, seq_len, features)
        )
        self.lstm2 = nn.LSTM(
            input_size = hidden_sizes[0],
            hidden_size = hidden_sizes[1],
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_sizes[1], input_size)
        # Note: no softmax here — nn.CrossEntropyLoss expects raw logits.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_size)
        Returns:
            logits: (batch, input_size)
        """

        x, _ = self.lstm1(x)           # (batch, seq_len, hidden[0])
        x, _ = self.lstm2(x)           # (batch, seq_len, hidden[1])
        x     = x[:, -1, :]            # take last timestep → (batch, hidden[1])
        x     = self.dropout(x)
        return self.fc(x)              # (batch, vocab_size)

def setup_device() -> torch.device:
    """Return CUDA device if available, else CPU."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
    else:
        logger.warning("No GPU found — training will run on CPU.")
    return device

def build_dataloader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, vocab_size: int, device: torch.device) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader.

    Args:
        inputs  (np.ndarray): shape (N, seq_len, vocab_size), float32
        targets (np.ndarray): shape (N,), int64
        batch_size (int): mini-batch size
        vocab_size (int): Size of the vocabulary
        device (torch.device): Device to move tensors to
    Returns:
        DataLoader
    """

    # We will one-hot on the fly inside the collate function to save RAM
    def collate_fn(batch):
        inputs_batch, targets_batch = zip(*batch)
        inputs_batch = torch.stack(inputs_batch)   # (batch, seq_len)
        
        # One-hot encoding on GPU (very memory efficient)
        inputs_onehot = torch.nn.functional.one_hot(inputs_batch, num_classes=vocab_size).float()
        # shape: (batch, seq_len, vocab_size)
        
        targets_batch = torch.stack(targets_batch)
        
        return inputs_onehot.to(device, non_blocking=True), targets_batch.to(device, non_blocking=True)

    dataset = TensorDataset(
        torch.from_numpy(inputs).long(),      # (N, seq_len)
        torch.from_numpy(targets).long(),     # (N,)
    )

    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_fn,
        pin_memory = False,
        num_workers = 0
    )

def train(num_units: list = INTERNAL_UNIT_SIZE, learning_rate: float = LEARNING_RATE,
    epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
    """Train the LSTM_OnEssen model and save weights.

    Args:
        num_units     (list):  Hidden sizes for each LSTM layer.
        learning_rate (float): Adam learning rate.
        epochs        (int):   Maximum training epochs.
        batch_size    (int):   Mini-batch size.
    """

    device = setup_device()

    inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)

    with open(MAPPING_PATH, "r") as fp:
        mapping = json.load(fp)
    vocab_size = len(mapping)

    logger.info(f"Input shape: {inputs.shape} | Memory: {inputs.nbytes / 1e9:.2f} GB")
    logger.info("Vocabulary size : %d", vocab_size)
    logger.info("Training samples: %d", len(inputs))
    
    dataloader = build_dataloader(inputs, targets, batch_size, vocab_size, device)

    model = LSTM_OnEssen(
        input_size   = vocab_size,
        hidden_sizes = num_units,
        dropout      = DROPOUT,
    ).to(device)

    logger.info("Model parameters: {:,}".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    ))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6 
    )

    # Mixed precision scaler (only active on CUDA)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None

    # Early stopping state
    best_loss      = float('inf')
    patience_left  = 5
    best_state     = None

    # Epoch loop
    logger.info("Starting training on %s ...", device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', enabled=True):
                    logits = model(batch_inputs)
                    loss = criterion(logits, batch_targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(batch_inputs)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * len(batch_inputs)

        epoch_loss /= len(inputs)
        scheduler.step(epoch_loss)

        logger.info("Epoch %3d/%d — loss: %.4f — lr: %.2e",
                    epoch, epochs, epoch_loss,
                    optimizer.param_groups[0]['lr'])

        # Early stopping
        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 5
        else:
            patience_left -= 1
            logger.info("No improvement. Patience left: %d", patience_left)
            if patience_left == 0:
                logger.info("Early stopping triggered at epoch %d.", epoch)
                break

    # Save
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Restore best weights before saving
    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size':       vocab_size,
        'hidden_sizes':     num_units,
        'dropout':          DROPOUT,
        'best_loss':        best_loss,
    }, MODEL_SAVE_PATH)

    logger.info("Model saved to %s  (best loss: %.4f)", MODEL_SAVE_PATH, best_loss)


if __name__ == "__main__":

    train()
