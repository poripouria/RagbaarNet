"""
Melody Generator for LSTM-OnEssen Model
========================================================================

This module contains functions to generate melodies using a trained PyTorch LSTM model for music generation tasks.
"""

import json
import numpy as np
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
from utils.logging_setup import setup_logging
from Models.Music.LSTM_OnEssen.train import LSTM_OnEssen, MODEL_SAVE_PATH
from Models.Music.LSTM_OnEssen.preprocess import SEQUENCE_LENGTH, MAPPING_PATH

logger = setup_logging("INFO", name="Models.Music.LSTM_OnEssen.generator")

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies.

    Args:
        model_path (str): Path to the trained PyTorch model.
    """

    def __init__(self, model_path: str = MODEL_SAVE_PATH):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = model_path
       
        # Load mapping
        with open(MAPPING_PATH, 'r') as f:
            self.mapping = json.load(f)
        
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}
        self.vocab_size = len(self.mapping)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        # Create model and load weights
        self.model = LSTM_OnEssen(
            input_size=self.vocab_size,
            hidden_sizes=checkpoint['hidden_sizes'],
            dropout=checkpoint.get('dropout', 0.1)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        logger.info(f"✅ Model loaded successfully from {model_path} (Best loss: {checkpoint.get('best_loss', 'N/A')})")

    def generate(self, seed_sequence: list = None, length: int = 500, 
                 temperature: float = 1.0, top_k: int = 12, top_p: float = 0.9):
        """Generates a melody using the trained LSTM model.
        
        Args:
            seed_sequence (list): List of integers representing the seed melody.
            length (int): Number of notes to generate.
            temperature (float): Sampling temperature for controlling randomness.
            top_k (int): Number of top-k candidates to consider for sampling.
        Returns:
            generated_sequence (list): List of integers representing the generated melody.
        """
        
        self.model.eval()

        if seed_sequence is None or len(seed_sequence) == 0:
            seed_sequence = [0] * SEQUENCE_LENGTH

        if isinstance(seed_sequence[0], str):
            seed_sequence = [self.mapping.get(s, 0) for s in seed_sequence]

        seed = torch.tensor(seed_sequence[-SEQUENCE_LENGTH:], dtype=torch.long, device=self.device)
        generated = seed.tolist()

        with torch.no_grad():
            for i in range(length):
                input_seq = torch.nn.functional.one_hot(seed, num_classes=self.vocab_size).float().unsqueeze(0)

                output = self.model(input_seq)
                output = output.squeeze(0) / temperature

                # Nucleus + Top-k Sampling (بهترین روش برای موسیقی)
                probs = torch.softmax(output, dim=-1)
                
                # Top-k
                topk_probs, topk_indices = torch.topk(probs, top_k)
                topk_probs = topk_probs / topk_probs.sum()

                # Top-p (Nucleus)
                sorted_probs, sorted_indices = torch.sort(topk_probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                topk_probs[sorted_indices_to_remove] = 0
                topk_probs = topk_probs / topk_probs.sum()

                next_note = torch.multinomial(topk_probs, num_samples=1).item()
                next_note = topk_indices[next_note].item()

                generated.append(next_note)
                seed = torch.cat([seed[1:], torch.tensor([next_note], device=self.device)])

                if next_note == self.mapping.get("/", 0):
                    break

        generated_notes = [self.reverse_mapping.get(idx, "?") for idx in generated]
        return generated_notes


if __name__ == "__main__":

    generator = MelodyGenerator()

    seed_exmpl = ["55", "_", "_", "52", "55", "60", "_"] * 3

    generated_melody = generator.generate(seed_exmpl, 200, 1.1, 12, 0.92)

    print("Generated melody (as integers):", " ".join(generated_melody))
