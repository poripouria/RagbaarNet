"""
Melody Generator for LSTM-OnEssen Model
========================================================================

This module contains functions to generate melodies using a trained PyTorch LSTM model for music generation tasks.
"""

import json
import numpy as np
import torch
import music21 as m21
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

        self._start_symbols = ["/"] * SEQUENCE_LENGTH
       
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

    def generate_melody(self, seed: str, num_steps: int = 500, temperature: float = 0.8):
        """Generates a melody using the trained LSTM model.
        
        Args:
            seed_sequence (list): List of integers representing the seed melody.
            length (int): Number of notes to generate.
            temperature (float): Sampling temperature for controlling randomness.
        
        Returns:
            generated_sequence (list): List of integers representing the generated melody.
        """

        self.model.eval()

        seed_list = seed.split()
        melody = seed_list.copy()
        seed_list = self._start_symbols + seed_list

        seed_idx = [self.mapping.get(symbol, 0) for symbol in seed_list]

        seed_tensor = torch.tensor(seed_idx[-SEQUENCE_LENGTH:], dtype=torch.long, device=self.device)

        for _ in range(num_steps):
            input_seq = torch.nn.functional.one_hot(seed_tensor, num_classes=self.vocab_size).float().unsqueeze(0)

            with torch.no_grad():
                output = self.model(input_seq)
                output = output.squeeze(0)

            probs = torch.softmax(output, dim=-1).cpu().numpy()
            output_int = self._sample_with_temperature(probs, temperature)

            seed_tensor = torch.cat([seed_tensor[1:], torch.tensor([output_int], device=self.device)])

            output_symbol = self.reverse_mapping.get(output_int, "?")

            if not output_symbol.isdigit() and output_symbol not in ["_", "r", "/"]:
                output_symbol = "r"

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        return melody

    def _sample_with_temperature(self, probabilities: np.ndarray, temperature: float):
        """Samples an index from a probability array reapplying softmax using temperature

        Args:
            predictions (nd.array): Array containing probabilities for each of the possible outputs.
            temperature (float): Float in interval [0, 1]. Numbers closer to 0 make the model more deterministic.
                A number closer to 1 makes the generation more unpredictable.

        Returns:
            index (int): Selected output symbol
        """
    
        probabilities = np.array(probabilities, dtype=np.float64)
        probabilities = np.clip(probabilities, 1e-10, 1.0)

        predictions = np.log(probabilities) / temperature
        exp_preds = np.exp(predictions)
        probabilities = exp_preds / np.sum(exp_preds)

        choices = range(len(probabilities))
        return np.random.choice(choices, p=probabilities)

    def save_melody(self, melody, step_duration=0.25, file_name="generated_melody.mid"):
        """Converts a melody into a MIDI file

        Args:
            melody (list of str): List of symbols representing the melody.
            step_duration (float): Duration of each time step in quarter length.
            file_name (str): Name of the MIDI file to save.

        Returns:
            None
        """

        stream = m21.stream.Stream()
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            symbol = str(symbol).strip()

            if symbol.startswith('_') and len(symbol) > 1:
                symbol = symbol[1:]

            if symbol not in ["_", "r", "/"] and not symbol.isdigit():
                symbol = "r"

            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length = step_duration * step_counter

                    try:
                        if start_symbol == "r" or start_symbol == "_":
                            event = m21.note.Rest(quarterLength=quarter_length)
                        else:
                            pitch = int(start_symbol)
                            event = m21.note.Note(pitch, quarterLength=quarter_length)
                        
                        stream.append(event)
                    except ValueError:
                        stream.append(m21.note.Rest(quarterLength=quarter_length))

                    step_counter = 1

                start_symbol = symbol
            else:
                step_counter += 1

        file_path = os.path.join(os.path.dirname(self.model_path), file_name)
        try:
            stream.write("midi", file_path)
            logger.info(f"✅ MIDI file saved successfully: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error saving MIDI: {e}")


if __name__ == "__main__":

    generator = MelodyGenerator()

    seed_exmpl = "55 _ _ 52 55 60 _ r _ _ _" * 2
    seed_exmpl2 = "67 _ 67 _ 67 _ _ 65 64 _ 64 _ 64 _ _"
    seed_exmpl3 = "67 _ _ _ _ _ 65 _ 64 _ 62 _ 60 _ _ _"

    generated_melody = generator.generate_melody(seed_exmpl3, 500, 0.8)

    print("Generated melody (as integers): %s", " ".join(generated_melody))

    generator.save_melody(generated_melody, step_duration=0.25, file_name="generated_melody3.mid")
