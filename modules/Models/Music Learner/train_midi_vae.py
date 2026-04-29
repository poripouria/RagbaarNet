from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Ensure we can import from modules/utils (even though folder names contain spaces)
THIS_DIR = os.path.dirname(__file__)
MODULES_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CODE_ROOT = os.path.abspath(os.path.join(MODULES_DIR, ".."))
sys.path.append(MODULES_DIR)

from utils.logging_setup import setup_logging  # noqa: E402
from utils.midi_processing import (  # noqa: E402
    PianoRollConfig,
    extract_random_segment,
    midi_to_pianoroll,
)

from music_vae import MusicVAE, vae_loss  # noqa: E402

logger = setup_logging("INFO", name="train_midi_vae")


class LakhMIDIPianoRollDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cfg: PianoRollConfig,
        *,
        num_samples: int,
        seed: int = 0,
        limit_files: int | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.seed = int(seed)

        midi_files: list[str] = []
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith((".mid", ".midi")):
                    midi_files.append(os.path.join(root, f))
        midi_files.sort()

        if limit_files is not None:
            midi_files = midi_files[: int(limit_files)]

        if not midi_files:
            raise RuntimeError(f"No MIDI files found in: {data_dir}")

        self.midi_files = midi_files
        self._cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return self.num_samples

    def _get_roll(self, midi_path: str) -> np.ndarray:
        if midi_path in self._cache:
            return self._cache[midi_path]
        roll = midi_to_pianoroll(
            midi_path,
            steps_per_beat=self.cfg.steps_per_beat,
            ignore_drums=self.cfg.ignore_drums,
        )
        self._cache[midi_path] = roll
        return roll

    def __getitem__(self, idx: int) -> torch.Tensor:
        midi_path = self.midi_files[idx % len(self.midi_files)]
        # Make sampling deterministic per idx (useful with multiple workers).
        rng = np.random.default_rng(self.seed + int(idx))
        roll = self._get_roll(midi_path)
        seg = extract_random_segment(roll, segment_steps=self.cfg.segment_steps, rng=rng)
        x = seg.reshape(-1).astype(np.float32)
        return torch.from_numpy(x)


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple VAE on LAKH MIDI piano-roll segments")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(CODE_ROOT, "Dataset", "Lakh MIDI", "sample_midi_files"),
        help="Path containing .mid files (recursively).",
    )
    parser.add_argument("--out_dir", type=str, default=os.path.join(CODE_ROOT, "assets", "test", "midi_vae"))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--latent_dim", type=int, default=64)

    parser.add_argument("--steps_per_beat", type=int, default=4)
    parser.add_argument("--segment_bars", type=int, default=4)
    parser.add_argument("--beats_per_bar", type=int, default=4)
    parser.add_argument("--include_drums", action="store_true")

    parser.add_argument(
        "--num_samples_per_epoch",
        type=int,
        default=1024,
        help="How many random segments to draw per epoch (dataset length).",
    )
    parser.add_argument("--limit_files", type=int, default=None, help="Limit number of MIDI files (debug).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    cfg = PianoRollConfig(
        steps_per_beat=args.steps_per_beat,
        segment_bars=args.segment_bars,
        beats_per_bar=args.beats_per_bar,
        ignore_drums=not args.include_drums,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Data dir: {args.data_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = LakhMIDIPianoRollDataset(
        args.data_dir,
        cfg,
        num_samples=args.num_samples_per_epoch,
        seed=args.seed,
        limit_files=args.limit_files,
    )

    input_dim = cfg.segment_steps * 128
    model = MusicVAE(input_dim=input_dim, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    _save_json(
        os.path.join(args.out_dir, "config.json"),
        {
            "args": vars(args),
            "pianoroll": asdict(cfg),
            "input_dim": input_dim,
        },
    )

    logger.info(
        f"Training: files={len(dataset.midi_files)} segment_steps={cfg.segment_steps} input_dim={input_dim}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n_batches = 0

        for x in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon_logits, mu, logvar = model(x)
            loss, parts = vae_loss(recon_logits, x, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            recon_sum += float(parts["recon"].cpu())
            kl_sum += float(parts["kl"].cpu())
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        avg_recon = recon_sum / max(n_batches, 1)
        avg_kl = kl_sum / max(n_batches, 1)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}"
        )

        ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": asdict(cfg),
                "input_dim": input_dim,
                "latent_dim": args.latent_dim,
            },
            ckpt_path,
        )

    logger.info(f"Done. Checkpoints saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
