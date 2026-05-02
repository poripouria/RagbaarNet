from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import mido

# Ensure we can import from modules/utils (even though folder names contain spaces)
THIS_DIR = os.path.dirname(__file__)
MODULES_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CODE_ROOT = os.path.abspath(os.path.join(MODULES_DIR, ".."))
sys.path.append(MODULES_DIR)

from utils.logging_setup import setup_logging  # noqa: E402
from utils.midi_processing import (  # noqa: E402
    augment_pianoroll,
    PianoRollConfig,
    extract_random_segment,
    midi_to_pianoroll,
    transpose_pianoroll,
)

from music_vae import MusicVAE, vae_loss  # noqa: E402

logger = setup_logging("INFO", name="train_midi_vae")


def _split_files(midi_files: list[str], val_ratio: float = 0.2) -> tuple[list[str], list[str]]:
    if not midi_files:
        return [], []
    split_idx = max(1, int(len(midi_files) * (1.0 - val_ratio)))
    if split_idx >= len(midi_files):
        split_idx = max(1, len(midi_files) - 1)
    return midi_files[:split_idx], midi_files[split_idx:]


def _scan_midi_files(data_dir: str, limit_files: int | None = None) -> list[str]:
    midi_files: list[str] = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith((".mid", ".midi")):
                midi_files.append(os.path.join(root, f))
    midi_files.sort()
    if limit_files is not None:
        midi_files = midi_files[: int(limit_files)]
    return midi_files


def _filter_readable_midi_files(midi_files: list[str], cfg: PianoRollConfig) -> list[str]:
    valid_files: list[str] = []
    skipped_files: list[str] = []
    for midi_path in midi_files:
        try:
            _ = midi_to_pianoroll(
                midi_path,
                steps_per_beat=cfg.steps_per_beat,
                ignore_drums=cfg.ignore_drums,
            )
            valid_files.append(midi_path)
        except Exception as exc:
            skipped_files.append(midi_path)
            logger.warning(f"Skipping unreadable MIDI file: {midi_path} ({exc.__class__.__name__})")

    if not valid_files:
        raise RuntimeError("No readable MIDI files found")

    if skipped_files:
        logger.info(f"Skipped {len(skipped_files)} unreadable MIDI file(s) during dataset scan")
    return valid_files


def _save_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _binary_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = pred.float()
    target = target.float()
    tp = torch.sum(pred * target)
    fp = torch.sum(pred * (1.0 - target))
    fn = torch.sum((1.0 - pred) * target)
    tn = torch.sum((1.0 - pred) * (1.0 - target))

    eps = torch.tensor(1e-8, device=pred.device)
    accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    pred_density = torch.mean(pred)
    target_density = torch.mean(target)
    density_error = torch.abs(pred_density - target_density)
    return {
        "accuracy": float(accuracy.detach().cpu()),
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "f1": float(f1.detach().cpu()),
        "pred_density": float(pred_density.detach().cpu()),
        "target_density": float(target_density.detach().cpu()),
        "density_error": float(density_error.detach().cpu()),
    }


def _decode_probs_to_roll(
    probs: np.ndarray,
    *,
    strategy: str = "adaptive_topk",
    threshold: float = 0.5,
    top_k: int = 4,
    reference_roll: np.ndarray | None = None,
) -> np.ndarray:
    """Convert probabilities to a binary roll using a musically safer strategy."""
    if probs.ndim != 2 or probs.shape[1] != 128:
        raise ValueError("probs must have shape (T, 128)")

    if strategy == "threshold":
        return (probs >= threshold).astype(np.float32)

    decoded = np.zeros_like(probs, dtype=np.float32)
    for t in range(probs.shape[0]):
        row = probs[t]
        if strategy == "adaptive_topk" and reference_roll is not None:
            k = int(reference_roll[t].sum())
        else:
            k = int(top_k)
        k = max(1, min(k, 12))
        indices = np.argpartition(row, -k)[-k:]
        decoded[t, indices] = 1.0
    return decoded


def _pianoroll_to_midi(roll: np.ndarray, output_path: str, *, steps_per_beat: int, tempo_bpm: int = 120) -> None:
    """Serialize a binary piano-roll to a simple single-track MIDI file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    roll = np.asarray(roll)
    if roll.ndim != 2 or roll.shape[1] != 128:
        raise ValueError("roll must have shape (T, 128)")

    ticks_per_beat = 480
    ticks_per_step = max(1, int(ticks_per_beat / steps_per_beat))
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo_bpm), time=0))

    active = set()
    for step in range(roll.shape[0]):
        current = {note for note in range(128) if roll[step, note] > 0.5}
        note_offs = sorted(active - current)
        note_ons = sorted(current - active)

        messages: list[mido.Message] = []
        for note in note_offs:
            messages.append(mido.Message("note_off", note=int(note), velocity=0, time=0))
        for note in note_ons:
            messages.append(mido.Message("note_on", note=int(note), velocity=64, time=0))

        if messages:
            messages[-1].time = ticks_per_step
            track.extend(messages)
        else:
            track.append(mido.MetaMessage("marker", text="", time=ticks_per_step))

        active = current

    # Flush any sustained notes at the end.
    for note in sorted(active):
        track.append(mido.Message("note_off", note=int(note), velocity=0, time=0))

    mid.save(output_path)


def _evaluate(model: MusicVAE, loader: DataLoader, device: torch.device, beta: float) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    recon_sum = 0.0
    kl_sum = 0.0
    accuracy_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    pred_density_sum = 0.0
    target_density_sum = 0.0
    density_error_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            recon_logits, mu, logvar = model(x)
            positive = x.sum()
            negative = x.numel() - positive
            pos_weight = torch.clamp(negative / (positive + 1e-6), min=1.0, max=40.0)
            loss, parts = vae_loss(recon_logits, x, mu, logvar, beta=beta, pos_weight=pos_weight)
            probs = torch.sigmoid(recon_logits)
            preds = (probs >= 0.5).float()
            batch_metrics = _binary_metrics(preds, x)

            loss_sum += float(loss.detach().cpu())
            recon_sum += float(parts["recon"].cpu())
            kl_sum += float(parts["kl"].cpu())
            accuracy_sum += batch_metrics["accuracy"]
            precision_sum += batch_metrics["precision"]
            recall_sum += batch_metrics["recall"]
            f1_sum += batch_metrics["f1"]
            pred_density_sum += batch_metrics["pred_density"]
            target_density_sum += batch_metrics["target_density"]
            density_error_sum += batch_metrics["density_error"]
            n_batches += 1

    denom = max(n_batches, 1)
    return {
        "loss": loss_sum / denom,
        "recon": recon_sum / denom,
        "kl": kl_sum / denom,
        "accuracy": accuracy_sum / denom,
        "precision": precision_sum / denom,
        "recall": recall_sum / denom,
        "f1": f1_sum / denom,
        "pred_density": pred_density_sum / denom,
        "target_density": target_density_sum / denom,
        "density_error": density_error_sum / denom,
    }


def _effective_beta(base_beta: float, epoch: int, warmup_epochs: int, cycle_epochs: int) -> float:
    if cycle_epochs > 0:
        cycle_epoch = ((epoch - 1) % cycle_epochs) + 1
        progress = min(1.0, cycle_epoch / max(1, warmup_epochs))
        return base_beta * progress
    return base_beta * min(1.0, epoch / max(1, warmup_epochs))


class LakhMIDIPianoRollDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        cfg: PianoRollConfig,
        *,
        num_samples: int,
        seed: int = 0,
        limit_files: int | None = None,
        transpose_range: int = 5,
        note_dropout: float = 0.0,
        time_mask_prob: float = 0.0,
        time_mask_width: int = 0,
        midi_files: list[str] | None = None,
        validate_files: bool = True,
    ) -> None:
        self.data_dir = data_dir
        self.cfg = cfg
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.transpose_range = int(transpose_range)
        self.note_dropout = float(note_dropout)
        self.time_mask_prob = float(time_mask_prob)
        self.time_mask_width = int(time_mask_width)

        if midi_files is None:
            midi_files = _scan_midi_files(data_dir, limit_files=limit_files)

        if not midi_files:
            raise RuntimeError(f"No MIDI files found in: {data_dir}")

        self.midi_files = _filter_readable_midi_files(midi_files, self.cfg) if validate_files else list(midi_files)
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
        if self.transpose_range > 0:
            semitone_shift = int(rng.integers(-self.transpose_range, self.transpose_range + 1))
            roll = transpose_pianoroll(roll, semitone_shift)
        if self.note_dropout > 0.0 or self.time_mask_prob > 0.0:
            roll = augment_pianoroll(
                roll,
                rng,
                note_dropout=self.note_dropout,
                time_mask_prob=self.time_mask_prob,
                time_mask_width=self.time_mask_width,
            )
        seg = extract_random_segment(roll, segment_steps=self.cfg.segment_steps, rng=rng)
        x = seg.reshape(-1).astype(np.float32)
        return torch.from_numpy(x)


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
    parser.add_argument("--kl_warmup_epochs", type=int, default=5)
    parser.add_argument("--kl_cycle_epochs", type=int, default=0, help="If > 0, restart KL warmup every N epochs.")

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
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--save_samples_every", type=int, default=1)
    parser.add_argument("--transpose_range", type=int, default=5)
    parser.add_argument("--note_dropout", type=float, default=0.02)
    parser.add_argument("--time_mask_prob", type=float, default=0.05)
    parser.add_argument("--time_mask_width", type=int, default=2)
    parser.add_argument("--decode_strategy", type=str, default="adaptive_topk", choices=["adaptive_topk", "topk", "threshold"])
    parser.add_argument("--decode_threshold", type=float, default=0.5)
    parser.add_argument("--decode_top_k", type=int, default=4)

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

    midi_files = _scan_midi_files(args.data_dir, limit_files=args.limit_files)
    midi_files = _filter_readable_midi_files(midi_files, cfg)

    train_files, val_files = _split_files(midi_files, val_ratio=args.val_ratio)
    if not val_files:
        val_files = train_files[:1]
    train_dataset = LakhMIDIPianoRollDataset(
        args.data_dir,
        cfg,
        num_samples=args.num_samples_per_epoch,
        seed=args.seed,
        transpose_range=args.transpose_range,
        note_dropout=args.note_dropout,
        time_mask_prob=args.time_mask_prob,
        time_mask_width=args.time_mask_width,
        midi_files=train_files,
        validate_files=False,
    )
    val_dataset = LakhMIDIPianoRollDataset(
        args.data_dir,
        cfg,
        num_samples=max(1, min(64, len(val_files) * 4)),
        seed=args.seed + 999,
        transpose_range=0,
        note_dropout=0.0,
        time_mask_prob=0.0,
        time_mask_width=0,
        midi_files=val_files,
        validate_files=False,
    )

    input_dim = cfg.segment_steps * 128
    model = MusicVAE(sequence_steps=cfg.segment_steps, latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    _save_json(
        os.path.join(args.out_dir, "config.json"),
        {
            "args": vars(args),
            "pianoroll": asdict(cfg),
            "input_dim": input_dim,
            "augmentation": {
                "transpose_range": args.transpose_range,
                "note_dropout": args.note_dropout,
                "time_mask_prob": args.time_mask_prob,
                "time_mask_width": args.time_mask_width,
            },
            "decoding": {
                "strategy": args.decode_strategy,
                "threshold": args.decode_threshold,
                "top_k": args.decode_top_k,
            },
            "kl_schedule": {
                "beta": args.beta,
                "warmup_epochs": args.kl_warmup_epochs,
                "cycle_epochs": args.kl_cycle_epochs,
            },
        },
    )

    logger.info(
        f"Training: files={len(train_files)} val_files={len(val_files)} segment_steps={cfg.segment_steps} input_dim={input_dim}"
    )

    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        n_batches = 0
        effective_beta = _effective_beta(args.beta, epoch, args.kl_warmup_epochs, args.kl_cycle_epochs)

        for x in train_loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon_logits, mu, logvar = model(x)
            positive = x.sum()
            negative = x.numel() - positive
            pos_weight = torch.clamp(negative / (positive + 1e-6), min=1.0, max=40.0)
            loss, parts = vae_loss(recon_logits, x, mu, logvar, beta=effective_beta, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.detach().cpu())
            recon_sum += float(parts["recon"].cpu())
            kl_sum += float(parts["kl"].cpu())
            n_batches += 1

        avg_loss = loss_sum / max(n_batches, 1)
        avg_recon = recon_sum / max(n_batches, 1)
        avg_kl = kl_sum / max(n_batches, 1)

        val_metrics = _evaluate(model, val_loader, device, beta=effective_beta)

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | beta={effective_beta:.4f} train_loss={avg_loss:.4f} train_recon={avg_recon:.4f} train_kl={avg_kl:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} val_recon={val_metrics['recon']:.4f} val_kl={val_metrics['kl']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_density={val_metrics['pred_density']:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "train_recon": avg_recon,
                "train_kl": avg_kl,
                "effective_beta": effective_beta,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
        )

        _save_json(os.path.join(args.out_dir, "history.json"), history)

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

        if epoch % max(1, args.save_samples_every) == 0:
            sample_x = next(iter(val_loader)).to(device)
            model.eval()
            with torch.no_grad():
                sample_logits, _, _ = model(sample_x)
                sample_probs = torch.sigmoid(sample_logits)
                sample_pred = []
                for i in range(sample_probs.shape[0]):
                    reference = sample_x[i].detach().cpu().numpy().reshape(cfg.segment_steps, 128)
                    decoded = _decode_probs_to_roll(
                        sample_probs[i].detach().cpu().numpy().reshape(cfg.segment_steps, 128),
                        strategy=args.decode_strategy,
                        threshold=args.decode_threshold,
                        top_k=args.decode_top_k,
                        reference_roll=reference,
                    )
                    sample_pred.append(decoded.reshape(-1))
                sample_pred = torch.from_numpy(np.stack(sample_pred).astype(np.float32)).to(device)

            sample_x_np = sample_x[0].detach().cpu().numpy().reshape(cfg.segment_steps, 128)
            sample_pred_np = sample_pred[0].detach().cpu().numpy().reshape(cfg.segment_steps, 128)
            sample_probs_np = sample_probs[0].detach().cpu().numpy().reshape(cfg.segment_steps, 128)

            np.savez_compressed(
                os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}.npz"),
                input_roll=sample_x_np,
                recon_roll=sample_pred_np,
                recon_probs=sample_probs_np,
            )
            _pianoroll_to_midi(
                sample_x_np,
                os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}_input.mid"),
                steps_per_beat=cfg.steps_per_beat,
            )
            _pianoroll_to_midi(
                sample_pred_np,
                os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}_recon.mid"),
                steps_per_beat=cfg.steps_per_beat,
            )

    logger.info(f"Done. Checkpoints saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
