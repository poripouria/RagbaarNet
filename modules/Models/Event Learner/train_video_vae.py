from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

THIS_DIR = os.path.dirname(__file__)
MODULES_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
CODE_ROOT = os.path.abspath(os.path.join(MODULES_DIR, ".."))
sys.path.append(MODULES_DIR)

from Segmentation.Segmentor import Segmentor  # noqa: E402
from utils.logging_setup import setup_logging  # noqa: E402
from event_vae import VideoClipVAE, vae_loss  # noqa: E402

logger = setup_logging("INFO", name="train_video_vae")


def _split_files(video_files: list[str], val_ratio: float = 0.2) -> tuple[list[str], list[str]]:
	if not video_files:
		return [], []
	split_idx = max(1, int(len(video_files) * (1.0 - val_ratio)))
	if split_idx >= len(video_files):
		split_idx = max(1, len(video_files) - 1)
	return video_files[:split_idx], video_files[split_idx:]


def _scan_video_files(data_dir: str, limit_files: int | None = None) -> list[str]:
	video_files: list[str] = []
	for root, _, files in os.walk(data_dir):
		for file_name in files:
			if file_name.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
				video_files.append(os.path.join(root, file_name))
	video_files.sort()
	if limit_files is not None:
		video_files = video_files[: int(limit_files)]
	return video_files


def _save_json(path: str, payload: dict) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as file_handle:
		json.dump(payload, file_handle, indent=2)


def _effective_beta(base_beta: float, epoch: int, warmup_epochs: int, cycle_epochs: int) -> float:
	if cycle_epochs > 0:
		cycle_epoch = ((epoch - 1) % cycle_epochs) + 1
		progress = min(1.0, cycle_epoch / max(1, warmup_epochs))
		return base_beta * progress
	return base_beta * min(1.0, epoch / max(1, warmup_epochs))


def _resize_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
	return cv2.resize(frame, (frame_size, frame_size), interpolation=cv2.INTER_AREA)


def _build_segmentation_palette() -> np.ndarray:
	palette = np.zeros((256, 3), dtype=np.uint8)
	palette[0] = [0, 0, 0]
	palette[255] = [0, 0, 0]

	cityscapes_colors = {
		0: [128, 64, 128],
		1: [244, 35, 232],
		2: [70, 70, 70],
		3: [102, 102, 156],
		4: [190, 153, 153],
		5: [153, 153, 153],
		6: [250, 170, 30],
		7: [220, 220, 0],
		8: [107, 142, 35],
		9: [152, 251, 152],
		10: [70, 130, 180],
		11: [220, 20, 60],
		12: [255, 0, 0],
		13: [0, 0, 142],
		14: [0, 0, 70],
		15: [0, 60, 100],
		16: [0, 80, 100],
		17: [0, 0, 230],
		18: [119, 11, 32],
		19: [160, 160, 160],
		20: [230, 150, 140],
		21: [128, 128, 128],
		22: [0, 0, 90],
		23: [0, 0, 110],
		24: [180, 165, 180],
		25: [150, 100, 100],
		26: [150, 120, 90],
		27: [153, 153, 153],
		28: [81, 0, 81],
		29: [111, 74, 0],
		30: [81, 81, 81],
	}
	for class_id, color in cityscapes_colors.items():
		palette[class_id] = color

	for class_id in range(31, 255):
		hue = (class_id * 137.5) % 360
		saturation = 75
		value = 220
		import colorsys
		r, g, b = colorsys.hsv_to_rgb(hue / 360.0, saturation / 100.0, value / 255.0)
		palette[class_id] = [int(r * 255), int(g * 255), int(b * 255)]

	return palette


def _segmentation_to_visual_frame(segmentation_map: np.ndarray, frame_size: int, palette: np.ndarray) -> np.ndarray:
	segmentation_map = np.asarray(segmentation_map, dtype=np.int32)
	visual = palette[np.clip(segmentation_map, 0, 255)]
	if visual.shape[:2] != (frame_size, frame_size):
		visual = cv2.resize(visual, (frame_size, frame_size), interpolation=cv2.INTER_NEAREST)
	return visual.astype(np.uint8)


def _segmentation_to_one_hot(segmentation_map: np.ndarray, num_classes: int) -> np.ndarray:
	segmentation_map = np.asarray(segmentation_map, dtype=np.int64)
	segmentation_map = np.clip(segmentation_map, 0, num_classes - 1)
	one_hot = np.eye(num_classes, dtype=np.float32)[segmentation_map]
	return np.transpose(one_hot, (2, 0, 1))


def _load_clip(
	video_path: str,
	clip_frames: int,
	frame_stride: int,
	frame_size: int,
	rng: np.random.Generator,
	segmentor: Segmentor,
	palette: np.ndarray,
	num_classes: int,
) -> np.ndarray:
	capture = cv2.VideoCapture(video_path)
	if not capture.isOpened():
		raise RuntimeError(f"Could not open video: {video_path}")

	frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	total_span = max(1, (clip_frames - 1) * frame_stride + 1)
	max_start = max(0, frame_count - total_span)
	start_index = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
	capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)

	frames: list[np.ndarray] = []
	last_frame: np.ndarray | None = None
	for step in range(total_span):
		success, frame = capture.read()
		if not success:
			if last_frame is None:
				break
			frame = last_frame.copy()
		else:
			last_frame = frame

		if step % frame_stride == 0:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			seg_result = segmentor(frame)
			seg_one_hot = _segmentation_to_one_hot(seg_result.segmentation_map, num_classes)
			frames.append(seg_one_hot)
			if len(frames) >= clip_frames:
				break

	capture.release()

	if not frames:
		raise RuntimeError(f"No readable frames found in: {video_path}")

	while len(frames) < clip_frames:
		frames.append(frames[-1].copy())

	clip = np.stack(frames[:clip_frames], axis=0).astype(np.float32)
	clip = np.transpose(clip, (1, 0, 2, 3))
	return clip


def _clip_to_visual_frames(clip_tensor: torch.Tensor, palette: np.ndarray) -> np.ndarray:
	clip = clip_tensor.detach().cpu()
	if clip.ndim != 4:
		raise ValueError("clip_tensor must have shape (C, T, H, W)")
	class_ids = torch.argmax(clip, dim=0).numpy().astype(np.int32)
	frames = palette[np.clip(class_ids, 0, 255)]
	return frames.astype(np.uint8)


def _logits_to_visual_frames(logits_tensor: torch.Tensor, palette: np.ndarray) -> np.ndarray:
	logits = logits_tensor.detach().cpu()
	if logits.ndim != 4:
		raise ValueError("logits_tensor must have shape (C, T, H, W)")
	class_ids = torch.argmax(logits, dim=0).numpy().astype(np.int32)
	frames = palette[np.clip(class_ids, 0, 255)]
	return frames.astype(np.uint8)


def _save_video(frames_rgb: np.ndarray, output_path: str, fps: float) -> None:
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
		raise ValueError("frames_rgb must have shape (T, H, W, 3)")
	h, w = frames_rgb.shape[1], frames_rgb.shape[2]
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(output_path, fourcc, float(fps), (w, h))
	if not writer.isOpened():
		raise RuntimeError(f"Could not open video writer for: {output_path}")
	for frame in frames_rgb:
		writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
	writer.release()


def _evaluate(model: VideoClipVAE, loader: DataLoader, device: torch.device, beta: float) -> dict[str, float]:
	model.eval()
	loss_sum = 0.0
	recon_sum = 0.0
	kl_sum = 0.0
	mae_sum = 0.0
	n_batches = 0

	with torch.no_grad():
		for x in loader:
			x = x.to(device)
			recon, mu, logvar = model(x)
			positive = x.sum()
			negative = x.numel() - positive
			pos_weight = torch.clamp(negative / (positive + 1e-6), min=1.0, max=40.0)
			loss, parts = vae_loss(recon, x, mu, logvar, beta=beta, pos_weight=pos_weight)
			mae = torch.mean(torch.abs(torch.sigmoid(recon) - x))

			loss_sum += float(loss.detach().cpu())
			recon_sum += float(parts["recon"].cpu())
			kl_sum += float(parts["kl"].cpu())
			mae_sum += float(mae.detach().cpu())
			n_batches += 1

	denom = max(n_batches, 1)
	return {
		"loss": loss_sum / denom,
		"recon": recon_sum / denom,
		"kl": kl_sum / denom,
		"mae": mae_sum / denom,
	}


class BDD100KVideoClipDataset(Dataset):
	def __init__(
		self,
		video_files: list[str],
		clip_frames: int,
		frame_stride: int,
		frame_size: int,
		num_samples: int,
		seed: int = 0,
		segmentor: Segmentor | None = None,
		palette: np.ndarray | None = None,
		validate_files: bool = True,
	) -> None:
		if not video_files:
			raise RuntimeError("No video files provided")
		self.video_files = list(video_files)
		self.clip_frames = int(clip_frames)
		self.frame_stride = int(frame_stride)
		self.frame_size = int(frame_size)
		self.num_samples = int(num_samples)
		self.seed = int(seed)
		self.num_classes = 32
		self.segmentor = segmentor or Segmentor("segformer-offline")
		self.palette = palette if palette is not None else _build_segmentation_palette()
		if validate_files:
			self.video_files = self._filter_readable_files(self.video_files)

	@staticmethod
	def _filter_readable_files(video_files: list[str]) -> list[str]:
		valid_files: list[str] = []
		for video_path in video_files:
			capture = cv2.VideoCapture(video_path)
			if capture.isOpened():
				valid_files.append(video_path)
			capture.release()
		return valid_files or list(video_files)

	def __len__(self) -> int:
		return self.num_samples

	def __getitem__(self, idx: int) -> torch.Tensor:
		video_path = self.video_files[idx % len(self.video_files)]
		rng = np.random.default_rng(self.seed + int(idx))
		clip = _load_clip(
			video_path,
			clip_frames=self.clip_frames,
			frame_stride=self.frame_stride,
			frame_size=self.frame_size,
			rng=rng,
			segmentor=self.segmentor,
			palette=self.palette,
			num_classes=self.num_classes,
		)
		return torch.from_numpy(clip)


def main() -> None:
	parser = argparse.ArgumentParser(description="Train a simple VAE on BDD100K video clips")
	parser.add_argument(
		"--data_dir",
		type=str,
		default=os.path.join(CODE_ROOT, "Dataset", "BDD100K", "Videos", "bdd100k_videos_train_00"),
		help="Path containing BDD100K videos (scanned recursively).",
	)
	parser.add_argument("--out_dir", type=str, default=os.path.join(CODE_ROOT, "assets", "test", "video_vae"))
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch_size", type=int, default=4)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--beta", type=float, default=0.1)
	parser.add_argument("--latent_dim", type=int, default=64)
	parser.add_argument("--kl_warmup_epochs", type=int, default=5)
	parser.add_argument("--kl_cycle_epochs", type=int, default=0)
	parser.add_argument("--clip_frames", type=int, default=16)
	parser.add_argument("--frame_size", type=int, default=64)
	parser.add_argument("--frame_stride", type=int, default=2)
	parser.add_argument("--num_samples_per_epoch", type=int, default=256)
	parser.add_argument("--limit_files", type=int, default=None)
	parser.add_argument("--seed", type=int, default=0)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--val_ratio", type=float, default=0.2)
	parser.add_argument("--save_samples_every", type=int, default=1)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	logger.info(f"Device: {device}")
	logger.info(f"Data dir: {args.data_dir}")

	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	segmentor = Segmentor("segformer-offline")
	palette = _build_segmentation_palette()
	num_classes = 32

	video_files = _scan_video_files(args.data_dir, limit_files=args.limit_files)
	if not video_files:
		raise RuntimeError(f"No video files found in: {args.data_dir}")

	train_files, val_files = _split_files(video_files, val_ratio=args.val_ratio)
	if not val_files:
		val_files = train_files[:1]

	train_dataset = BDD100KVideoClipDataset(
		train_files,
		clip_frames=args.clip_frames,
		frame_stride=args.frame_stride,
		frame_size=args.frame_size,
		num_samples=args.num_samples_per_epoch,
		seed=args.seed,
		segmentor=segmentor,
		palette=palette,
		num_classes=num_classes,
		validate_files=True,
	)
	val_dataset = BDD100KVideoClipDataset(
		val_files,
		clip_frames=args.clip_frames,
		frame_stride=args.frame_stride,
		frame_size=args.frame_size,
		num_samples=max(1, min(64, len(val_files) * 4)),
		seed=args.seed + 999,
		segmentor=segmentor,
		palette=palette,
		num_classes=num_classes,
		validate_files=True,
	)

	model = VideoClipVAE(
		clip_frames=args.clip_frames,
		frame_size=args.frame_size,
		input_channels=num_classes,
		latent_dim=args.latent_dim,
	).to(device)
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
			"model": {
				"clip_frames": args.clip_frames,
				"frame_size": args.frame_size,
				"input_channels": num_classes,
				"latent_dim": args.latent_dim,
			},
			"kl_schedule": {
				"beta": args.beta,
				"warmup_epochs": args.kl_warmup_epochs,
				"cycle_epochs": args.kl_cycle_epochs,
			},
		},
	)

	logger.info(
		f"Training: files={len(train_files)} val_files={len(val_files)} clip_frames={args.clip_frames} frame_size={args.frame_size}"
	)

	history: list[dict] = []
	frame_rate = 1.0 / max(1, args.frame_stride)

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
			recon, mu, logvar = model(x)
			positive = x.sum()
			negative = x.numel() - positive
			pos_weight = torch.clamp(negative / (positive + 1e-6), min=1.0, max=40.0)
			loss, parts = vae_loss(recon, x, mu, logvar, beta=effective_beta, pos_weight=pos_weight)
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
			f"| val_loss={val_metrics['loss']:.4f} val_recon={val_metrics['recon']:.4f} val_kl={val_metrics['kl']:.4f} val_mae={val_metrics['mae']:.4f}"
		)

		history.append(
			{
				"epoch": epoch,
				"train_loss": avg_loss,
				"train_recon": avg_recon,
				"train_kl": avg_kl,
				"effective_beta": effective_beta,
				**{f"val_{key}": value for key, value in val_metrics.items()},
			}
		)
		_save_json(os.path.join(args.out_dir, "history.json"), history)

		ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{epoch:03d}.pt")
		torch.save(
			{
				"epoch": epoch,
				"model_state": model.state_dict(),
				"optimizer_state": optimizer.state_dict(),
				"input_shape": {
					"clip_frames": args.clip_frames,
					"frame_size": args.frame_size,
				},
				"latent_dim": args.latent_dim,
			},
			ckpt_path,
		)

		if epoch % max(1, args.save_samples_every) == 0:
			sample_x = next(iter(val_loader)).to(device)
			model.eval()
			with torch.no_grad():
				sample_recon, _, _ = model(sample_x)

			sample_x_np = _clip_to_visual_frames(sample_x[0], palette)
			sample_recon_np = _logits_to_visual_frames(sample_recon[0], palette)
			np.savez_compressed(
				os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}.npz"),
				input_clip=sample_x_np,
				recon_clip=sample_recon_np,
			)
			_save_video(sample_x_np, os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}_input.mp4"), fps=frame_rate)
			_save_video(sample_recon_np, os.path.join(args.out_dir, f"sample_epoch_{epoch:03d}_recon.mp4"), fps=frame_rate)

	logger.info(f"Done. Checkpoints saved to: {args.out_dir}")


if __name__ == "__main__":
	main()