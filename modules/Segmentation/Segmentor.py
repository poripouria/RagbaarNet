"""
Modular Segmentation Framework for Computer Vision
=================================================

This module provides an extensible framework for semantic segmentation using various models.
It supports YOLO and Segformer models with easy integration for additional models.
"""

import torch
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
from ultralytics import YOLO
from transformers import SegformerConfig, SegformerImageProcessor, SegformerForSemanticSegmentation
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.logging_setup import setup_logging

logger = setup_logging("INFO", name="segmentation.segmentor")


@dataclass
class SegmentationResult:
    """
    Data class to store segmentation results from any model.

    Attributes:
        segmentation_map: Numpy array containing pixel-wise class predictions
        confidence_map: Optional confidence scores for each pixel
        class_labels: List of class names
        bounding_boxes: Optional bounding boxes for detected objects
        masks: Optional individual object masks
        metadata: Additional model-specific information
    """

    segmentation_map: np.ndarray
    confidence_map: Optional[np.ndarray] = None
    class_labels: List[str] = None
    bounding_boxes: Optional[List[Dict]] = None
    masks: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = None

class BaseSegmentor(ABC):
    """
    Abstract base class for all segmentation models.

    This class defines the interface that all segmentation models must implement,
    ensuring consistency and extensibility across different model architectures.
    """

    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Initialize the base segmentor.

        Args:
            model_path: Path to the model file or model identifier
            device: Device to run the model on ('auto', 'cpu', 'cuda')
        """

        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.is_loaded = False

    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for model inference."""

        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        device_norm = str(device).lower()
        if device_norm.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available; falling back to CPU.")
            return 'cpu'
        return device

    @abstractmethod
    def load_model(self) -> None:
        """Load the segmentation model."""
        pass

    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> Any:
        """
        Preprocess the input image for the specific model.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            Preprocessed input ready for model inference
        """
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        Perform segmentation on the input image.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            SegmentationResult containing all segmentation information
        """
        pass

    @abstractmethod
    def get_class_labels(self) -> List[str]:
        """Get the list of class labels for this model."""
        pass

    def __call__(self, image: np.ndarray) -> SegmentationResult:
        """
        Convenience method to call predict directly.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            SegmentationResult containing all segmentation information
        """

        if not self.is_loaded:
            self.load_model()
        return self.predict(image)

class YOLOSegmentor(BaseSegmentor):
    """
    YOLO-based segmentation implementation.

    Supports various YOLO models for instance segmentation with object detection
    capabilities including bounding boxes and individual object masks.
    """

    def __init__(self, model_path: str = "yolo11/yolo11s-seg.pt", device: str = 'auto'):
        """
        Initialize YOLO segmentor.

        Args:
            model_path: Path to YOLO model file
            device: Device to run the model on
        """

        super().__init__(model_path, device)

    def load_model(self) -> None:
        """Load the YOLO model."""

        try:
            # Check if model path exists in Pre-trained Models directory
            if not os.path.exists(self.model_path):
                pretrained_path = os.path.join("modules/Segmentation/Pre-trained Models", self.model_path)
                if os.path.exists(pretrained_path):
                    self.model_path = pretrained_path

            self.model = YOLO(self.model_path, task='segment')
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("✅ YOLO model loaded on %s", self.device)

        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for YOLO.

        Args:
            image: Input image in RGB format

        Returns:
            Image ready for YOLO inference
        """
        # YOLO handles preprocessing internally
        return image

    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        Perform YOLO segmentation.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            SegmentationResult with instance segmentation information
        """

        if not self.is_loaded:
            self.load_model()

        h, w = image.shape[:2]

        # YOLO Ultralytics expects BGR when passing numpy array directly
        yolo_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Ensure evaluation/inference context
        with torch.inference_mode():
            results = self.model.predict(
                source = yolo_input,
                device = self.device,
                half = (self.device.startswith("cuda") and torch.cuda.is_available()),
                verbose = False
            )[0]

        # Initialize maps and lists for results
        segmentation_map = np.full((h, w), 255, dtype=np.uint16)  # 255 = background sentinel (no detection)
        confidence_map   = np.zeros((h, w), dtype=np.float32)
        bounding_boxes = []
        masks = []

        if results.masks is not None and len(results.masks) > 0:
            # Convert tensors to numpy once
            masks_data = results.masks.data.cpu().numpy()      # (N, H, W) normalized
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy()

            for mask, box, conf, cls in zip(masks_data, boxes, confs, clss):
                class_id = int(cls)

                # Resize mask
                mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                # Fill segmentation map
                segmentation_map[mask_binary == 1] = class_id

                # Update confidence map (keep maximum)
                confidence_map = np.maximum(confidence_map, mask_binary * float(conf))

                # Store additional info
                masks.append(mask_binary)
                bounding_boxes.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': class_id,
                    'class_name': self.model.names[class_id]
                })

        # Ordered class labels (safe)
        class_labels = [self.model.names[i] for i in sorted(self.model.names.keys())]

        return SegmentationResult(
            segmentation_map = segmentation_map,
            confidence_map = confidence_map,
            class_labels = class_labels,
            bounding_boxes = bounding_boxes,
            masks = masks,
            metadata = {
                'model_type': 'YOLO',
                'model_path': self.model_path,
                'device': self.device,
                'num_detected': len(masks),
                'num_classes': len(class_labels)
            }
        )

    def get_class_labels(self) -> List[str]:
        """Get YOLO class labels."""
        if not self.is_loaded:
            self.load_model()
        return list(self.model.names.values())

class SegformerSegmentor(BaseSegmentor):
    """
    Segformer-based semantic segmentation implementation.

    Provides dense pixel-wise segmentation using transformer-based architecture
    with support for various Segformer model variants.
    """

    def __init__(
        self,
        model_path: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
        device: str = 'auto',
        local_files_only: Optional[bool] = None,
        return_confidence: bool = True,
    ):
        """
        Initialize Segformer segmentor.

        Args:
            model_path: Hugging Face model identifier or local path
            device: Device to run the model on
            return_confidence: Whether to compute per-pixel confidence map
        """

        super().__init__(model_path, device)
        self.processor = None
        self._local_files_only_override = local_files_only
        self.return_confidence = return_confidence
        self.cityscapes_labels = None
        # ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
        #  "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
        #  "truck", "bus", "train", "motorcycle", "bicycle"]

        # Check and log if GPU is available and will be used
        if torch.cuda.is_available():
            logger.info("🔥CUDA is available. Segformer will run on GPU.")
        else:
            logger.info("⚠️CUDA is NOT available. Segformer will run on CPU, which may be slow.")

    def _resolve_model_identifier(self, local_files_only: bool) -> Tuple[str, bool]:
        """Resolve the model identifier/path with offline/online support to load.

        In offline mode we try to find a local directory (env override or common project paths).
        In online mode we avoid implicitly overriding the HF identifier (unless a local path was
        explicitly provided).

        Returns:
            (identifier, is_local_path)
        """

        if self.model_path and os.path.exists(self.model_path):
            return self.model_path, True

        if not local_files_only:
            return self.model_path, False

        # Offline mode - try common locations
        candidates: List[str] = []
        env_path = os.environ.get("RAGBAARNET_SEGFORMER_PATH") or os.environ.get("SEGFORMER_MODEL_PATH")
        if env_path:
            candidates.append(env_path)

        if self.model_path:
            candidates.extend([
                os.path.join("modules", "Segmentation", "Pre-trained Models", self.model_path),
                os.path.join("modules", "Segmentation", "Pre-trained Models", self.model_path.replace("/", "--")),
            ])

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                return candidate, True

        return self.model_path, False

    def load_model(self) -> None:
        """Load the Segformer model with safety considerations."""

        try:
            allow_net = os.environ.get("RAGBAARNET_ALLOW_NET", "").strip().lower() in {"1", "true", "yes"}
            local_files_only = self._local_files_only_override if self._local_files_only_override is not None else not allow_net

            resolved_id, is_local = self._resolve_model_identifier(local_files_only)

            # Load processor + model from the same place (local dir or HF cache).
            try:
                self.processor = SegformerImageProcessor.from_pretrained(
                    resolved_id, local_files_only=local_files_only
                )
            except Exception:
                # Some local snapshots may only include weights/config (e.g., model.safetensors + config.json)
                # but not preprocessor_config.json. In that case, fall back to defaults.
                self.processor = SegformerImageProcessor()  # fallback

            # Load config and force correct labels
            config = SegformerConfig.from_pretrained(resolved_id, local_files_only=local_files_only)
            self.cityscapes_labels = [config.id2label[i] for i in range(config.num_labels)]

            # Load model with safety checks and support for both safetensors and pytorch_model.bin formats.
            use_safetensors = True
            if is_local and not os.path.exists(os.path.join(resolved_id, "model.safetensors")):
                use_safetensors = False

            self.model = SegformerForSemanticSegmentation.from_pretrained(
                resolved_id,
                config=config,
                use_safetensors=use_safetensors,
                local_files_only=local_files_only,
                ignore_mismatched_sizes=True
            )

            if is_local:
                self.model_path = resolved_id

            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True

            torch.backends.cudnn.benchmark = True

            logger.info(f"✅ Segformer ({len(self.cityscapes_labels)} classes) loaded on {self.device}")

        except Exception as e:
            if self._local_files_only_override is None:
                allow_net = os.environ.get("RAGBAARNET_ALLOW_NET", "").strip().lower() in {"1", "true", "yes"}
                local_files_only = not allow_net
            else:
                local_files_only = self._local_files_only_override

            resolved_id, is_local = self._resolve_model_identifier(local_files_only=local_files_only)

            if local_files_only and not is_local:
                raise RuntimeError(
                    f"Failed to load Segformer in offline mode.\n"
                    f"Model not found at: {resolved_id}\n"
                    "Set RAGBAARNET_SEGFORMER_PATH or allow network (RAGBAARNET_ALLOW_NET=1)"
                ) from e
            raise RuntimeError(f"Failed to load Segformer model: {e}") from e

    def preprocess_image(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess image for Segformer.

        Args:
            image: Input image in RGB format

        Returns:
            Preprocessed inputs ready for Segformer
        """

        if self.processor is None:
            self.load_model()

        inputs = self.processor(images=image, return_tensors="pt", do_rescale=True, do_normalize=True)

        if self.device.startswith('cuda') and torch.cuda.is_available():
            inputs = {
                k: v.pin_memory().to(self.device, non_blocking=True)
                for k, v in inputs.items()
            }
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        return inputs

    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        Perform Segformer segmentation.

        Args:
            image: Input image as numpy array (RGB format)

        Returns:
            SegmentationResult with semantic segmentation information
        """
        if not self.is_loaded:
            self.load_model()

        # Preprocess the image
        h, w = image.shape[:2]
        inputs = self.preprocess_image(image)

        # Perform inference (optimized: inference_mode + autocast on CUDA)
        with torch.inference_mode():
            if self.device == 'cuda' and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        logits = outputs.logits  # [1, num_classes, height, width]

        # Upsample to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=(h, w), mode='bilinear', align_corners=False
        )

        # Final predictions
        segmentation_map = torch.argmax(upsampled_logits, dim=1).cpu().numpy()[0].astype(np.uint16)
        confidence_map = None
        if self.return_confidence:
            softmax_probs = torch.softmax(upsampled_logits, dim=1)
            confidence_map = torch.max(softmax_probs, dim=1)[0].cpu().numpy()[0]
            # probs = torch.softmax(upsampled_logits, dim=1)
            # confidence_map = (probs.max(dim=1).values.cpu().numpy()[0])

        unique, counts = np.unique(segmentation_map, return_counts=True)
        class_areas = dict(zip(unique.tolist(), counts.tolist()))

        return SegmentationResult(
            segmentation_map=segmentation_map,
            confidence_map=confidence_map,
            class_labels=self.cityscapes_labels,
            bounding_boxes=None,    # Segformer doesn't provide bounding boxes
            masks=None,             # Dense segmentation, no individual masks
            metadata={
                'model_type': 'Segformer',
                'model_path': self.model_path,
                'device': self.device,
                'num_detected': int((segmentation_map > 0).sum()),
                'num_classes': len(self.cityscapes_labels),
                'class_areas': class_areas
            }
        )

    def get_class_labels(self) -> List[str]:
        """Get Segformer class labels."""

        if not self.cityscapes_labels:
            self.load_model()
        return self.cityscapes_labels


class Segmentor:
    """
    Main Segmentor class that provides a unified interface for different segmentation models.

    This class acts as a factory and manager for different segmentation models,
    allowing easy switching between models and unified result handling.
    """

    def __init__(self, model_type: str = 'segformer', model_path: str = None, device: str = 'auto'):
        """
        Initialize the main Segmentor.

        Args:
            model_type: Type of model ('yolo', 'segformer')
            model_path: Path to model or model identifier
            device: Device to run the model on
        """

        self.model_type = model_type.lower()
        self.device = device
        self.segmentor = self._create_segmentor(model_type, model_path, device)

    def _create_segmentor(self, model_type: str, model_path: str, device: str) -> BaseSegmentor:
        """Create the appropriate segmentor based on model type."""

        model_type_norm = model_type.lower().strip().replace("_", "-")

        if model_type_norm == 'yolo':
            if model_path is None:
                model_path = "yolo11s.pt"
            return YOLOSegmentor(model_path, device)
        elif model_type_norm == 'segformer':
            if model_path is None:
                model_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"

            is_local_path = model_path and os.path.exists(model_path)

            return SegformerSegmentor(model_path=model_path, device=device, local_files_only=is_local_path)

        else:
            raise ValueError(f"Unsupported model type: {model_type}.\n"
                             f"Supported types: 'yolo', 'segformer'")

    def __call__(self, image: Union[np.ndarray, str]) -> SegmentationResult:
        """
        Perform segmentation on input image.

        Args:
            image: Input image as numpy array (RGB) or path to image file

        Returns:
            SegmentationResult containing all segmentation information
        """

        # Handle different input types
        if isinstance(image, str):
            image_path = image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from path: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
            # Assume it's already in RGB format
            pass
        else:
            raise ValueError("Image must be a numpy array (RGB) or path to image file")

        return self.segmentor(image)

    def get_class_labels(self) -> List[str]:
        """Get class labels for the current model."""

        return self.segmentor.get_class_labels()

    def visualize_results(self, image: np.ndarray, result: SegmentationResult,
                         show_confidence: bool = False, figsize: Tuple[int, int] = (15, 5)) -> None:
        """
        Visualize segmentation results.

        Args:
            image: Original input image
            result: SegmentationResult from segmentation
            show_confidence: Whether to show confidence map
            figsize: Figure size for matplotlib
        """

        # Lazy import heavy plotting libs to avoid startup/runtime overhead when not used
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        num_plots = 3 if show_confidence else 2
        fig, axes = plt.subplots(1, num_plots, figsize=figsize)

        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')

        # Segmentation map
        if result.class_labels:
            # Generate colors for the segmentation map
            num_classes = len(result.class_labels)

            # Try to use tab20 colormap first (works well for up to 20 classes)
            if num_classes <= 20:
                try:
                    base_cmap = plt.colormaps.get_cmap('tab20')
                    colors = [base_cmap(i) for i in range(num_classes)]
                except Exception:
                    # Fallback if tab20 is not available
                    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
            else:
                # For more than 20 classes, sample from a continuous colormap
                base_cmap = plt.colormaps.get_cmap('gist_ncar')
                colors = [base_cmap(i / num_classes) for i in range(num_classes)]

            cmap = mcolors.ListedColormap(colors)

            im = axes[1].imshow(result.segmentation_map, cmap=cmap,
                              vmin=0, vmax=len(result.class_labels)-1)
            axes[1].set_title(f"{result.metadata['model_type']} Segmentation")
            axes[1].axis('off')

            # Add colorbar with labels
            cbar = plt.colorbar(im, ax=axes[1], ticks=range(len(result.class_labels)))
            cbar.ax.set_yticklabels(result.class_labels, fontsize=8)
        else:
            axes[1].imshow(result.segmentation_map, cmap='viridis')
            axes[1].set_title(f"{result.metadata['model_type']} Segmentation")
            axes[1].axis('off')

        # Confidence map (if requested and available)
        if show_confidence and result.confidence_map is not None:
            im_conf = axes[2].imshow(result.confidence_map, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title("Confidence Map")
            axes[2].axis('off')
            plt.colorbar(im_conf, ax=axes[2])

        plt.tight_layout()
        plt.show()

    def switch_model(self, model_type: str, model_path: str = None) -> None:
        """
        Switch to a different segmentation model.

        Args:
            model_type: New model type ('yolo', 'segformer')
            model_path: Path to new model
        """

        self.model_type = model_type.lower()
        self.segmentor = self._create_segmentor(model_type, model_path, self.device)
        logger.info("Switched to %s model", model_type)
