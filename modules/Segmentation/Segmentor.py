"""
Modular Segmentation Framework for Computer Vision
=================================================

This module provides an extensible framework for semantic segmentation using various models.
It supports YOLO and Segformer models with easy integration for additional models.
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union, Optional, Any
from dataclasses import dataclass
import os

# Model-specific imports
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


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
    
    def __init__(self, model_path: str = "yolov8s-seg.pt", device: str = 'auto'):
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
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            self.is_loaded = True
            print(f"✅ YOLO model loaded on {self.device}")
            
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
            
        # Get YOLO results
        results = self.model(image)[0]
        
        # Create segmentation map
        segmentation_map = np.zeros(image.shape[:2], dtype=np.uint8)
        bounding_boxes = []
        masks = []
        confidence_scores = []
        
        if results.masks is not None:
            for i, (mask, box, conf, cls) in enumerate(zip(
                results.masks.data.cpu().numpy(),
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy()
            )):
                # Resize mask to original image size
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Assign class ID to segmentation map
                segmentation_map[mask_binary == 1] = int(cls) + 1  # +1 to avoid background (0)
                
                # Store bounding box information
                bounding_boxes.append({
                    'bbox': box.tolist(),
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
                
                masks.append(mask_binary)
                confidence_scores.append(float(conf))
        
        # Create confidence map
        confidence_map = np.zeros(image.shape[:2], dtype=np.float32)
        for i, (mask, conf) in enumerate(zip(masks, confidence_scores)):
            confidence_map[mask == 1] = conf
        
        return SegmentationResult(
            segmentation_map=segmentation_map,
            confidence_map=confidence_map,
            class_labels=list(self.model.names.values()),
            bounding_boxes=bounding_boxes,
            masks=masks,
            metadata={
                'model_type': 'YOLO',
                'model_path': self.model_path,
                'device': self.device,
                'raw_results': results
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
    
    def __init__(self, model_path: str = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024", device: str = 'auto'):
        """
        Initialize Segformer segmentor.
        
        Args:
            model_path: Hugging Face model identifier or local path
            device: Device to run the model on
        """
        super().__init__(model_path, device)
        self.processor = None
        self.cityscapes_labels = [
            "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
            "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
            "truck", "bus", "train", "motorcycle", "bicycle"
        ]
        
    def load_model(self) -> None:
        """Load the Segformer model with safety considerations."""
        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.model_path,
                use_safetensors=True
            )       
            
            self.model.to(self.device)
            self.processor = SegformerImageProcessor()
            self.is_loaded = True
            print(f"✅ Segformer model loaded on: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load Segformer model: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        Preprocess image for Segformer.
        
        Args:
            image: Input image in RGB format
            
        Returns:
            Preprocessed inputs ready for Segformer
        """
        inputs = self.processor(images=image, return_tensors="pt")
        # Move inputs to the same device as the model
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
        inputs = self.preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits  # [1, num_classes, height, width]
        
        # Upsample to original image size
        original_height, original_width = image.shape[:2]
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(original_height, original_width),
            mode='bilinear',
            align_corners=False
        )
        
        # Get predictions and confidence
        softmax_probs = torch.softmax(upsampled_logits, dim=1)
        confidence_map = torch.max(softmax_probs, dim=1)[0].cpu().numpy()[0]
        segmentation_map = torch.argmax(upsampled_logits, dim=1).cpu().numpy()[0]
        
        return SegmentationResult(
            segmentation_map=segmentation_map,
            confidence_map=confidence_map,
            class_labels=self.cityscapes_labels,
            bounding_boxes=None,  # Segformer doesn't provide bounding boxes
            masks=None,  # Dense segmentation, no individual masks
            metadata={
                'model_type': 'Segformer',
                'model_path': self.model_path,
                'device': self.device,
                'raw_logits': upsampled_logits.cpu()
            }
        )
    
    def get_class_labels(self) -> List[str]:
        """Get Segformer class labels."""
        return self.cityscapes_labels


class Segmentor:
    """
    Main Segmentor class that provides a unified interface for different segmentation models.
    
    This class acts as a factory and manager for different segmentation models,
    allowing easy switching between models and unified result handling.
    """
    
    def __init__(self, model_type: str = 'yolo', model_path: str = None, device: str = 'auto'):
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
        if model_type.lower() == 'yolo':
            if model_path is None:
                model_path = "yolov8m-seg.pt"
            return YOLOSegmentor(model_path, device)
        elif model_type.lower() == 'segformer':
            if model_path is None:
                model_path = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
            return SegformerSegmentor(model_path, device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'yolo', 'segformer'")
    
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
            image = cv2.imread(image)
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
                except:
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
        print(f"Switched to {model_type} model")
