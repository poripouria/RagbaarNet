"""
Example usage of the Segmentor Framework
========================================

This script demonstrates how to use the modular Segmentor class
with different models and input types.
"""

import cv2
import numpy as np
import os
from Segmentor import Segmentor

def main():
    """Main demonstration function."""

    # Load test image
    image_path = "assets/test/nexet(frame_9da98090-9b48-496f-a829-62daad54f574_00004-1280_720).jpg"
    image = cv2.imread(image_path)    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"✅ Successfully loaded image with shape: {image.shape}")
    
    print("🚀 Testing Segmentor Framework")
    print("=" * 50)
    
    # Test YOLO Segmentor
    print("\n1. Testing YOLO Segmentor:")
    print("-" * 30)
    yolo_segmentor = Segmentor('yolo', 'modules/Segmentation/Pre-trained Models/yolov8m-seg.pt')
    yolo_result = yolo_segmentor(image)
    
    print(f"✅ YOLO Segmentation completed")
    print(f"   - Segmentation map shape: {yolo_result.segmentation_map.shape}")
    print(f"   - Number of classes: {len(yolo_result.class_labels)}")
    print(f"   - Detected objects: {len(yolo_result.bounding_boxes) if yolo_result.bounding_boxes else 0}")
    print(f"   - Device used: {yolo_result.metadata['device']}")
    
    # Display results
    yolo_segmentor.visualize_results(image, yolo_result, show_confidence=True)
    
    # Test Segformer Segmentor
    print("\n2. Testing Segformer Segmentor:")
    print("-" * 35)
    segformer_segmentor = Segmentor('segformer')
    segformer_result = segformer_segmentor(image)
    
    print(f"✅ Segformer Segmentation completed")
    print(f"   - Segmentation map shape: {segformer_result.segmentation_map.shape}")
    print(f"   - Number of classes: {len(segformer_result.class_labels)}")
    print(f"   - Confidence map available: {segformer_result.confidence_map is not None}")
    print(f"   - Device used: {segformer_result.metadata['device']}")
    
    # Display results
    segformer_segmentor.visualize_results(image, segformer_result, show_confidence=True)
    
    # Test model switching
    print("\n3. Testing Model Switching:")
    print("-" * 30)
    
    # Create a single segmentor and switch between models
    segmentor = Segmentor('yolo')
    
    # First, use YOLO
    result1 = segmentor(image)
    print(f"✅ Current model: {result1.metadata['model_type']}")
    
    # Switch to Segformer
    segmentor.switch_model('segformer')
    result2 = segmentor(image)
    print(f"✅ Switched to: {result2.metadata['model_type']}")
    
    # Compare results side by side
    print("\n4. Comparing Results:")
    print("-" * 25)
    
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # YOLO results
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(yolo_result.segmentation_map, cmap='tab20')
    axes[0, 1].set_title("YOLO Segmentation")
    axes[0, 1].axis('off')
    
    # Segformer results
    axes[1, 0].imshow(segformer_result.segmentation_map, cmap='tab20')
    axes[1, 0].set_title("Segformer Segmentation")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(segformer_result.confidence_map, cmap='hot')
    axes[1, 1].set_title("Segformer Confidence")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("\n🎉 Framework testing completed successfully!")
    print("\nNext steps:")
    print("- Add more models by extending BaseSegmentor")
    print("- Customize visualization methods")
    print("- Integrate with video processing pipeline")


if __name__ == "__main__":
    main()
