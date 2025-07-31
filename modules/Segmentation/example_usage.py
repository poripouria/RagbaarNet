"""
Example usage of the Segmentor Framework
========================================

This script demonstrates how to use the modular Segmentor class
with different models and input types.
"""

import cv2
from Segmentor import Segmentor

if __name__ == "__main__":
    # # Load test image
    # image_path = "assets/test/nexet(frame_9da98090-9b48-496f-a829-62daad54f574_00004-1280_720).jpg"
    # image = cv2.imread(image_path)    
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(f"✅ Successfully loaded image with shape: {image.shape}")
    
    # print("🚀 Testing Segmentor Framework")
    # print("=" * 50)
    
    # # Test YOLO Segmentor
    # print("\n1. Testing YOLO Segmentor:")
    # print("-" * 30)
    yolo_segmentor = Segmentor('yolo', 'modules/Segmentation/Pre-trained Models/yolov8m-seg.pt')
    # yolo_result = yolo_segmentor(image)
    
    # print(f"✅ YOLO Segmentation completed")
    # print(f"   - Segmentation map shape: {yolo_result.segmentation_map.shape}")
    # print(f"   - Number of classes: {len(yolo_result.class_labels)}")
    # print(f"   - Detected objects: {len(yolo_result.bounding_boxes) if yolo_result.bounding_boxes else 0}")
    # print(f"   - Device used: {yolo_result.metadata['device']}")
    
    # # Display results
    # yolo_segmentor.visualize_results(image, yolo_result, show_confidence=True)
    
    # # Test Segformer Segmentor
    # print("\n2. Testing Segformer Segmentor:")
    # print("-" * 35)
    segformer_segmentor = Segmentor('segformer')
    # segformer_result = segformer_segmentor(image)
    
    # print(f"✅ Segformer Segmentation completed")
    # print(f"   - Segmentation map shape: {segformer_result.segmentation_map.shape}")
    # print(f"   - Number of classes: {len(segformer_result.class_labels)}")
    # print(f"   - Confidence map available: {segformer_result.confidence_map is not None}")
    # print(f"   - Device used: {segformer_result.metadata['device']}")
    
    # # Display results
    # segformer_segmentor.visualize_results(image, segformer_result, show_confidence=True)
    
    # Test on Video
    print("\n3. Testing Video Segmentation:")
    print("-" * 30)

    video_path = "assets/test/00a2f5b6-d4217a96.mov"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # print(f"✅ Successfully loaded video frame with shape: {frame.shape}")

        # Test YOLO Segmentor
        yolo_result = yolo_segmentor(frame)
        # print(f"✅ YOLO Segmentation completed")
        # print(f"   - Segmentation map shape: {yolo_result.segmentation_map.shape}")
        # print(f"   - Number of classes: {len(yolo_result.class_labels)}")
        # print(f"   - Detected objects: {len(yolo_result.bounding_boxes) if yolo_result.bounding_boxes else 0}")
        # print(f"   - Device used: {yolo_result.metadata['device']}")

        # Display results
        yolo_segmentor.visualize_results(frame, yolo_result, show_confidence=True)

        # Test Segformer Segmentor
        segformer_result = segformer_segmentor(frame)
        # print(f"✅ Segformer Segmentation completed")
        # print(f"   - Segmentation map shape: {segformer_result.segmentation_map.shape}")
        # print(f"   - Number of classes: {len(segformer_result.class_labels)}")
        # print(f"   - Confidence map available: {segformer_result.confidence_map is not None}")
        # print(f"   - Device used: {segformer_result.metadata['device']}")

        # Display results
        segformer_segmentor.visualize_results(frame, segformer_result, show_confidence=True)

    cap.release()
    print("🎉 Video segmentation completed successfully!")
