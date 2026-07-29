"""
Modular Scene Detection Framework for passing to Music Generation Module.
=========================================================================

This module provides an extensible framework for detecting scene events based on input data,
"""

import os
import sys
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from Segmentation.Segmentor import SegmentationResult
from utils.logging_setup import setup_logging

logger = setup_logging("INFO", name="Detection.Detector")

class ROI:
    """
    ROI defined by 4 corner points + 4 bezier control points
    """

    def __init__(
        self,
        corners: List[Tuple[float, float]],
        controls: List[Tuple[float, float]],
        frame_size: Tuple[int, int] = (1280, 720),
    ):
        """
        Args:
            corners: List of 4 corner points (x, y)
            controls: List of 4 bezier control points (x, y)
            frame_size: (width, height) of the frame these masks must align with.
                Must match the actual segmentation_map / mask resolution used at
                collision-check time, or intersects_mask() will silently misfire
                (wrong shape -> wrong/empty results).
        """

        if len(corners) != 4 or len(controls) != 4:
            raise ValueError("ROI must have exactly 4 corners and 4 control points")

        self.corners = corners
        self.controls = controls
        self.frame_width, self.frame_height = frame_size

        self.polygon = self._build_polygon()
        self.edges = self._build_edges()

        self.boundary_mask = self._build_boundary_mask(width=self.frame_width, height=self.frame_height)
        self.edge_masks = self._build_edge_masks(width=self.frame_width, height=self.frame_height)

    def _build_boundary_mask(self, width, height, thickness=3):

        mask = np.zeros((height, width), dtype=np.uint8)

        pts = np.array(self.polygon, dtype=np.int32)

        cv2.polylines(
            mask,
            [pts],
            isClosed=True,
            color=255,
            thickness=thickness
        )

        return mask.astype(bool)

    def _build_edge_masks(self, width, height, thickness=3):

        edge_masks = []

        samples_per_edge = len(self.polygon) // 4

        for i in range(4):

            mask = np.zeros((height, width), dtype=np.uint8)

            start = i * samples_per_edge
            end = (i + 1) * samples_per_edge

            pts = np.array(
                self.polygon[start:end],
                dtype=np.int32
            )

            cv2.polylines(
                mask,
                [pts],
                isClosed=False,
                color=255,
                thickness=thickness
            )

            edge_masks.append(mask.astype(bool))

        return edge_masks

    def _quad_bezier(self, p0, p1, p2, t):

        return (
            (1 - t) ** 2 * np.array(p0)
            + 2 * (1 - t) * t * np.array(p1)
            + t ** 2 * np.array(p2)
        )

    def _build_polygon(self):

        poly = []

        n = len(self.corners)

        for i in range(n):

            p0 = self.corners[i]
            p2 = self.corners[(i + 1) % n]
            p1 = self.controls[i]

            for t in np.linspace(0, 1, 20):
                pt = self._quad_bezier(p0, p1, p2, t)
                poly.append((pt[0], pt[1]))

        return poly

    def _build_edges(self):

        edges = []

        for i in range(len(self.polygon)):

            a = self.polygon[i]
            b = self.polygon[(i + 1) % len(self.polygon)]

            edges.append((a, b))

        return edges

    def calculate_intersection_area(self, mask):

        intersection = np.logical_and(mask, self.boundary_mask)
        area = np.sum(intersection)

        return area

    def calculate_ROI_area(self):

        area = np.sum(self.boundary_mask)
        return area

    def intersects_mask(self, mask, return_edges=False):

        touching = np.logical_and(mask, self.boundary_mask).any()

        if not return_edges:
            return touching

        edge_names = ["top", "right", "bottom", "left"]
        edges = []

        for name, edge_mask in zip(edge_names, self.edge_masks):
            if np.logical_and(mask, edge_mask).any():
                edges.append(name)

        erea = self.calculate_intersection_area(mask)

        return {
            "touching": touching,
            "edges": edges,
            "area": erea
        }

class BaseDetector(ABC):
    """
    Abstract base class for scene event detectors.
    This class defines the interface that all detection models must implement,
    ensuring consistency and extensibility across different detection strategies.
    """

    def __init__(self):
        # All sequential data is tracked by frame_id, which is passed in at each call.
        self.frame_counter = 0  

    @abstractmethod
    def detect_scene_events(self, *args, **kwargs):
        """
        Detect scene events.
        This method must be implemented by all subclasses.
        """
        pass

class ROIEventsDetector(BaseDetector):
    """
    Scene Event Detector that tracks objects and detects events based on ROI interactions.
    """

    def __init__(self):
        super().__init__()

        # state: keeps track of objects
        self.state = {
            "objects": {},          # object_id -> object info
            "next_object_id": 0
        }

        self.roi = None                 # Will be set per frame if provided
        self.prev_roi_payload = None    # Tracks ROI coordinates and frame dimensions
        self.max_missing_frames = 6     # Number of frames to keep an object in memory after it disappears

    def __call__(self,
            input: SegmentationResult,
            frame_id: int = 0,
            roi: Dict[str, Any] = None
        ):

        if not hasattr(input, "segmentation_map"):
            raise ValueError("Input must be a SegmentationResult instance")
        
        frame_height, frame_width = input.segmentation_map.shape[:2]
        self._set_roi(roi, frame_size=(frame_width, frame_height))
        self.frame_counter = frame_id

        detected = self.detect_scene_events(input.bounding_boxes, input.masks)

        return detected

    def _set_roi(self, roi_payload, frame_size):
        
        if not roi_payload:
            self.roi = None
            self.prev_roi_payload = None
            return

        roi_state = (roi_payload, frame_size)
        if self.prev_roi_payload != roi_state:
            self.prev_roi_payload = roi_state
            self.roi = ROI(
                corners=roi_payload.get("corners", []),
                controls=roi_payload.get("controls", []),
                frame_size=frame_size,
            )
            
            logger.info(f"💢 ROI updated for frame {self.frame_counter}. ROI area: {self.roi.calculate_ROI_area()}")

    def assign_object_ids(self, objects, masks, max_distance=200):
        """
        Assign unique IDs to detected objects based on their bounding boxes and class names. 
        The rule is to match objects across frames based on IoU proximity and class similarity, 
        while also considering the maximum allowed distance for matching.
        """

        updated_objects = {}
        used_tracks = set()

        for obj in objects:

            cls = obj["class_name"]
            bbox = obj["bbox"]
            mask = masks.get(cls, None)
            
            matched_id = None
            best_score = float("-inf")

            x1, y1, x2, y2 = bbox
            centroid = obj["centroid"] if "centroid" in obj.keys() else ((x1 + x2) / 2, (y1 + y2) / 2)

            # Search previous objects
            for object_id, previous in self.state["objects"].items():

                penalty = 0

                # Class name mismatch penalty
                if previous["class_name"] != cls:
                    # Allow for class variants (e.g., "person_3" vs "person_1")
                    if previous["class_name"].split("_")[0] == cls.split("_")[0]:  
                        penalty += -100
                    else:
                        continue        # Hard constraint: different classes cannot match

                # Already used in this frame penalty
                if object_id in used_tracks: 
                    continue            # Hard constraint: already matched in this frame

                # Distance penalty
                pcx, pcy = previous["centroid"]
                cx, cy = centroid
                distance = ((cx-pcx)**2 + (cy-pcy)**2)**0.5
                if distance > max_distance:
                    penalty += -((distance / max_distance) * 200)

                # Age reward: older objects are more likely to be the same object
                age = previous["age"]
                age_reward = min(age, 100)

                # Compute IoU for bounding boxes
                IoU = None
                px1, py1, px2, py2 = previous["bbox"]
                ix1, iy1, ix2, iy2 = max(x1, px1), max(y1, py1), min(x2, px2), min(y2, py2)
                if ix1 >= ix2 or iy1 >= iy2:
                    IoU = 0.0
                else:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    area = (x2 - x1) * (y2 - y1)
                    areap = (px2 - px1) * (py2 - py1)
                    union = area + areap - inter
                    IoU = inter / union if union > 0 else 0.0

                score = (
                    IoU * 500 + 
                    age_reward * 10 +
                    penalty
                )
                if score > best_score:
                    best_score = score
                    matched_id = object_id

            # Existing object
            if best_score > 0:
                obj_id = matched_id
                used_tracks.add(obj_id)
                previous = self.state["objects"][obj_id]
                is_touching = previous["touching"]
                age = previous["age"] + 1

            # New object
            else:
                obj_id = self.state["next_object_id"]
                self.state["next_object_id"] += 1
                is_touching = False
                age = 0

            obj["object_id"] = obj_id

            updated_objects[obj_id] = {
                "class_name": cls,
                "centroid": centroid,
                "bbox": bbox,
                "mask": mask,
                "touching": is_touching,
                "missing_frames": 0,
                "age": age,
                "last_seen_frame": self.frame_counter,
            }

        for object_id, previous in self.state["objects"].items():
            if object_id in updated_objects:
                continue
            previous["missing_frames"] += 1
            if previous["missing_frames"] <= self.max_missing_frames:
                updated_objects[object_id] = previous

        # Replace old objects
        self.state["objects"] = updated_objects

    def detect_scene_events(self, bounding_boxes=None, masks=None):
        """
        Detect scene events and return a list of events.
        Here, event is defined as an object touching or releasing the ROI boundary.
        """

        events = []

        if bounding_boxes is None and masks is None:
            logger.warning("No bounding boxes or masks provided for scene event detection.")
            return events
        
        self.assign_object_ids(bounding_boxes, masks)

        for obj in bounding_boxes:

            obj_id = obj["object_id"]
            obj_class = obj["class_name"]
            obj_mask = masks.get(obj_class, None)

            if obj_mask is None:
                logger.warning(f"No mask found for object class '{obj_class}'. Skipping event detection.")
                continue

            collision = self.roi.intersects_mask(
                mask=obj_mask,
                return_edges=True
            )
            touching = collision["touching"]
            edges = collision["edges"]
            erea = collision["area"]

            track = self.state["objects"].get(obj_id, {})
            prev = track.get("touching", False)

            event_type = None
            if touching and not prev:
                event_type = "ROI_TOUCH"
                self.state["objects"][obj_id]["touching"] = True
            elif not touching and prev:
                event_type = "ROI_RELEASE"
                self.state["objects"][obj_id]["touching"] = False

            if event_type:
                events.append({
                    "type": event_type,
                    "object_id": obj_id,
                    "class": obj_class,
                    "edges": edges,
                    "area": erea,
                    "area/ROI": float(erea /self.roi.calculate_ROI_area()) if self.roi else None,
                })

        logger.info(f"Detected {len(events)} scene events")
        return events


class Detector:
    """
    Main Detector class that provides a unified interface for different detection strategies.
    It initializes the appropriate detector based on the specified strategy and delegates
    the detection task to the initialized detector.
    """

    def __init__(self, strategy: str = "roi-events"):
        
        self.strategy = strategy
        self.detector = self._create_detector(strategy)
        logger.info(f"Detector initialized with strategy: {strategy}")

    def _create_detector(self, strategy: str):

        if strategy == "roi-events":
            return ROIEventsDetector()
        else:
            raise ValueError(f"Unknown detection strategy: {strategy}")

    def switch_strategy(self, new_strategy: str):
        """
        Switch the detection strategy at runtime.
        This allows for dynamic changes in detection behavior without restarting the application.
        """

        if new_strategy != self.strategy:
            self.strategy = new_strategy
            self.detector = self._create_detector(new_strategy)
            logger.info(f"Switched detection strategy to: {new_strategy}")
        else:
            logger.info(f"Detection strategy remains unchanged: {new_strategy}")

    def __call__(self, input: SegmentationResult, frame_id: int = 0, roi: Dict[str, Any] = None):

        return self.detector(input=input, frame_id=frame_id, roi=roi)
