import cv2
import numpy as np
from ultralytics import YOLO
import torch

class YOLOv11Detector:
    def __init__(self, model_path, conf_threshold=0.7, min_area=500):
        """
        Initialize YOLOv11 detector with the fine-tuned model
        
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.min_area = min_area 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
    def detect(self, frame):
        """Detection with filtering
        returns list of detections in format [(x, y, w, h), confidence, class_id]"""
        results = self.model(frame, conf=self.conf_threshold, device=self.device)
        
        detections = []
        frame_height, frame_width = frame.shape[:2]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # coordinates extraction
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Calculate bounding box area
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Filter out noise
                    if area < self.min_area:
                        continue
                    
                    # Filter out detections at bottom edge (like grasses)
                    if y2 > frame_height * 0.95:  # Bottom 5% of frame
                        continue
                    
                    # Filter by aspect ratio (players should be taller than wide)
                    aspect_ratio = height / width if width > 0 else 0
                    if aspect_ratio < 1.2:  # Players should be at least 1.2x taller than wide
                        continue
                    
                    x, y, w, h = x1, y1, width, height
                    detections.append(([x, y, w, h], confidence, class_id))
        
        return detections