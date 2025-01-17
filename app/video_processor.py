# app/video_processor.py
import cv2
import torch
from ultralytics import YOLO
import os
from dotenv import load_dotenv
from .database import SessionLocal, Detection
import numpy as np
from datetime import datetime

load_dotenv()

class VideoProcessor:
    def __init__(self):
        self.model = YOLO(os.getenv('MODEL_PATH'))
        self.conf_threshold = float(os.getenv('CONFIDENCE_THRESHOLD'))

    def process_frame(self, frame, frame_number, video_name):
        results = self.model(frame)[0]
        annotated_frame = frame.copy()
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if score > self.conf_threshold:
                class_name = self.model.names[int(class_id)]
                confidence = f"{score:.2%}"
                
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Add label with confidence
                label = f"{class_name} {confidence}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop bbox image
                bbox_image = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Create detection object
                detection = {
                    'frame': annotated_frame,
                    'bbox_image': bbox_image,
                    'class_name': class_name,
                    'confidence': score,
                    'coordinates': f"{x1},{y1},{x2},{y2}",
                    'frame_number': frame_number
                }
                detections.append(detection)

                # Save to database
                self.save_detection(detection, video_name)

        return annotated_frame, detections

    def save_detection(self, detection, video_name):
        db = SessionLocal()
        try:
            _, img_encoded = cv2.imencode('.jpg', detection['bbox_image'])
            db_detection = Detection(
                video_name=video_name,
                logo_class=detection['class_name'],
                confidence=detection['confidence'],
                bbox_image=img_encoded.tobytes(),
                frame_number=detection['frame_number'],
                bbox_coordinates=detection['coordinates']
            )
            db.add(db_detection)
            db.commit()
        finally:
            db.close()

    def process_video(self, video_path, video_name):
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame, _ = self.process_frame(frame, frame_count, video_name)
            frame_count += 1
            
            yield processed_frame
            
        cap.release()