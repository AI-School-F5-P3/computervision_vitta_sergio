import cv2
import torch
from ultralytics import YOLO
from datetime import datetime
import numpy as np
import psycopg2
from pathlib import Path

class LogoDetector:
    def __init__(self, model_path):
        """
        Inicializa el detector de logos
        Args:
            model_path: Ruta al modelo YOLOv8 entrenado
        """
        self.model = YOLO(model_path)
        self.db_connection = self.connect_to_db()
        
    def connect_to_db(self):
        """Conecta a la base de datos PostgreSQL"""
        return psycopg2.connect(
            dbname="logo_detection",
            user="user",
            password="password",
            host="localhost"
        )

    def process_video(self, video_path):
        """
        Procesa un video y detecta logos
        Args:
            video_path: Ruta al archivo de video
        Returns:
            dict: Resultados del análisis
        """
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results = {
            'video_name': Path(video_path).name,
            'total_frames': total_frames,
            'fps': fps,
            'detections': [],
            'statistics': {}
        }
        
        frame_count = 0
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
                
            # Detectar logos en el frame actual
            detections = self.model(frame)[0]
            
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = detection
                class_name = self.model.names[int(class_id)]
                
                # Guardar la detección
                detection_info = {
                    'frame': frame_count,
                    'time': frame_count / fps,
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2]
                }
                results['detections'].append(detection_info)
                
                # Guardar el crop del logo
                if conf > 0.5:  # Solo guardar detecciones con alta confianza
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    self.save_detection_to_db(
                        results['video_name'],
                        frame_count,
                        class_name,
                        conf,
                        crop
                    )
            
            frame_count += 1
            
        video.release()
        
        # Calcular estadísticas
        results['statistics'] = self.calculate_statistics(
            results['detections'],
            total_frames,
            fps
        )
        
        return results

    def calculate_statistics(self, detections, total_frames, fps):
        """Calcula estadísticas de las detecciones"""
        stats = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in stats:
                stats[class_name] = {
                    'total_appearances': 0,
                    'total_frames': 0,
                    'avg_confidence': 0,
                    'time_percentage': 0
                }
            stats[class_name]['total_appearances'] += 1
            stats[class_name]['total_frames'] += 1
            stats[class_name]['avg_confidence'] += detection['confidence']
        
        # Calcular promedios y porcentajes
        for class_name in stats:
            stats[class_name]['avg_confidence'] /= stats[class_name]['total_appearances']
            stats[class_name]['time_percentage'] = (
                stats[class_name]['total_frames'] / total_frames
            ) * 100
            
        return stats

    def save_detection_to_db(self, video_name, frame, class_name, confidence, crop_image):
        """Guarda la detección en PostgreSQL"""
        cursor = self.db_connection.cursor()
        
        # Convertir imagen a bytes para almacenar en la DB
        _, img_encoded = cv2.imencode('.jpg', crop_image)
        img_bytes = img_encoded.tobytes()
        
        cursor.execute("""
            INSERT INTO detections 
            (video_name, frame_number, class_name, confidence, crop_image, detection_time)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            video_name,
            frame,
            class_name,
            confidence,
            img_bytes,
            datetime.now()
        ))
        
        self.db_connection.commit()
        cursor.close()