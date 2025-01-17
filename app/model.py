# app/model.py
import torch
from ultralytics import YOLO
import cv2
import numpy as np
from .config import settings
import logging
from typing import List, Dict, Tuple, Optional
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LogoDetector:
    def __init__(self):
        """
        Inicializa el detector de logos con el modelo YOLO entrenado
        """
        try:
            # Verificar que el archivo del modelo existe
            if not os.path.exists(settings.MODEL_PATH):
                raise FileNotFoundError(f"Model file not found at {settings.MODEL_PATH}")

            # Verificar CUDA
            if torch.cuda.is_available():
                logger.info("CUDA is available. Using GPU.")
                self.device = 'cuda'
            else:
                logger.info("CUDA is not available. Using CPU.")
                self.device = 'cpu'

            # Cargar el modelo con manejo de errores específico
            try:
                self.model = YOLO(settings.MODEL_PATH)
                self.model.to(self.device)
            except Exception as e:
                logger.error(f"Error loading YOLO model: {str(e)}")
                raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

            # Configurar threshold
            self.conf_threshold = settings.CONFIDENCE_THRESHOLD
            
            # Verificar que el modelo se cargó correctamente
            if not hasattr(self.model, 'names'):
                raise RuntimeError("Model loaded but classes are not available")
                
            logger.info(f"Model loaded successfully. Available classes: {self.model.names}")
            
        except Exception as e:
            logger.error(f"Error initializing LogoDetector: {str(e)}")
            raise

    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Realiza predicciones sobre una imagen

        Args:
            image (np.ndarray): Imagen en formato BGR (OpenCV)

        Returns:
            List[Dict]: Lista de detecciones
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image input")
            return []

        try:
            # Convertir imagen a tensor y mover al dispositivo correcto
            with torch.no_grad():
                results = self.model(image, device=self.device)[0]
            
            detections = []
            if hasattr(results, 'boxes') and hasattr(results.boxes, 'data'):
                for r in results.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = r
                    if score > self.conf_threshold:
                        class_name = self.model.names[int(class_id)]
                        bbox_image = image[int(y1):int(y2), int(x1):int(x2)]
                        
                        detection = {
                            'class_name': class_name,
                            'confidence': score,
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'bbox_image': bbox_image if bbox_image.size > 0 else None
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return []

    def draw_predictions(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Dibuja las predicciones en la imagen

        Args:
            image (np.ndarray): Imagen original
            detections (List[Dict]): Lista de detecciones del método predict()

        Returns:
            np.ndarray: Imagen con las detecciones dibujadas
        """
        if image is None or not isinstance(image, np.ndarray):
            logger.error("Invalid image input")
            return np.array([])

        try:
            annotated_image = image.copy()
            
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                class_name = detection['class_name']
                confidence = detection['confidence']

                # Dibujar bounding box
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Preparar texto
                label = f"{class_name} {confidence:.2%}"
                
                # Obtener tamaño del texto
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # Dibujar fondo para el texto
                cv2.rectangle(
                    annotated_image,
                    (x1, y1 - label_height - baseline - 10),
                    (x1 + label_width, y1),
                    (0, 255, 0),
                    -1
                )
                
                # Dibujar texto
                cv2.putText(
                    annotated_image,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    2
                )

            return annotated_image

        except Exception as e:
            logger.error(f"Error drawing predictions: {str(e)}")
            return image

    def get_model_info(self) -> Dict:
        """
        Devuelve información sobre el modelo cargado
        """
        try:
            return {
                "model_path": settings.MODEL_PATH,
                "device": self.device,
                "confidence_threshold": self.conf_threshold,
                "classes": self.model.names if hasattr(self.model, 'names') else [],
                "model_type": str(type(self.model)),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None
            }
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}