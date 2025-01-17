# app/utils.py
import cv2
import numpy as np
import tempfile
import os
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import logging
from datetime import datetime
from pytube import YouTube
import hashlib
import json
from PIL import Image
import io
import base64
from .config import settings

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """Crea los directorios necesarios para el proyecto."""
    dirs = [
        settings.STORAGE_PATH,
        os.path.dirname(settings.MODEL_PATH),
        'temp',
        'logs'
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory checked/created: {dir_path}")

def download_youtube_video(url: str) -> Tuple[str, str]:
    """
    Descarga un video de YouTube y lo guarda temporalmente.
    
    Args:
        url: URL del video de YouTube
        
    Returns:
        Tuple[str, str]: (ruta del archivo temporal, título del video)
    """
    try:
        yt = YouTube(url)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        
        # Crear nombre de archivo temporal único
        temp_dir = Path('temp')
        temp_dir.mkdir(exist_ok=True)
        temp_filename = f"{hashlib.md5(url.encode()).hexdigest()}.mp4"
        temp_path = temp_dir / temp_filename
        
        # Descargar video
        video_stream.download(output_path=str(temp_dir), filename=temp_filename)
        logger.info(f"Video downloaded: {yt.title}")
        
        return str(temp_path), yt.title
    except Exception as e:
        logger.error(f"Error downloading YouTube video: {str(e)}")
        raise

def get_video_info(video_path: str) -> Dict:
    """
    Obtiene información sobre un archivo de video.
    
    Args:
        video_path: Ruta al archivo de video
        
    Returns:
        Dict con información del video
    """
    cap = cv2.VideoCapture(video_path)
    info = {
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fps': int(cap.get(cv2.CAP_PROP_FPS)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS))
    }
    cap.release()
    return info

def frame_to_base64(frame: np.ndarray) -> str:
    """
    Convierte un frame de video a base64 para transmisión web.
    
    Args:
        frame: Array numpy del frame
        
    Returns:
        String en formato base64
    """
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def save_detection_image(image: np.ndarray, detection_id: int) -> str:
    """
    Guarda una imagen de detección en el sistema de archivos.
    
    Args:
        image: Array numpy de la imagen
        detection_id: ID de la detección
        
    Returns:
        Ruta donde se guardó la imagen
    """
    save_path = Path(settings.STORAGE_PATH) / f"detection_{detection_id}.jpg"
    cv2.imwrite(str(save_path), image)
    return str(save_path)

def validate_video_file(file_path: str) -> bool:
    """
    Valida si un archivo de video es válido y cumple con los requisitos.
    
    Args:
        file_path: Ruta al archivo de video
        
    Returns:
        bool indicando si el archivo es válido
    """
    # Verificar extensión
    if not any(file_path.lower().endswith(ext) for ext in settings.SUPPORTED_FORMATS):
        logger.warning(f"Invalid video format: {file_path}")
        return False
    
    # Verificar tamaño
    if os.path.getsize(file_path) > settings.MAX_VIDEO_SIZE:
        logger.warning(f"Video file too large: {file_path}")
        return False
    
    # Verificar si se puede abrir
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video file: {file_path}")
        cap.release()
        return False
    
    cap.release()
    return True

def create_video_writer(video_path: str, output_path: Optional[str] = None) -> Tuple[cv2.VideoWriter, str]:
    """
    Crea un escritor de video para guardar los frames procesados.
    
    Args:
        video_path: Ruta al video original
        output_path: Ruta opcional para el video de salida
        
    Returns:
        Tuple[cv2.VideoWriter, str]: (objeto writer, ruta de salida)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"processed_{timestamp}.mp4"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    cap.release()
    return writer, output_path

def cleanup_temp_files(max_age_hours: int = 24) -> None:
    """
    Limpia archivos temporales antiguos.
    
    Args:
        max_age_hours: Edad máxima de los archivos en horas
    """
    temp_dir = Path('temp')
    if not temp_dir.exists():
        return
        
    current_time = datetime.now()
    for file_path in temp_dir.glob('*'):
        file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
        if file_age.total_seconds() > max_age_hours * 3600:
            try:
                file_path.unlink()
                logger.info(f"Deleted old temp file: {file_path}")
            except Exception as e:
                logger.error(f"Error deleting temp file {file_path}: {str(e)}")

def generate_unique_filename(prefix: str = "", suffix: str = "") -> str:
    """
    Genera un nombre de archivo único basado en timestamp y hash.
    
    Args:
        prefix: Prefijo para el nombre del archivo
        suffix: Sufijo para el nombre del archivo
        
    Returns:
        Nombre de archivo único
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hash = hashlib.md5(str(datetime.now().timestamp()).encode()).hexdigest()[:8]
    return f"{prefix}{timestamp}_{random_hash}{suffix}"

def get_color_by_class(class_name: str) -> Tuple[int, int, int]:
    """
    Genera un color consistente para cada clase de logo.
    
    Args:
        class_name: Nombre de la clase
        
    Returns:
        Tuple[int, int, int]: Color en formato BGR
    """
    # Usar hash del nombre de la clase para generar color
    hash_value = int(hashlib.md5(class_name.encode()).hexdigest()[:6], 16)
    return (hash_value & 255, (hash_value >> 8) & 255, (hash_value >> 16) & 255)

def create_result_summary(detections: List[Dict]) -> Dict:
    """
    Crea un resumen de las detecciones realizadas.
    
    Args:
        detections: Lista de detecciones
        
    Returns:
        Dict con el resumen de resultados
    """
    class_counts = {}
    total_confidence = 0
    total_detections = len(detections)
    
    for detection in detections:
        class_name = detection['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        total_confidence += detection['confidence']
    
    return {
        'total_detections': total_detections,
        'class_distribution': class_counts,
        'average_confidence': total_confidence / total_detections if total_detections > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }