import os
from dotenv import load_dotenv
from pathlib import Path

# Cargar variables de entorno
load_dotenv()

class Settings:
    # Rutas del proyecto
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODEL_PATH = os.getenv('MODEL_PATH', 'data/models/best.pt')
    STORAGE_PATH = os.getenv('STORAGE_PATH', 'storage/detections')

    # Configuración de la base de datos
    DB_USER = os.getenv('DB_USER', 'postgres')
    DB_PASSWORD = os.getenv('DB_PASSWORD', 'postgres')
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_PORT = os.getenv('DB_PORT', '5432')
    DB_NAME = os.getenv('DB_NAME', 'logo_detection')
    
    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    # Configuración del modelo
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
    DEVICE = os.getenv('DEVICE', 'cuda' if os.path.exists('/opt/cuda') else 'cpu')

    # Configuración de la API
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', '8000'))

    # Configuración de procesamiento de video
    MAX_VIDEO_SIZE = int(os.getenv('MAX_VIDEO_SIZE', '100000000'))  # 100MB en bytes
    SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov']
    
    # Configuración de almacenamiento
    SAVE_DETECTIONS = os.getenv('SAVE_DETECTIONS', 'True').lower() == 'true'
    
    def __init__(self):
        # Crear directorios necesarios si no existen
        os.makedirs(self.STORAGE_PATH, exist_ok=True)
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)

# Instancia global de configuración
settings = Settings()