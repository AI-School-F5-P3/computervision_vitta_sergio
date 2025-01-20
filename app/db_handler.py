# db_handler.py
import os
from datetime import datetime
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class DatabaseHandler:
    def __init__(self):
        self.conn_params = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }

    def get_connection(self):
        return psycopg2.connect(**self.conn_params)

    def create_tables(self):
        """Crear las tablas necesarias si no existen"""
        create_tables_query = """
        CREATE TABLE IF NOT EXISTS detections (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Se agrega timestamp
            file_name VARCHAR(255),
            file_type VARCHAR(50),
            source_type VARCHAR(50),
            total_detections INTEGER
        );

        CREATE TABLE IF NOT EXISTS detection_details (
            id SERIAL PRIMARY KEY,
            detection_id INTEGER REFERENCES detections(id) ON DELETE CASCADE,
            logo_name VARCHAR(100),
            confidence FLOAT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(create_tables_query)

    def save_detection(self, file_name, file_type, source_type, detections):
        """
        Guardar una detección y sus detalles
        
        Args:
            file_name (str): Nombre del archivo o URL
            file_type (str): Tipo de archivo (image/video/url/webcam)
            source_type (str): Origen (upload/sample/url/webcam)
            detections (list): Lista de detecciones [{'name': str, 'confidence': float}, ...]
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Insertar la detección principal
                cur.execute(
                    """
                    INSERT INTO detections (file_name, file_type, source_type, total_detections)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (file_name, file_type, source_type, len(detections))
                )
                detection_id = cur.fetchone()[0]

                # Insertar los detalles de cada detección
                for det in detections:
                    cur.execute(
                        """
                        INSERT INTO detection_details (detection_id, logo_name, confidence)
                        VALUES (%s, %s, %s)
                        """,
                        (detection_id, det['name'], det['confidence'])
                    )