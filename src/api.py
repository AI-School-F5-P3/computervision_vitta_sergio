from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import shutil
import tempfile
from pathlib import Path
from typing import List
from pydantic import BaseModel
from logo_detector import LogoDetector

app = FastAPI()
detector = LogoDetector("path/to/best.pt")

class DetectionResponse(BaseModel):
    video_name: str
    total_frames: int
    fps: float
    statistics: dict
    detections: List[dict]

@app.post("/detect-logos/", response_model=DetectionResponse)
async def detect_logos(video: UploadFile = File(...)):
    """
    Endpoint para procesar un video y detectar logos
    """
    # Guardar el archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename).suffix) as tmp:
        shutil.copyfileobj(video.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Procesar el video
        results = detector.process_video(tmp_path)
        return results
    finally:
        # Limpiar archivo temporal
        Path(tmp_path).unlink()

@app.get("/statistics/{video_name}")
async def get_video_statistics(video_name: str):
    """
    Obtener estadísticas de un video específico
    """
    cursor = detector.db_connection.cursor()
    cursor.execute("""
        SELECT 
            class_name,
            COUNT(*) as total_detections,
            AVG(confidence) as avg_confidence
        FROM detections
        WHERE video_name = %s
        GROUP BY class_name
    """, (video_name,))
    
    stats = {}
    for row in cursor.fetchall():
        stats[row[0]] = {
            "total_detections": row[1],
            "avg_confidence": float(row[2])
        }
    
    cursor.close()
    return JSONResponse(content=stats)