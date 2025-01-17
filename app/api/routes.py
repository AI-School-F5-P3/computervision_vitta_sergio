# app/api/routes.py
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from ..video_processor import VideoProcessor
from ..database import SessionLocal, Detection
import tempfile
import os
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
from typing import List, Optional
from datetime import datetime
from pytube import YouTube
import numpy as np
import io

app = FastAPI(title="Logo Detection API")
processor = VideoProcessor()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp4', '.avi', '.mov')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Procesar video
            def generate_frames():
                for processed_frame in processor.process_video(temp_file.name, file.filename):
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Limpiar archivo temporal
            os.unlink(temp_file.name)
            
            return StreamingResponse(
                generate_frames(),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-youtube/")
async def process_youtube_video(url: str):
    try:
        # Descargar video de YouTube
        yt = YouTube(url)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            video_stream.download(filename=temp_file.name)
            
            # Procesar video
            def generate_frames():
                for processed_frame in processor.process_video(temp_file.name, yt.title):
                    _, buffer = cv2.imencode('.jpg', processed_frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Limpiar archivo temporal
            os.unlink(temp_file.name)
            
            return StreamingResponse(
                generate_frames(),
                media_type='multipart/x-mixed-replace; boundary=frame'
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detections/")
async def get_detections(
    video_name: Optional[str] = None,
    logo_class: Optional[str] = None,
    min_confidence: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=100, ge=1),
    offset: int = Query(default=0, ge=0)
):
    db = SessionLocal()
    try:
        query = db.query(Detection)
        
        if video_name:
            query = query.filter(Detection.video_name == video_name)
        if logo_class:
            query = query.filter(Detection.logo_class == logo_class)
        if min_confidence > 0:
            query = query.filter(Detection.confidence >= min_confidence)
            
        detections = query.offset(offset).limit(limit).all()
        
        return [{
            "id": d.id,
            "video_name": d.video_name,
            "logo_class": d.logo_class,
            "confidence": d.confidence,
            "timestamp": d.timestamp,
            "frame_number": d.frame_number,
            "bbox_coordinates": d.bbox_coordinates
        } for d in detections]
    finally:
        db.close()

@app.get("/detection/{detection_id}/image")
async def get_detection_image(detection_id: int):
    db = SessionLocal()
    try:
        detection = db.query(Detection).filter(Detection.id == detection_id).first()
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
            
        return StreamingResponse(
            io.BytesIO(detection.bbox_image),
            media_type="image/jpeg"
        )
    finally:
        db.close()

@app.get("/stats/")
async def get_stats():
    db = SessionLocal()
    try:
        total_detections = db.query(Detection).count()
        videos_processed = db.query(Detection.video_name).distinct().count()
        logo_classes = db.query(Detection.logo_class).distinct().all()
        avg_confidence = db.query(func.avg(Detection.confidence)).scalar()
        
        return {
            "total_detections": total_detections,
            "videos_processed": videos_processed,
            "logo_classes": [logo[0] for logo in logo_classes],
            "average_confidence": float(avg_confidence) if avg_confidence else 0
        }
    finally:
        db.close()

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime()}