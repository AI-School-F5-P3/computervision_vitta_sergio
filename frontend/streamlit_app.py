# frontend/streamlit_app.py
import streamlit as st
import cv2
import tempfile
from pytube import YouTube
import os
from app.video_processor import VideoProcessor
import numpy as np
from contextlib import contextmanager
import time

st.title("Logo Detection App")

# Initialize video processor
processor = VideoProcessor()

@contextmanager
def temporary_file(suffix=None):
    """Context manager para manejar archivos temporales de forma segura"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield temp_file
    finally:
        temp_file.close()
        # Añadir un pequeño delay antes de intentar eliminar el archivo
        time.sleep(0.1)
        try:
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)
        except Exception as e:
            st.error(f"No se pudo eliminar el archivo temporal: {str(e)}")

# File upload
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

# YouTube URL input
youtube_url = st.text_input("Or paste a YouTube URL")

if uploaded_file is not None:
    with temporary_file(suffix='.mp4') as tfile:
        # Guardar el archivo subido
        tfile.write(uploaded_file.read())
        tfile.flush()
        
        # Mostrar video
        st.video(tfile.name)
        
        if st.button('Process Video'):
            st.write("Processing video...")
            
            # Create a placeholder for the processed video
            video_placeholder = st.empty()
            
            try:
                # Process each frame
                for processed_frame in processor.process_video(tfile.name, uploaded_file.name):
                    # Convert frame to RGB for displaying
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb)
                
                st.write("Processing complete!")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")

elif youtube_url:
    if st.button('Process YouTube Video'):
        st.write("Downloading YouTube video...")
        
        try:
            with temporary_file(suffix='.mp4') as tfile:
                # Download YouTube video
                yt = YouTube(youtube_url)
                video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
                video_stream.download(filename=tfile.name)
                
                st.video(tfile.name)
                st.write("Processing video...")
                
                # Create a placeholder for the processed video
                video_placeholder = st.empty()
                
                # Process each frame
                for processed_frame in processor.process_video(tfile.name, yt.title):
                    # Convert frame to RGB for displaying
                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb)
                
                st.write("Processing complete!")
                
        except Exception as e:
            st.error(f"Error processing YouTube video: {str(e)}")

# Añadir información adicional en la sidebar
with st.sidebar:
    st.header("Information")
    st.write("This app detects logos in videos using a custom trained YOLO model.")
    st.write("Supported video formats:", ", ".join(['MP4', 'AVI', 'MOV']))
    
    # Mostrar estadísticas si están disponibles
    if 'total_detections' in st.session_state:
        st.header("Detection Statistics")
        st.write(f"Total detections: {st.session_state.total_detections}")
        st.write(f"Processed frames: {st.session_state.processed_frames}")