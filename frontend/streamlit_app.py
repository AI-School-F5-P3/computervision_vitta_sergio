# frontend/streamlit_app.py
import streamlit as st
import cv2
import tempfile
from pytube import YouTube
import os
from app.video_processor import VideoProcessor
import numpy as np

st.title("Logo Detection App")

# Initialize video processor
processor = VideoProcessor()

# File upload
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])

# YouTube URL input
youtube_url = st.text_input("Or paste a YouTube URL")

if uploaded_file is not None:
    # Save uploaded file to temp directory
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Process video
    video_name = uploaded_file.name
    st.video(tfile.name)
    
    if st.button('Process Video'):
        st.write("Processing video...")
        
        # Create a placeholder for the processed video
        video_placeholder = st.empty()
        
        # Process each frame
        for processed_frame in processor.process_video(tfile.name, video_name):
            # Convert frame to RGB for displaying
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb)
        
        st.write("Processing complete!")
        
    # Clean up
    os.unlink(tfile.name)

elif youtube_url:
    if st.button('Process YouTube Video'):
        st.write("Downloading YouTube video...")
        
        try:
            # Download YouTube video
            yt = YouTube(youtube_url)
            video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
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
                
                # Clean up
                os.unlink(tfile.name)
                
        except Exception as e:
            st.error(f"Error processing YouTube video: {str(e)}")