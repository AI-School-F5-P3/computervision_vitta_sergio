import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import torch

# Configuración de la página
st.set_page_config(
    page_title="Detector de Logos",
    page_icon="🔍",
    layout="wide"
)

# Título del dashboard
st.title("Detector de Logos de Marcas")
st.markdown("Detecta logos de Adidas, Reebok, Nike, Apple, Sony y Samsung")

# Función para cargar el modelo
@st.cache_resource
def load_model():
    # Cargar el modelo entrenado
    model = torch.hub.load('ultralytics/yolov11', 'custom', path='path/to/your/model.pt')
    return model

# Cargar el modelo
model = load_model()

# Función para procesar imagen
def process_image(image):
    # Realizar la predicción
    results = model(image)
    
    # Convertir los resultados a una imagen con las detecciones
    return results.render()[0]

# Función para procesar video
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Realizar la predicción en el frame
        results = model(frame)
        
        # Convertir BGR a RGB
        frame_rgb = cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB)
        
        # Mostrar el frame
        stframe.image(frame_rgb)
    
    cap.release()

# Selector de tipo de entrada
input_type = st.radio("Selecciona el tipo de entrada:", ["Imagen", "Video"])

if input_type == "Imagen":
    # Upload de imagen
    uploaded_file = st.file_uploader("Sube una imagen", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Convertir el archivo a imagen
        image = Image.open(uploaded_file)
        
        # Mostrar imagen original
        st.subheader("Imagen Original")
        st.image(image)
        
        # Procesar y mostrar imagen con detecciones
        st.subheader("Detecciones")
        processed_image = process_image(image)
        st.image(processed_image)

else:
    # Upload de video
    uploaded_file = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_file is not None:
        st.video(uploaded_file)
        
        if st.button("Procesar Video"):
            process_video(uploaded_file)

# Información adicional
st.sidebar.header("Información")
st.sidebar.markdown("""
Este detector puede identificar los siguientes logos:
- Adidas
- Reebok
- Nike
- Apple
- Sony
- Samsung
""")
