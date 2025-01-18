import streamlit as st
# Configuración de la página debe ser lo primero
st.set_page_config(
    page_title="Detector de Logos",
    page_icon="🔍",
    layout="wide"
)

import cv2
import numpy as np
from PIL import Image
import tempfile
import torch
import pafy
import urllib.request
from io import BytesIO
from ultralytics import YOLO
import time
import base64
import os

# Debug info
if st.sidebar.checkbox("Debug Info"):
    st.sidebar.write("Current directory:", os.getcwd())
    st.sidebar.write("Script location:", os.path.dirname(os.path.abspath(__file__)))
    
    # Listar archivos en src/samples
    samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'samples')
    if os.path.exists(samples_path):
        st.sidebar.write("Files in samples directory:", os.listdir(samples_path))
    else:
        st.sidebar.write("Samples directory not found:", samples_path)

# Función para cargar las imágenes de muestra
def load_sample_images():
    # Obtener la ruta absoluta del directorio actual
    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, 'src', 'samples')
    
    # Listar los archivos disponibles en el directorio
    if os.path.exists(samples_dir):
        available_files = os.listdir(samples_dir)
        
        # Definir las muestras solo con los archivos que realmente existen
        sample_files = {}
        for file in available_files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov')):
                # Usar el nombre del archivo como etiqueta, sin la extensión
                name = os.path.splitext(file)[0]
                sample_files[file] = name
    else:
        sample_files = {}
    
    return sample_files, samples_dir

# Función para cargar el modelo
@st.cache_resource
def load_model():
    model = YOLO('../train_data/trained_models/logo_model_v1_20250118_140314.pt')
    return model

# Cargar el modelo
model = load_model()

# Función para procesar imagen
def process_image(image):
    results = model(image, conf=0.25)
    
    # Obtenemos las predicciones
    boxes = results[0].boxes
    
    # Mostramos información sobre las detecciones
    if len(boxes) > 0:
        st.write(f"Se encontraron {len(boxes)} logos:")
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            st.write(f"- {class_name} (confianza: {conf:.2%})")
    
    # Retornamos la imagen con todas las detecciones dibujadas
    return results[0].plot()

# Función para procesar video
def process_video(video_file, conf_threshold=0.25, speed_options=None, selected_speed=None, detection_area=None):
    def process_video_frames():
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        
        # Usar el área central para la visualización
        video_display = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame y mostrar detecciones
            results = model(frame, conf=conf_threshold)
            boxes = results[0].boxes
            
            # Actualizar información de detecciones en el panel derecho
            detections_info = []
            for box in boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                detections_info.append(f"{class_name} ({conf:.2%})")
            
            if detections_info and detection_area is not None:
                detection_area.write("\n".join(detections_info))
            elif detection_area is not None:
                detection_area.write("No se detectaron logos")
                
            # Mostrar frame con detecciones
            frame_with_detections = results[0].plot()
            frame_with_detections_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
            video_display.image(frame_with_detections_rgb, use_container_width=True)
            
            # Controlar la velocidad de reproducción
            if speed_options and selected_speed:
                speed = speed_options[selected_speed]
                time.sleep(1/speed)
        
        cap.release()
    
    # Control de reproducción
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    
    if not st.session_state.video_processed:
        process_video_frames()
        st.session_state.video_processed = True
    
    # Botón para volver a procesar
    if st.button("Procesar video de nuevo"):
        st.session_state.video_processed = False
        process_video_frames()

# Función para procesar video de YouTube
def process_youtube_video(url):
    video = pafy.new(url)
    best = video.getbest(preftype="mp4")
    cap = cv2.VideoCapture(best.url)
    stframe = st.empty()
    info_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=0.25)
        boxes = results[0].boxes
        
        # Actualizar información de detecciones
        detections_info = []
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detections_info.append(f"{class_name} ({conf:.2%})")
        
        if detections_info:
            info_text.write(f"Logos detectados: {', '.join(detections_info)}")
        else:
            info_text.write("No se detectaron logos")
            
        frame_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb)
    
    cap.release()

# Función para procesar webcam
def process_webcam():
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    info_text = st.empty()
    
    # Botón de parada con key única
    stop_button_placeholder = st.empty()
    stop = stop_button_placeholder.button('Detener Webcam', key='stop_webcam_button')
    
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, conf=0.25)
        boxes = results[0].boxes
        
        # Actualizar información de detecciones
        detections_info = []
        for box in boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = model.names[cls]
            detections_info.append(f"{class_name} ({conf:.2%})")
        
        if detections_info:
            info_text.write(f"Logos detectados: {', '.join(detections_info)}")
        else:
            info_text.write("No se detectaron logos")
            
        frame_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, use_container_width=True)
        
        # Actualizar el botón de parada con key única
        stop = stop_button_placeholder.button('Detener Webcam', key=f'stop_webcam_button_{time.time()}')
        
        if stop or not st.session_state.run_webcam:
            break
    
    cap.release()
    st.session_state.run_webcam = False  # Resetear el estado cuando se detiene
    stframe.empty()  # Limpiar el frame
    info_text.empty()  # Limpiar el texto de información

# Función para codificar la imagen en base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Obtener la ruta absoluta de la imagen
def get_img_with_href(local_img_path):
    bin_str = get_base64_of_bin_file(local_img_path)
    return bin_str

# Ruta a la imagen de fondo
background_image_path = "./src/internet-3116062_1280.webp"
background_image = get_img_with_href(background_image_path)

# CSS personalizado para el fondo
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/webp;base64,{background_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    
    /* Para mejorar la legibilidad del contenido */
    .stMarkdown, .stButton, .stSelectbox, .stSlider {{
        background-color: #1A1C1F;  /* Color exacto del dropfile */
        padding: 10px;
        border-radius: 5px;
        color: white !important;
    }}
    
    /* Estilo para los selectbox y sliders */
    .stSlider [data-baseweb="slider"] {{
        background-color: #2A2D31;  /* Un poco más claro para contraste */
    }}
    
    .stSelectbox [data-baseweb="select"] {{
        background-color: #2A2D31;
    }}
    
    /* Estilo para los textos */
    .stMarkdown p {{
        color: white !important;
    }}
    
    h1, h3 {{
        color: white;
        text-shadow: 2px 2px 4px #000000;
    }}
    
    /* Estilo para los subheaders */
    .stSubheader {{
        color: white;
        background-color: #1A1C1F;
        padding: 10px;
        border-radius: 5px;
    }}
    
    /* Estilo para el área de upload */
    .uploadedFile {{
        background-color: #1A1C1F;
        padding: 10px;
        border-radius: 5px;
        color: white;
    }}

    /* Estilo para los botones */
    .stButton > button {{
        background-color: #2A2D31;
        color: white;
        border: none;
    }}

    .stButton > button:hover {{
        background-color: #3A3D41;
    }}
    
    /* Estilo para las imágenes de muestra */
    .sample-image {{
        border-radius: 5px;
        margin-bottom: 10px;
        background-color: #1A1C1F;
    }}
    
    /* Estilo para el contenedor de muestras */
    .stImage {{
        background-color: #1A1C1F;
        padding: 5px;
        border-radius: 5px;
    }}
    
    /* Estilo para los caption de las imágenes */
    .stImage img {{
        border-radius: 5px;
    }}
    
    .stImage caption {{
        color: #9BA1A6 !important;
        font-size: 12px;
        text-align: center;
        margin-top: 5px;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Crear tres columnas principales: izquierda (entrada), centro (visualización) y derecha (controles)
left_column, center_column, right_column = st.columns([1, 2, 1])

with left_column:
    # Samples from Test Set
    st.subheader("Samples")
    
    # Obtener las muestras
    sample_files, samples_dir = load_sample_images()
    
    # Crear un selectbox para elegir la muestra
    selected_sample = st.selectbox(
        "Selecciona una muestra",
        options=list(sample_files.keys()),
        format_func=lambda x: sample_files[x]  # Usar el nombre como etiqueta
    )
    
    # Botón para procesar la muestra seleccionada
    if st.button("Procesar muestra"):
        try:
            file_path = os.path.join(samples_dir, selected_sample)
            if selected_sample.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(file_path)
                processed_image = process_image(image)
                with center_column:
                    st.image(processed_image, caption="Muestra procesada", use_container_width=True)
            elif selected_sample.lower().endswith(('.mp4', '.avi', '.mov')):
                with center_column:
                    # Configuración de velocidad
                    speed_options = {
                        'Velocidad Normal (x1)': 1.0,
                        'Velocidad x3': 3.0,
                        'Velocidad x5': 5.0
                    }
                    
                    selected_speed = st.selectbox(
                        'Velocidad de reproducción',
                        options=list(speed_options.keys()),
                        index=0,
                        key='speed_selector'
                    )
                    
                    # Área para mostrar detecciones y video
                    detection_area = st.empty()
                    video_display = st.empty()
                    
                    # Procesar el video
                    cap = cv2.VideoCapture(file_path)
                    
                    try:
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reiniciar video al final
                                continue
                            
                            # Procesar frame y mostrar detecciones
                            results = model(frame, conf=0.25)
                            boxes = results[0].boxes
                            
                            # Actualizar información de detecciones
                            detections_info = []
                            for box in boxes:
                                conf = float(box.conf[0])
                                cls = int(box.cls[0])
                                class_name = model.names[cls]
                                detections_info.append(f"{class_name} ({conf:.2%})")
                            
                            if detections_info:
                                detection_area.write("\n".join(detections_info))
                            else:
                                detection_area.write("No se detectaron logos")
                            
                            # Mostrar frame con detecciones
                            frame_with_detections = results[0].plot()
                            frame_with_detections_rgb = cv2.cvtColor(frame_with_detections, cv2.COLOR_BGR2RGB)
                            video_display.image(frame_with_detections_rgb, use_container_width=True)
                            
                            # Controlar la velocidad de reproducción
                            speed = speed_options[selected_speed]
                            time.sleep(1/speed)
                            
                    except Exception as e:
                        st.error(f"Error durante la reproducción: {str(e)}")
                    finally:
                        cap.release()
                
        except Exception as e:
            st.error(f"Error procesando la muestra: {str(e)}")
    
    # Upload Image or Video File
    st.subheader("Upload Image or Video File")
    uploaded_file = st.file_uploader("Drop file here or", type=['png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'])
    
    # Paste YouTube or Image URL
    st.subheader("Paste YouTube or Image URL")
    url_input = st.text_input("Paste a link...", key="url_input")
    
    # Try With Webcam
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False
    
    webcam_col1, webcam_col2 = st.columns([1, 1])
    with webcam_col1:
        if st.button("Iniciar Webcam", key='start_webcam_button'):
            st.session_state.run_webcam = True
    

with center_column:

    st.markdown("<h1 style='text-align: center;'>Detector de Logos de Marcas</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>Detecta logos de Adidas, Reebok, Nike, Apple, Sony y Samsung</h3>", unsafe_allow_html=True)
    
    # Área de visualización
    if uploaded_file is not None:
        if uploaded_file.type.startswith('image'):
            image = Image.open(uploaded_file)
            processed_image = process_image(image)
            st.image(processed_image, caption="Processed Image", use_container_width=True)
        else:
            # Área para mostrar detecciones
            detection_area = st.empty()
            
            # Configuración de velocidad
            speed_options = {
                'Velocidad Normal (x1)': 1.0,
                'Velocidad x3': 3.0,
                'Velocidad x5': 5.0
            }
            selected_speed = st.selectbox(
                'Velocidad de reproducción',
                options=list(speed_options.keys()),
                index=0
            )
            
            # Obtener el umbral de confianza del slider
            confidence = st.slider("Confidence Threshold:", 0, 100, 50, format="%d%%")
            conf_threshold = confidence / 100
            
            # Procesar el video con los parámetros
            process_video(
                uploaded_file, 
                conf_threshold=conf_threshold,
                speed_options=speed_options,
                selected_speed=selected_speed,
                detection_area=detection_area
            )
    elif url_input:
        if 'youtube.com' in url_input or 'youtu.be' in url_input:
            process_youtube_video(url_input)
        else:
            try:
                response = urllib.request.urlopen(url_input)
                image_data = response.read()
                image = Image.open(BytesIO(image_data))
                processed_image = process_image(image)
                st.image(processed_image, caption="Processed Image", use_container_width=True)
            except:
                st.error("Invalid URL or unable to process the image")
    elif st.session_state.run_webcam:
        process_webcam()

with right_column:
    # Controles y configuración
    st.subheader("Configuración")
    
    # Control de confianza
    confidence = st.slider("Confidence Threshold:", 0, 100, 50, format="%d%%")
    conf_threshold = confidence / 100
    
    # Control de superposición
    overlap = st.slider("Overlap Threshold:", 0, 100, 50, format="%d%%")
    
    # Modo de visualización
    display_mode = st.selectbox(
        "Label Display Mode:",
        ["Draw Confidence"]
    )
    
    # Si hay un video, mostrar control de velocidad
    if 'video_processed' in st.session_state:
        st.subheader("Control de Video")
        speed_options = {
            'Velocidad Normal (x1)': 1.0,
            'Velocidad x3': 3.0,
            'Velocidad x5': 5.0
        }
        selected_speed = st.selectbox(
            'Velocidad de reproducción',
            options=list(speed_options.keys()),
            index=0
        )
    
    # Área para mostrar detecciones
    st.subheader("Detecciones")
    detection_area = st.empty()

# Información en el sidebar
with st.sidebar:
    st.header("Información")
    st.markdown("""
    Este detector puede identificar los siguientes logos:
    - Adidas
    - Reebok
    - Nike
    - Apple
    - Sony
    - Samsung
    """)
