# Detector de Logos de Marcas

Esta aplicación utiliza un modelo de detección de objetos para identificar logos de marcas en imágenes y videos. Está diseñada para detectar logos de Adidas, Nike, Apple y Samsung.

## Características

- **Detección de Logos:** Identifica logos de marcas en imágenes y videos.
- **Interfaz de Usuario Intuitiva:** Utiliza Streamlit para una experiencia de usuario sencilla y directa.
- **Soporte de Video:** Procesa videos y permite detener la reproducción.
- **Configuración Personalizable:** Ajusta los umbrales de confianza y superposición.

## Requisitos

- Python 3.x
- Streamlit
- OpenCV
- PIL
- Ultralytics YOLOv8

## Instalación

1. Clona este repositorio:
   ```bash
   git clone git@github.com:AI-School-F5-P3/computervision_vitta_sergio.git
   ```
2. Navega al directorio del proyecto:
   ```bash
   cd computervision_vitta_sergio
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

1. Ejecuta la aplicación:
   ```bash
   streamlit run app/main.py
   ```
2. Abre el navegador en la dirección que aparece en la terminal (por defecto `http://localhost:8501`).

## Funcionalidades

- **Cargar Imágenes o Videos:** Puedes subir archivos de imagen o video para procesar.
- **Detección en Tiempo Real:** Procesa videos en tiempo real y muestra las detecciones.
- **Control de Reproducción:** Detén la reproducción de videos con un botón.
- **Visualización de Detecciones:** Muestra las detecciones en un panel a la derecha.

## Entrenamiento del Modelo

El modelo fue entrenado utilizando más de 6000 imágenes etiquetadas de logos de marcas. Aquí se detallan los pasos y parámetros utilizados:

1. **Dataset:** 
   - Se utilizó un conjunto de datos con más de 6000 imágenes, cada una etiquetada con las posiciones de los logos de las marcas.

2. **Modelo:** 
   - Se utilizó el modelo YOLO (You Only Look Once) de Ultralytics, conocido por su rapidez y precisión en la detección de objetos.

3. **Parámetros de Entrenamiento:**
   - **Épocas:** El modelo fue entrenado durante 150 épocas para asegurar una buena convergencia.
   - **Tamaño del Lote:** Se utilizó un tamaño de lote de 32 para equilibrar el uso de memoria y la velocidad de entrenamiento.
   - **Tasa de Aprendizaje:** Se configuró una tasa de aprendizaje inicial de 0.001, ajustada dinámicamente durante el entrenamiento.
   - **Aumento de Datos:** Se aplicaron técnicas de aumento de datos como rotación, cambio de escala y ajuste de brillo para mejorar la robustez del modelo.

4. **Evaluación:**
   - El modelo fue evaluado utilizando un conjunto de validación separado, logrando una precisión del 85% en la detección de logos.

## Estructura del Proyecto

- `app/main.py`: Código principal de la aplicación.
- `train_data