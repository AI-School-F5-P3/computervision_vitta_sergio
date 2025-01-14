from ultralytics import YOLO
from roboflow import Roboflow
import yaml
import os

def setup_training():
    """Descarga y prepara el dataset desde Roboflow"""
    # Reemplaza estas variables con tu información de Roboflow
    ROBOFLOW_API_KEY = "mAqOEuX2rkRv1WBe1KtU"
    WORKSPACE = "sergio-sxcoh"
    PROJECT = "logos-gep6x"
    VERSION = 1

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    
    return dataset.location

def train_model(data_yaml_path):
    """Entrena el modelo YOLOv8 con los parámetros optimizados"""
    # Cargar el modelo base
    model = YOLO('yolov8n.pt')  # 'n' es el más pequeño y rápido, puedes usar 's' o 'm' para más precisión
    
    # Configurar parámetros de entrenamiento
    results = model.train(
        data=data_yaml_path,
        epochs=100,          # Número de epochs
        imgsz=640,          # Tamaño de la imagen
        batch=16,           # Tamaño del batch
        patience=20,        # Early stopping
        optimizer='Adam',   # Optimizador
        lr0=0.001,         # Learning rate inicial
        weight_decay=0.0005,# Weight decay para regularización
        dropout=0.2,       # Dropout para reducir overfitting
        augment=True,      # Data augmentation
        mixup=0.1,         # Mixup augmentation
        copy_paste=0.1,    # Copy-paste augmentation
        degrees=10.0,      # Rotación máxima
        save=True,         # Guardar resultados
        project='runs/detect',
        name='logo_detector'
    )
    
    return results

def validate_model(model_path):
    """Valida el modelo entrenado"""
    model = YOLO(model_path)
    metrics = model.val()
    return metrics

def main():
    # 1. Preparar el dataset
    dataset_path = setup_training()
    data_yaml_path = os.path.join(dataset_path, 'data.yaml')
    
    # 2. Entrenar el modelo
    print("Iniciando entrenamiento...")
    results = train_model(data_yaml_path)
    
    # 3. Obtener la ruta del mejor modelo
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    # 4. Validar el modelo
    print("Validando modelo...")
    metrics = validate_model(best_model_path)
    
    # 5. Imprimir resultados
    print("\nEntrenamiento completado!")
    print(f"Mejor modelo guardado en: {best_model_path}")
    print("\nMétricas de validación:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    
if __name__ == "__main__":
    main()