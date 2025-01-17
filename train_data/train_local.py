from ultralytics import YOLO
import yaml
import os
import torch
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from roboflow import Roboflow

def verificar_gpu():
    print("\n=== Información del Sistema ===")
    print(f"PyTorch versión: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA versión: {torch.version.cuda}")
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
    print("===========================\n")

def obtener_dataset_roboflow(api_key, workspace, project_name, version):
    """Descarga el dataset desde Roboflow"""
    rf = Roboflow(api_key=api_key)
    proyecto = rf.workspace(workspace).project(project_name)
    dataset = proyecto.version(version).download("yolov8")
    return dataset.location

class EntrenamientoCallback:
    def __init__(self, epochs):
        self.epochs = epochs
        self.tiempo_inicio = time.time()
        self.mejor_map = 0
        
    def on_train_epoch_start(self, trainer):
        self.epoch_inicio = time.time()
        print(f"\nIniciando época {trainer.epoch + 1}/{self.epochs}")
        
    def on_train_epoch_end(self, trainer):
        tiempo_transcurrido = time.time() - self.tiempo_inicio
        tiempo_por_epoca = tiempo_transcurrido / (trainer.epoch + 1)
        epocas_restantes = self.epochs - (trainer.epoch + 1)
        tiempo_restante = epocas_restantes * tiempo_por_epoca
        
        tiempo_restante = str(timedelta(seconds=int(tiempo_restante)))
        
        if hasattr(trainer, 'metrics'):
            map_actual = trainer.metrics.get('map50-95', 0)
            if map_actual > self.mejor_map:
                self.mejor_map = map_actual
                print(f"¡Nuevo mejor mAP: {self.mejor_map:.4f}!")
        
        print(f"Época {trainer.epoch + 1} completada")
        print(f"Tiempo restante estimado: {tiempo_restante}")
        print(f"Mejor mAP hasta ahora: {self.mejor_map:.4f}")

def entrenar_modelo(epochs=150):
    print("Usando dataset local...")
    # Construir ruta al dataset local
    ruta_base = Path(__file__).parent
    ruta_dataset = ruta_base / 'dataset' / 'logos'
    ruta_yaml = ruta_dataset / 'data.yaml'
    
    print(f"Dataset ubicado en: {ruta_dataset}")
    print(f"Archivo de configuración: {ruta_yaml}")
    
    # Verificar CUDA
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
        print(f"Versión CUDA: {torch.version.cuda}")
        device = 0  # Usar primera GPU
    else:
        print("ADVERTENCIA: No se detectó GPU. El entrenamiento será muy lento en CPU.")
        print("Considera instalar CUDA y PyTorch con soporte CUDA")
        device = 'cpu'
    
    # Verificar que existe el dataset
    if not ruta_dataset.exists():
        raise FileNotFoundError(f"No se encontró el dataset en {ruta_dataset}")
    if not ruta_yaml.exists():
        raise FileNotFoundError(f"No se encontró el archivo data.yaml en {ruta_yaml}")
    
    # Inicializar modelo
    modelo = YOLO('yolov8n.pt')
    
    try:
        # Entrenar modelo con los parámetros directamente
        resultados = modelo.train(
            data=str(ruta_yaml),
            epochs=epochs,
            batch=16,  # Aumentar si hay suficiente memoria GPU
            imgsz=640,
            patience=30,
            optimizer='Adam',
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3,
            cos_lr=True,
            close_mosaic=10,
            device=device,  # Especificar dispositivo
            augment=True
        )
        
        # Evaluar modelo
        print("\nEvaluando modelo...")
        metricas = modelo.val()
        
        # Guardar resultados
        print("\nGuardando resultados...")
        fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Exportar modelo en diferentes formatos
        modelo.export(format='onnx', filename=f'modelo_logos_{fecha_actual}.onnx')
        
        # Guardar métricas
        with open(f'metricas_{fecha_actual}.txt', 'w') as f:
            f.write(f"mAP: {metricas.box.map:.4f}\n")
            f.write(f"Precisión: {metricas.box.p:.4f}\n")
            f.write(f"Recall: {metricas.box.r:.4f}\n")
        
        print("\n¡Entrenamiento completado con éxito!")
        print(f"Métricas finales:")
        print(f"mAP: {metricas.box.map:.4f}")
        print(f"Precisión: {metricas.box.p:.4f}")
        print(f"Recall: {metricas.box.r:.4f}")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        raise

if __name__ == "__main__":
    verificar_gpu()
    print("=== Iniciando programa de entrenamiento de detección de logos ===")
    
    entrenar_modelo(epochs=150)