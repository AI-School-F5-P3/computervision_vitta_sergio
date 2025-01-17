from ultralytics import YOLO
import yaml
import os
import torch
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from roboflow import Roboflow

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

def entrenar_modelo(api_key, workspace, project_name, version, epochs=150):
    print("Descargando dataset desde Roboflow...")
    ruta_dataset = obtener_dataset_roboflow(api_key, workspace, project_name, version)
    print(f"Dataset descargado en: {ruta_dataset}")
    
    # Inicializar modelo
    modelo = YOLO('yolov8n.pt')
    
    # Configurar parámetros optimizados para logos
    hiperparametros = {
        'epochs': epochs,
        'batch': 16,
        'imgsz': 640,
        'patience': 30,
        'optimizer': 'Adam',
        'lr0': 0.001,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'cos_lr': True,
        'close_mosaic': 10,
        'label_smoothing': 0.1,
        'device': '0' if torch.cuda.is_available() else 'cpu',
        'augment': True,
    }
    
    # Crear callback personalizado
    callback = EntrenamientoCallback(epochs)
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    print(f"Dispositivo utilizado: {hiperparametros['device']}")
    print(f"Épocas totales: {epochs}")
    
    try:
        # Usar el archivo data.yaml generado por Roboflow
        resultados = modelo.train(
            data=os.path.join(ruta_dataset, 'data.yaml'),
            callbacks=[callback],
            **hiperparametros
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
    # Configuración de Roboflow
    ROBOFLOW_API_KEY = "TU_API_KEY"  # Reemplaza con tu API key
    WORKSPACE = "TU_WORKSPACE"        # Reemplaza con tu workspace
    PROJECT_NAME = "TU_PROYECTO"      # Reemplaza con el nombre de tu proyecto
    VERSION = 1                       # Reemplaza con el número de versión
    
    print("=== Iniciando programa de entrenamiento de detección de logos ===")
    print(f"Proyecto Roboflow: {PROJECT_NAME}")
    print(f"Versión: {VERSION}")
    
    if torch.cuda.is_available():
        print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
    else:
        print("No se detectó GPU. El entrenamiento será más lento en CPU.")
    
    entrenar_modelo(
        api_key=ROBOFLOW_API_KEY,
        workspace=WORKSPACE,
        project_name=PROJECT_NAME,
        version=VERSION
    )