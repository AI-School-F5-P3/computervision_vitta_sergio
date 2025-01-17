from ultralytics import YOLO
from pathlib import Path
import yaml

# Buscar el modelo en diferentes ubicaciones posibles
posibles_rutas = [
    Path('runs/detect/train6/weights/best.pt'),
    Path('../runs/detect/train6/weights/best.pt'),
    Path('modelos_entrenados').glob('*.pt')  # Buscar cualquier modelo .pt en la carpeta
]

# Encontrar una ruta válida
modelo_path = None
for ruta in posibles_rutas:
    if isinstance(ruta, Path):
        if ruta.exists():
            modelo_path = str(ruta)
            break
    else:  # Para el caso de glob
        modelos = list(ruta)
        if modelos:
            modelo_path = str(modelos[0])
            break

if modelo_path is None:
    raise FileNotFoundError("No se encontró ningún modelo entrenado. Verifica las rutas.")

print(f"Cargando modelo desde: {modelo_path}")
modelo = YOLO(modelo_path)

# Buscar el archivo data.yaml
yaml_paths = [
    Path('dataset/logos/data.yaml'),
    Path('train_data/dataset/logos/data.yaml'),
    Path('../train_data/dataset/logos/data.yaml')
]

yaml_data = None
for yaml_path in yaml_paths:
    if yaml_path.exists():
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        break

print("\nClases que detecta el modelo:")
if yaml_data and 'names' in yaml_data:
    # Mostrar nombres originales del yaml
    print("\nNombres originales del dataset:")
    for idx, name in enumerate(yaml_data['names']):
        print(f"Clase {idx}: {name}")
    
    # Mostrar estadísticas por clase
    print("\nEstadísticas por clase:")
    for idx, name in enumerate(yaml_data['names']):
        print(f"Clase {idx} ({name}):")
        print(f"- Precisión: {modelo.metrics.box.p[idx] if hasattr(modelo, 'metrics') else 'No disponible'}")
        print(f"- Recall: {modelo.metrics.box.r[idx] if hasattr(modelo, 'metrics') else 'No disponible'}")
else:
    # Si no encontramos el yaml, mostrar solo los nombres del modelo
    print("\nNombres de las clases (del modelo):")
    for idx, name in modelo.names.items():
        print(f"Clase {idx}: {name}")