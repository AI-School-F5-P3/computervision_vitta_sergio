import cv2
from ultralytics import YOLO
import numpy as np

def test_model(model_path, image_path):
    # Cargar el modelo entrenado
    model = YOLO(model_path)
    
    # Cargar imagen
    image = cv2.imread(image_path)
    
    # Realizar predicción
    results = model(image)[0]
    
    # Dibujar resultados
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        
        if score > 0.5:  # Umbral de confianza
            # Dibujar bounding box
            cv2.rectangle(image, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        (0, 255, 0), 2)
            
            # Añadir etiqueta con nombre y confianza
            label = f"{model.names[int(class_id)]}: {score:.2f}"
            cv2.putText(image, label, 
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 2)
    
    # Mostrar imagen
    cv2.imshow('Test Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model(
        model_path='runs/detect/logo_detector_model/weights/best.pt',
        image_path='ruta/a/tu/imagen/de/prueba.jpg'
    )