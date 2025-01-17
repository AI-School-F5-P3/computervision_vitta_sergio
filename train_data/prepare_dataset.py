import yaml

# Define el orden específico de las clases
clases = {
    'adidas': 0,
    'apple': 1,
    'nike': 2,
    'reebok': 3,
    'samsung': 4,
    'sony': 5
}

# Crear el archivo data.yaml
config = {
    'path': '../datasets/logos',  # Ruta al dataset
    'train': 'train/images',      # Ruta relativa a las imágenes de entrenamiento
    'val': 'valid/images',        # Ruta relativa a las imágenes de validación
    'test': 'test/images',        # Ruta relativa a las imágenes de prueba
    'names': clases,              # Diccionario de nombres de clases
    'nc': len(clases)             # Número de clases
}

# Guardar el archivo yaml
with open('dataset/logos/data.yaml', 'w') as f:
    yaml.dump(config, f, sort_keys=False)

# Verificar el contenido del archivo
print("Configuración del dataset:")
print("\nClases definidas:")
for nombre, id in clases.items():
    print(f"- {nombre}: {id}")

print("\nArchivo data.yaml creado con éxito.") 