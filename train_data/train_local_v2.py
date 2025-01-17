from ultralytics import YOLO
import yaml
import os
import torch
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime, timedelta

def check_gpu():
    print("\n=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print("===========================\n")

class TrainCallback:
    def __init__(self, epochs):
        self.epochs = epochs
        self.start_time = time.time()
        self.best_map = 0
        
    def on_train_epoch_start(self, trainer):
        self.epoch_start = time.time()
        print(f"\nStarting epoch {trainer.epoch + 1}/{self.epochs}")
        
    def on_train_epoch_end(self, trainer):
        try:
            elapsed_time = time.time() - self.start_time
            time_per_epoch = elapsed_time / (trainer.epoch + 1)
            remaining_epochs = self.epochs - (trainer.epoch + 1)
            remaining_time = remaining_epochs * time_per_epoch
            
            remaining_time = str(timedelta(seconds=int(remaining_time)))
            
            if hasattr(trainer, 'metrics'):
                current_map = trainer.metrics.get('map50-95', 0)
                if current_map > self.best_map:
                    self.best_map = current_map
                    print(f"New best mAP: {self.best_map:.4f}!")
            
            print(f"Epoch {trainer.epoch + 1} completed")
            print(f"Estimated time remaining: {remaining_time}")
            print(f"Best mAP so far: {self.best_map:.4f}")
        except Exception as e:
            print(f"Warning: Error in callback: {e}")

def train_model(epochs=150):
    print("Using local dataset...")
    # Check GPU and force its use
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU with CUDA to run")
    
    device = 0  # Force GPU usage
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    
    # Build path to local dataset
    base_path = Path(__file__).parent
    dataset_path = base_path / 'datasetv2'
    yaml_path = dataset_path / 'data.yaml'
    
    print(f"Dataset located at: {dataset_path}")
    print(f"Configuration file: {yaml_path}")
    
    # Verify dataset exists and check classes
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml file not found at {yaml_path}")
    
    # Verify class configuration
    with open(yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    expected_classes = ['adidas', 'apple', 'nike', 'reebok', 'samsung', 'sony']
    
    if 'names' not in data_config:
        raise ValueError("Classes not defined in data.yaml")
    
    if data_config['names'] != expected_classes:
        raise ValueError(
            "Incorrect class configuration in data.yaml. "
            f"Expected: {expected_classes}, "
            f"Got: {data_config['names']}"
        )
    
    print("Class configuration verified:")
    for idx, name in enumerate(expected_classes):
        print(f"  {idx}: {name}")
    
    # Verify dataset structure
    train_path = dataset_path / 'train' / 'images'
    valid_path = dataset_path / 'valid' / 'images'
    if not train_path.exists():
        raise FileNotFoundError(f"Training images not found at {train_path}")
    if not valid_path.exists():
        raise FileNotFoundError(f"Validation images not found at {valid_path}")
    
    # Verify CUDA is working
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.init()
        except Exception as e:
            print(f"Warning: Error initializing CUDA: {e}")
    
    # Initialize model and callback
    model = YOLO('yolov8n.pt')
    callback = TrainCallback(epochs)
    
    try:
        # Train model with parameters
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=32,
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
            device=device,
            augment=True,
            callbacks=[callback],
            amp=True
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        metrics = model.val()
        
        # Save results
        print("\nSaving results...")
        current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory for saving results if it doesn't exist
        results_directory = Path('trained_models')
        results_directory.mkdir(exist_ok=True)
        
        # Save model in both formats
        version = 1
        while True:
            model_name = f'logo_model_v{version}_{current_date}'
            save_path_pt = results_directory / f'{model_name}.pt'
            if not save_path_pt.exists():
                break
            version += 1
            
        # Define ONNX path using the same model_name
        save_path_onnx = results_directory / f'{model_name}.onnx'
        
        # Save .pt model
        model.save(str(save_path_pt))
        print(f"\nModel saved at: {save_path_pt}")
        
        # Export to ONNX
        print("\nExporting model to ONNX...")
        try:
            model.export(
                format='onnx',
                imgsz=640,
                opset=12,
                file=str(save_path_onnx)  # Added: specify output file
            )
            print(f"Model exported successfully to: {save_path_onnx}")
        except Exception as e:
            print(f"Error exporting model to ONNX: {e}")
            print("Trained model is still available in .pt format")
        
        # Save metrics in the same directory
        metrics_path = results_directory / f'metrics_{model_name}.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"mAP: {metrics.box.map:.4f}\n")
            f.write(f"Precision: {metrics.box.p:.4f}\n")
            f.write(f"Recall: {metrics.box.r:.4f}\n")
        
        # Save training configuration
        config_path = results_directory / f'config_{model_name}.txt'
        with open(config_path, 'w') as f:
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch size: 32\n")
            f.write(f"Image size: 640\n")
            f.write(f"Optimizer: Adam\n")
            f.write(f"Learning rate: 0.001\n")
            f.write(f"Dataset: {dataset_path}\n")
        
        print("\nTraining completed successfully!")
        print(f"Final metrics:")
        print(f"mAP: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.p:.4f}")
        print(f"Recall: {metrics.box.r:.4f}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == "__main__":
    check_gpu()
    print("=== Starting logo detection training program ===")
    
    train_model(epochs=150)