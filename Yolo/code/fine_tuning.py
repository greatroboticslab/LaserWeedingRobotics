#!/usr/bin/env python3
"""
YOLO11 Fine-tuning Script for Weed Detection
Optimized for maximum accuracy based on 2024-2025 best practices
"""

import argparse
import os
import torch
from ultralytics import YOLO
import yaml
from pathlib import Path

def setup_training_config():
    """
    Configure optimal hyperparameters based on latest research
    """
    return {
        # Core training parameters
        'epochs': 100,  # Optimized for fine-tuning (use 200+ for from scratch)
        'patience': 30,  # Early stopping patience (reduced for fine-tuning)
        'batch': -1,  # Auto batch size (60% GPU memory)
        'imgsz': 640,  # Standard input size
        
        # Learning rate and optimization (optimized for fine-tuning)
        'lr0': 0.005,  # Lower initial learning rate for fine-tuning
        'lrf': 0.01,  # Final learning rate factor
        'momentum': 0.937,  # SGD momentum
        'weight_decay': 0.0005,  # Weight decay for regularization
        'warmup_epochs': 3,  # Warmup epochs
        'warmup_momentum': 0.8,  # Warmup initial momentum
        'warmup_bias_lr': 0.1,  # Warmup initial bias lr
        
        # Optimizer settings
        'optimizer': 'SGD',  # SGD performs better for YOLO
        'cos_lr': True,  # Cosine learning rate scheduler
        
        # Data augmentation
        'hsv_h': 0.015,  # Hue augmentation
        'hsv_s': 0.7,  # Saturation augmentation  
        'hsv_v': 0.4,  # Value augmentation
        'degrees': 0.0,  # Rotation degrees
        'translate': 0.1,  # Translation fraction
        'scale': 0.5,  # Scaling factor
        'shear': 0.0,  # Shear degrees
        'perspective': 0.0,  # Perspective transformation
        'flipud': 0.0,  # Vertical flip probability
        'fliplr': 0.5,  # Horizontal flip probability
        'mosaic': 1.0,  # Mosaic augmentation probability
        'mixup': 0.0,  # Mixup augmentation probability
        'copy_paste': 0.0,  # Copy-paste augmentation probability
        'close_mosaic': 10,  # Disable mosaic for last N epochs
        
        # Model configuration
        'pretrained': True,  # Use pretrained weights
        'freeze': None,  # Don't freeze layers for fine-tuning
        
        # Loss function weights
        'cls': 0.5,  # Classification loss gain
        'box': 7.5,  # Box regression loss gain
        'dfl': 1.5,  # Distribution focal loss gain
        
        # Validation and saving
        'val': True,  # Validate during training
        'save': True,  # Save checkpoints
        'save_period': 10,  # Save checkpoint every N epochs
        'cache': True,  # Cache images for faster loading
        'device': '',  # Auto-select device
        'workers': 8,  # Number of dataloader workers
        'project': 'runs/train',  # Project directory
        'name': 'weed_detection_finetuned',  # Experiment name
        'exist_ok': False,  # Overwrite existing project/name
        'plots': True,  # Generate training plots
        'deterministic': True,  # Deterministic training
        'single_cls': False,  # Single class training
        'rect': False,  # Rectangular training
        'resume': False,  # Resume training
        'amp': True,  # Automatic mixed precision
        'fraction': 1.0,  # Dataset fraction to train on
        'profile': False,  # Profile ONNX and TensorRT speeds
        'overlap_mask': True,  # Use overlap mask
        'mask_ratio': 4,  # Mask downsample ratio
        'dropout': 0.0,  # Dropout rate
        'seed': 0,  # Random seed
        'verbose': True,  # Verbose output
    }

def validate_dataset(data_path):
    """
    Validate dataset structure and configuration
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")
    
    with open(data_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key in data.yaml: {key}")
    
    print(f"✅ Dataset validated: {config['nc']} classes, {len(config['names'])} names")
    return config

def setup_model(model_path, pretrained=True):
    """
    Initialize YOLO model with optimal settings
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = YOLO(model_path)
    print(f"✅ Model loaded: {model_path}")
    print(f"📊 Model info: {model.info()}")
    
    return model

def train_model(model, data_path, config):
    """
    Execute optimized training with monitoring
    """
    print("\n🚀 Starting fine-tuning with optimized hyperparameters...")
    print(f"📈 Training for {config['epochs']} epochs with early stopping (patience={config['patience']})")
    
    # Train the model
    results = model.train(
        data=data_path,
        **config
    )
    
    print("\n✅ Training completed!")
    print(f"📊 Best results saved to: {results.save_dir}")
    
    return results

def evaluate_model(model, data_path):
    """
    Comprehensive model evaluation
    """
    print("\n🧪 Evaluating model performance...")
    
    # Run validation on test set
    metrics = model.val(data=data_path, split='test')
    
    print(f"📊 Test Results:")
    print(f"   mAP@0.5: {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLO11 for weed detection')
    parser.add_argument('--model', default='yolo11nweed.pt', help='Path to your trained weed model')
    parser.add_argument('--data', default='yoloweeddataset/data.yaml', help='Path to dataset config')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', default=-1, help='Batch size (-1 for auto)')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--lr0', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--device', default='', help='Device (cuda/cpu)')
    parser.add_argument('--project', default='runs/train', help='Project directory')
    parser.add_argument('--name', default='weed_detection_finetuned', help='Experiment name')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    
    args = parser.parse_args()
    
    # Setup training configuration
    config = setup_training_config()
    
    # Override with command line arguments
    config.update({
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'resume': args.resume,
        'pretrained': args.pretrained,
    })
    
    print("🌿 YOLO11 Weed Detection Fine-tuning")
    print("=" * 50)
    print(f"📁 Using dataset: {args.data}")
    print(f"🤖 Using model: {args.model}")
    
    try:
        # Validate dataset
        dataset_config = validate_dataset(args.data)
        
        # Setup model
        model = setup_model(args.model, args.pretrained)
        
        # Check hardware
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 Using device: {device}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Train model
        results = train_model(model, args.data, config)
        
        # Load best model for evaluation
        best_model = YOLO(results.save_dir / 'weights' / 'best.pt')
        
        # Evaluate model
        metrics = evaluate_model(best_model, args.data)
        
        print(f"\n🎯 Fine-tuning completed successfully!")
        print(f"📁 Results saved to: {results.save_dir}")
        print(f"🏆 Best model: {results.save_dir / 'weights' / 'best.pt'}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()