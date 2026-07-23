# RoadSignNet-Sal

A deep learning project for **road sign detection and classification** using Saliency-Aware Learning (SAL) and advanced neural network architectures. This project implements state-of-the-art object detection with 43 traffic sign classes using hybrid neural network backbones, knowledge distillation, and hard negative mining.

## 📋 Overview

RoadSignNet-Sal is designed to accurately detect and classify road signs in real-world driving scenarios. The project features:

- **Multi-backbone architecture support** (YOLOv8, EfficientNet, DenseNet, ViT)
- **Knowledge distillation** for model compression and efficient inference
- **Hard negative mining** to improve model robustness
- **Hybrid neural network architectures** combining multiple backbones
- **43-class traffic sign classification** 
- **Saliency-aware learning** for improved localization
- **Mixed precision training** for faster computation
- **ONNX & TorchScript export** for deployment

## 🎯 Features

### Model Architectures
- **Teacher Models**: Large-capacity models (EfficientNet-B3 + ViT, DenseNet + EfficientNet + ViT hybrids)
- **Student Models**: Lightweight models optimized for real-time inference
- **Custom KAN Classification Head**: Kolmogorov-Arnold Network based classifier

### Training Strategies
- **Knowledge Distillation**: Transfer knowledge from large teacher models to compact students
- **Hard Negative Mining**: Identify and include challenging false positives during training
- **Balanced Loss**: Weighted class balancing for imbalanced datasets
- **Early Stopping & Checkpoint Management**: Automatic model saving and recovery

### Data Processing
- **Multi-dataset merging** from Roboflow and custom sources
- **43-class mapping** from 56-class datasets
- **Augmentation**: Mosaic, mixup, color jitter, random erasing
- **Multi-resolution support**: 224x224, 256x256, 320x320, 512x512

## 📁 Project Structure

```
RoadSignNet-Sal/
├── README.md                           # This file
├── check_params.py                     # Parameter checking utility
├── class_weights.py                    # Class weighting for imbalanced data
├── config/                             # Training configuration files
│   ├── config_v6_retrain_yolov8n.yaml             # YOLOv8n retrain config
│   ├── config_v6_retrain_b3_vitsmall.yaml        # EfficientNet-B3 + ViT config
│   ├── config_v6_retrain_densenet_hybrid.yaml    # DenseNet hybrid config
│   ├── config_v6_finetune.yaml                   # Fine-tuning config
│   ├── config_teacher_mapped43_distill.yaml      # Distillation config
│   ├── config_v6_hardneg_finetune.yaml           # Hard negative mining config
│   └── config_v6_realtime_teacher_img256.yaml    # Real-time model config
├── data/                               # Dataset storage
│   └── README.roboflow.txt             # Roboflow dataset documentation
├── downloads/                          # Downloaded datasets
├── outputs/                            # Training outputs, checkpoints, exports
└── [additional Python modules]         # Model definitions, training loops, etc.

```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.13+
- CUDA 11.7+ (for GPU training)
- Required packages: `pip install -r requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/maisha0055/RoadSignNet-Sal.git
cd RoadSignNet-Sal

# Install dependencies
pip install -r requirements.txt
```

### Training

#### 1. Basic Training with Default Config
```bash
python train.py --config config/config_v6_finetune.yaml
```

#### 2. Training with YOLOv8n Backbone
```bash
python train.py --config config/config_v6_retrain_yolov8n.yaml
```

#### 3. Training with Hybrid Backbone (Recommended)
```bash
python train.py --config config/config_v6_retrain_densenet_hybrid.yaml
```

#### 4. Knowledge Distillation Training
```bash
python train.py --config config/config_teacher_mapped43_distill.yaml
```

#### 5. Hard Negative Mining Fine-tuning
```bash
python train.py --config config/config_v6_hardneg_finetune.yaml
```

### Inference

```bash
python inference.py \
    --model outputs/v6_finetune/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --conf-threshold 0.5
```

### Model Export

```bash
# Export to ONNX
python export.py --model checkpoints/best_model.pth --format onnx

# Export to TorchScript
python export.py --model checkpoints/best_model.pth --format torchscript
```

## 📊 Dataset

The project uses a 43-class traffic sign dataset compiled from:
- **Roboflow Traffic Sign Detection dataset** (v1, 2023)
- Custom datasets with augmentation
- Total: ~15,000+ annotated images

### Class Labels (43 classes)
bicycle, bus_stop, children, crosswalk, do_not_stop, do_not_turn_left, do_not_turn_right, do_not_u_turn, enter_left_lane, give_way, green_light, left_lane_enter, left_turn, narrow_road, no_entry, no_overtaking, no_parking, no_stop, no_waiting, parking, railway_crossing, red_light, refueling, right_turn, road_main, road_work, school_nearby, speed_bump, speed_limit_20, speed_limit_30, speed_limit_40, speed_limit_50, speed_limit_60, speed_limit_70, speed_limit_80, speed_limit_100, speed_limit_120, stop, t_intersection_l, t_intersection_r, traffic_sign, truck, yellow_light

### Data Format
- **YOLO Format**: Images + corresponding `.txt` label files with normalized bounding box coordinates
- **Augmentation**: Rotation (±10°), shear (±10°), scale, color jitter

## 🔧 Configuration

Each training configuration includes:

```yaml
experiment:
  name: experiment_name
  description: Brief description
  
model:
  num_classes: 43
  width_multiplier: 1.0-1.5  # Model scaling
  backbone: "efficientnet_b0+vit_tiny_patch16_224"  # Backbone selection
  use_kan_cls: true  # Use KAN classifier
  
training:
  epochs: 30-50
  batch_size: 4-8
  img_size: 224-512
  optimizer: adamw
  lr: 1e-4 to 3e-4
  scheduler: cosine_annealing_warmup
  amp: true  # Mixed precision
  
loss:
  lambda_cls: 1.0-1.5  # Classification loss weight
  lambda_box: 1.5-1.8  # Bounding box loss weight
  lambda_obj: 1.5-2.3  # Objectness loss weight
  
data:
  train_img_dir: downloads/merged_all_mapped43/train/images
  val_img_dir: downloads/merged_all_mapped43/valid/images
  test_img_dir: downloads/merged_all_mapped43/test/images
```

## 💡 Training Strategies

### 1. **Knowledge Distillation**
- Large teacher model trains on full dataset
- Smaller student model learns from teacher's predictions
- Reduces model size while maintaining accuracy
- Config: `config_teacher_mapped43_distill.yaml`

### 2. **Hard Negative Mining**
- Identify misclassified regions during inference
- Retrain with mined false positives
- Improves precision and robustness
- Config: `config_v6_hardneg_finetune.yaml`

### 3. **Fine-tuning**
- Start from pre-trained weights
- Low learning rate training
- 190 epochs at 2e-5 LR
- Config: `config_v6_finetune.yaml`

### 4. **Multi-resolution Training**
- Progressive resolution: 224 → 256 → 320 → 512
- Improves generalization
- Different configs for each resolution

## 📈 Performance

### Model Variants

| Model | Backbone | Params | Image Size | Training Time | Use Case |
|-------|----------|--------|-----------|---------------|----------|
| YOLOv8n Retrain | YOLOv8n | ~3M | 224 | ~2h | Fast inference |
| EfficientNet-B0 + ViT | Hybrid | ~8M | 256 | ~3h | Balanced |
| EfficientNet-B3 + ViT | Hybrid | ~15M | 224 | ~6h | High accuracy |
| DenseNet + EfficientNet + ViT | Hybrid | ~25M | 224 | ~8h | Best accuracy |
| Student (Distilled) | EfficientNet-B0 | ~5M | 512 | ~4h | Optimized inference |

## 🔄 Workflow

```
Raw Dataset
    ↓
Dataset Merging & Mapping (56 → 43 classes)
    ↓
Train/Val/Test Split
    ↓
Teacher Model Training (Large, High Accuracy)
    ↓
Hard Negative Mining (Identify FP regions)
    ↓
Fine-tuning with Hard Negatives
    ↓
Knowledge Distillation (Compress model)
    ↓
Student Model Export (ONNX/TorchScript)
    ↓
Deployment
```

## 📦 Outputs

Training generates:
- `outputs/{experiment_name}/checkpoints/` - Model checkpoints
- `outputs/{experiment_name}/logs/` - Training logs (TensorBoard compatible)
- `outputs/{experiment_name}/exports/` - ONNX & TorchScript models
- `class_weights.py` - Computed class weights for balancing

## 🛠️ Utilities

### check_params.py
Calculates parameter count for different model width multipliers:
```bash
python check_params.py
# Output: Width 1.00: X.XXM params
```

### class_weights.py
Pre-computed class weights to handle imbalanced dataset:
```python
from class_weights import CLASS_WEIGHTS
# Use in loss function
```

## 🖼️ Visualization

The project includes debug visualizations:
- `debug_sample_1.png` - Sample inference results
- `debug_sample_2.png` - Multi-object detection examples
- `debug_target_generation.png` - Target generation pipeline
- `debug_multi_object_targets.png` - Multiple objects in single frame

## 📚 Requirements

Core dependencies:
- torch>=1.13.0
- torchvision>=0.14.0
- opencv-python>=4.5.0
- pyyaml>=6.0
- tensorboard>=2.10
- numpy>=1.21.0
- PIL/Pillow>=9.0.0

See `requirements.txt` for complete list.

## 🔍 Debugging

### Check Model Parameters
```bash
python check_params.py
```

### View Class Distribution
```python
from class_weights import CLASS_WEIGHTS
print(f"Num classes: {len(CLASS_WEIGHTS)}")
```

### TensorBoard Monitoring
```bash
tensorboard --logdir outputs/v6_finetune/logs
# Access at http://localhost:6006
```

## 🚨 Known Issues & Tips

1. **GPU Memory**: Reduce `batch_size` if OOM errors occur
2. **Slow Training**: Enable `mixed_precision: true` for faster convergence
3. **Data Loading**: Ensure paths in config match your directory structure
4. **Class Imbalance**: Use pre-computed `CLASS_WEIGHTS` in loss function
5. **Hard Negative Mining**: Requires first training a model to mine negatives

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Submit a pull request

## 📄 License

This project is open source. Check LICENSE file for details.

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

**Last Updated**: July 2026  
**Project Status**: Active Development
