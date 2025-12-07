# DUSA TTA with PyTorch Lightning

A clean, modular implementation of DUSA (Diffusion-based Unsupervised Test-Time Adaptation) using PyTorch Lightning and Hydra.

## Overview

This project implements TTA for image classification by using a generative flow model (REPA SiT) to guide the adaptation of a discriminative classifier (e.g., ConvNeXt-L from timm). No training phase—pure test-time adaptation on corrupted datasets.

## Key Features

- ✅ **Lightning-based**: Clean trainer with automatic DDP/FSDP support
- ✅ **Hydra configs**: Structured YAML configs for experiments, no command-line chaos
- ✅ **W&B logging**: Automatic tracking of metrics, losses, and artifacts
- ✅ **Modular design**: Easy to swap discriminative/generative models
- ✅ **Multi-GPU ready**: Single GPU or distributed with minimal config changes
- ✅ **timm classifiers**: Lightweight, no MMOpenLab dependencies

## Project Structure

```
DUSA_TTA_Lightning/
├── configs/                  # Hydra configuration files
│   ├── config.yaml          # Main config with defaults
│   ├── experiment/          # Experiment-specific overrides
│   │   └── convnext_imagenet_c.yaml
│   ├── model/               # Model configs
│   │   ├── discriminative/
│   │   │   └── convnext_large.yaml
│   │   └── generative/
│   │       └── repa_sit.yaml
│   ├── data/                # Data configs
│   │   └── imagenet_c.yaml
│   └── trainer/             # Trainer configs
│       └── default.yaml
├── src/
│   ├── data/                # Data modules and preprocessing
│   │   ├── __init__.py
│   │   ├── imagenet_c.py   # ImageNet-C dataset
│   │   └── transforms.py    # Task/auxiliary transforms
│   ├── models/              # Model implementations
│   │   ├── __init__.py
│   │   ├── discriminative.py  # timm-based classifiers
│   │   ├── generative/        # Flow models
│   │   │   ├── __init__.py
│   │   │   ├── repa_sit.py   # REPA SiT implementation
│   │   │   ├── scheduler.py  # Flow scheduler
│   │   │   └── vae.py        # VAE encoder
│   │   └── combined.py       # Unified model wrapper
│   ├── tta/                 # TTA training logic
│   │   ├── __init__.py
│   │   ├── module.py        # LightningModule for TTA
│   │   └── losses.py        # REPA loss functions
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── logging.py       # W&B integration
│       └── metrics.py       # Accuracy, AUC, etc.
├── scripts/
│   └── run_tta.py           # Main entry point
├── requirements.txt
└── .gitignore
```

## Installation

```bash
# Create environment
conda create -n dusa_tta python=3.10
conda activate dusa_tta

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run TTA on ImageNet-C with ConvNeXt-L
python scripts/run_tta.py experiment=convnext_imagenet_c

# Multi-GPU (DDP)
python scripts/run_tta.py experiment=convnext_imagenet_c trainer.devices=4

# Override specific parameters
python scripts/run_tta.py \
    experiment=convnext_imagenet_c \
    model.generative.topk=6 \
    model.generative.rand_budget=4 \
    data.batch_size=16
```

## Configuration

Configs are managed via Hydra. Key sections:

- **experiment**: Top-level presets (e.g., `convnext_imagenet_c.yaml`)
- **model.discriminative**: Classifier config (backbone, pretrained weights)
- **model.generative**: Flow model config (SiT, VAE, loss weights)
- **data**: Dataset paths, corruptions, batch size
- **trainer**: Lightning trainer settings (devices, precision, strategy)

## Design Principles

1. **No MMOpenLab**: Uses timm for classifiers, Hugging Face Diffusers for VAE
2. **Clean separation**: Task preprocessing (224px, ImageNet norm) vs. auxiliary preprocessing (256px, [-1,1])
3. **Flexible distribution**: Lightning handles DDP/FSDP/DP automatically
4. **Hydra composition**: Override any config via CLI or experiment files

## Key Differences from Original DUSA

| Feature | Original DUSA | This Implementation |
|---------|--------------|---------------------|
| Framework | MMOpenLab (mmpretrain, mmengine) | PyTorch Lightning + Hydra |
| Discriminative | MMPretrain models | timm models |
| Generative | Custom registry | Standalone modules |
| Config | Python dicts + CLI args | Hydra YAML configs |
| Logging | Custom + MMEngine | W&B + TensorBoard |
| Distributed | Manual DDP setup | Lightning auto-strategy |

## Core Components Explained

### 1. Discriminative Model (`src/models/discriminative.py`)
- Wraps timm models (e.g., ConvNeXt-L)
- Provides feature extraction for optional alignment
- Configurable training mode (all params vs. norm-only)

### 2. Generative Model (`src/models/generative/repa_sit.py`)
- **SiT**: Scalable Interpolant Transformer for flow matching
- **VAE**: Stable Diffusion VAE for latent encoding
- **Flow Scheduler**: Linear interpolation scheduler
- **REPA Loss**: `norm_l2_loss + cosine_loss`

### 3. Combined Model (`src/models/combined.py`)
- Unifies discriminative + generative
- Handles different preprocessing pipelines
- Implements forward modes (logits, normed_logits_with_logits)

### 4. TTA Module (`src/tta/module.py`)
- Lightning module for test-time adaptation
- No training phase—only adaptation
- Supports continual vs. reset-per-task
- Manual optimization for fine control

### 5. Data Module (`src/data/`)
- ImageNet-C dataset with 19 corruptions × 5 severities
- Dual transforms: task (224px, ImageNet norm) + raw (256px, [-1,1])
- Custom collate for batched + list data

## REPA Loss Mechanism

The REPA loss bridges generative and discriminative models:

```
1. Top-K + Random Sampling:
   - Select top-k classes by normalized logits
   - Sample random classes from remaining by temperature-scaled original logits

2. VAE Encoding:
   x0 = VAE(image) * 0.18215

3. Flow Matching:
   x_t = (1-t) * x0 + t * z
   v_t = z - x0  (target velocity)

4. SiT Prediction:
   For each selected class c_i:
     v_pred_i = SiT(x_t, t, c_i)

5. Weighted Loss:
   v_weighted = Σ softmax(logits)[i] * v_pred_i
   loss = norm_l2(v_weighted, v_t) + cosine(v_weighted, v_t)

6. Backprop:
   Update SiT + task model norms via Adam
```

### Why This Works for TTA

- **Generative prior**: SiT learns class-conditioned distributions
- **Unsupervised signal**: No labels needed—uses model's own predictions
- **Entropy regularization**: Weighted loss encourages confident, correct predictions
- **Norm adaptation**: Task model normalization layers adapt to test distribution

## Roadmap

- [x] Core REPA SiT implementation
- [x] ConvNeXt-L discriminative model
- [x] ImageNet-C data module
- [x] Lightning TTA module
- [x] Hydra configs + W&B logging
- [ ] Additional flow models (Linear Flow, DiT)
- [ ] Segmentation tasks (future)
- [ ] Multi-node distributed training
- [ ] Checkpointing and resumption

## Citation

If you use this code, please cite the original DUSA paper:
```bibtex
@inproceedings{dusa2024,
  title={DUSA: Diffusion-based Unsupervised Test-Time Adaptation},
  author={...},
  booktitle={...},
  year={2024}
}
```

## License

MIT
