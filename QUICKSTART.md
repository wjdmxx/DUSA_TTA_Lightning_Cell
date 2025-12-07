# Quick Start Guide

## Setup

```bash
# 1. Clone repository
cd DUSA_TTA_Lightning

# 2. Create environment
conda create -n dusa_tta python=3.10
conda activate dusa_tta

# 3. Install dependencies
pip install -r requirements.txt
```

## Configuration

### 1. Update Data Path

Edit `configs/config.yaml`:
```yaml
data_root: "/path/to/ImageNet-C"
```

### 2. Update Model Checkpoint

Edit `configs/model/combined.yaml`:
```yaml
generative:
  sit_checkpoint: "/path/to/REPA-SiT-XL-2-256.pt"
```

### 3. Configure W&B (Optional)

Edit `configs/config.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_entity: "your_username"  # Your W&B username/team
```

## Running TTA

### Basic Usage

```bash
# Run with default config
python scripts/run_tta.py

# Run with experiment preset
python scripts/run_tta.py experiment=convnext_imagenet_c
```

### Override Parameters

```bash
# Change batch size
python scripts/run_tta.py data.batch_size=32

# Use multiple GPUs
python scripts/run_tta.py trainer.devices=4

# Test only specific corruptions
python scripts/run_tta.py data.corruptions='["gaussian_noise","glass_blur"]'

# Continual TTA (don't reset between tasks)
python scripts/run_tta.py tta.continual=true

# Adjust learning rate
python scripts/run_tta.py optimizer.learning_rate=5e-6
```

### Advanced: Modify Model Architecture

```bash
# Use different SiT model
python scripts/run_tta.py \
    model.generative.sit_model_name="SiT-B/2"

# Change top-k and random budget
python scripts/run_tta.py \
    model.generative.topk=6 \
    model.generative.rand_budget=4

# Use different timm classifier
python scripts/run_tta.py \
    model.discriminative.model_name="efficientnet_b0"
```

## Expected Output

```
================================================================================
DUSA Test-Time Adaptation
================================================================================
Task 1/95: gaussian_noise_severity1
================================================================================
Epoch 0: 100%|███████| 3125/3125 [05:23<00:00, 9.65it/s, loss=0.234, top1=76.8]
Completed task: gaussian_noise_severity1
================================================================================
Task 2/95: gaussian_noise_severity2
...
```

## Output Structure

```
outputs/
└── dusa_tta_convnext/
    └── 2025-12-07/
        └── 15-30-00/
            ├── .hydra/
            │   └── config.yaml  # Full config used
            ├── tensorboard_logs/
            │   └── version_0/
            └── run_tta.log
```

## Distributed Training

### DDP (Multi-GPU, Single Node)

```bash
python scripts/run_tta.py trainer.devices=4 trainer.strategy=ddp
```

### FSDP (For Large Models)

```bash
python scripts/run_tta.py \
    trainer.devices=4 \
    trainer.strategy=fsdp \
    trainer.precision="bf16-mixed"
```

## Troubleshooting

### Out of Memory

```bash
# Reduce batch size
python scripts/run_tta.py data.batch_size=8

# Use gradient accumulation
python scripts/run_tta.py trainer.accumulate_grad_batches=2
```

### Slow Training

```bash
# Reduce precision
python scripts/run_tta.py trainer.precision="16-mixed"

# Increase workers
python scripts/run_tta.py data.num_workers=8
```

### CUDA Out of Memory on Auxiliary Model

The generative model can be placed on a different GPU if needed. Edit `src/models/generative/repa_sit.py` to specify device.

## Next Steps

- Check `configs/experiment/` for more preset configurations
- Modify `src/models/` to add new models
- Extend `src/data/` for new datasets
- Customize `src/tta/module.py` for different TTA strategies
