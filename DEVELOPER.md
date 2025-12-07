# Developer Guide

## Adding New Models

### Adding a New Discriminative Model

1. **If using timm**: Just update config
```yaml
# configs/model/discriminative/resnet50.yaml
model_name: "resnet50"
pretrained: true
checkpoint_path: null
num_classes: 1000
```

2. **If custom model**: Extend `TimmClassifier`
```python
# src/models/discriminative.py
class CustomClassifier(nn.Module):
    def forward(self, x, return_features=False):
        # Implement forward pass
        logits = ...
        features = ... if return_features else None
        return logits, features
    
    def get_feature_dim(self):
        return self.feature_dim
    
    def set_train_mode(self, update_all=False, update_norm_only=True):
        # Configure trainable parameters
        pass
```

### Adding a New Generative Model

Example: Implementing Linear Flow Model

```python
# src/models/generative/linear_flow.py
class LinearFlow(nn.Module):
    """Linear flow matching model."""
    
    def __init__(self, ...):
        super().__init__()
        # Build model
    
    def forward(self, images, normed_logits, ori_logits, batch_infos):
        # 1. Preprocess images
        # 2. Encode to latent
        # 3. Sample classes
        # 4. Flow matching forward
        # 5. Compute loss
        loss, aux_metrics = ...
        return loss, aux_metrics
```

Update `src/models/generative/__init__.py`:
```python
from .linear_flow import LinearFlow, create_linear_flow
```

Create config:
```yaml
# configs/model/generative/linear_flow.yaml
model_type: "linear_flow"
...
```

## Adding New Datasets

### Example: CIFAR-10-C

```python
# src/data/cifar10_c.py
class CIFAR10CDataset(Dataset):
    CORRUPTIONS = ["...", ]
    SEVERITIES = [1, 2, 3, 4, 5]
    
    def __init__(self, root, corruption, severity, transform, raw_transform):
        # Build image list
        pass
    
    def __getitem__(self, idx):
        return {
            "task_image": ...,
            "raw_image": ...,
            "label": ...,
        }

def create_cifar10_c_tasks(...):
    # Create list of (task_name, dataset) tuples
    pass
```

Create config:
```yaml
# configs/data/cifar10_c.yaml
root: "path/to/CIFAR-10-C"
corruptions: null
severities: null
batch_size: 128
num_workers: 4
task_input_size: 32  # CIFAR size
task_mean: [0.4914, 0.4822, 0.4465]
task_std: [0.2023, 0.1994, 0.2010]
```

## Adding New TTA Strategies

### Example: TENT (Update Norm Layers Only)

Already supported! Just configure:
```yaml
tta:
  update_auxiliary: false  # No generative model
  update_task_norm_only: true
  forward_mode: "logits"  # No auxiliary forward
```

### Example: Custom TTA Strategy

Extend `DUSATTAModule`:
```python
# src/tta/custom_tta.py
class CustomTTAModule(DUSATTAModule):
    def training_step(self, batch, batch_idx):
        # Custom TTA logic
        logits, loss, metrics = self(batch, batch_idx)
        
        # Custom loss computation
        custom_loss = self.compute_custom_loss(logits, batch)
        
        # Combine losses
        total_loss = loss + custom_loss
        
        # Optimization
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        
        return total_loss
```

## Modifying Loss Functions

### REPA Loss Variants

Edit `src/models/generative/repa_sit.py`:
```python
def _compute_repa_loss(self, pred, target):
    # Original: norm_l2 + cosine
    norm_l2 = ...
    cos_loss = ...
    
    # Add new terms
    mse_loss = F.mse_loss(pred, target)
    
    # Weighted combination
    return norm_l2 + cos_loss + 0.1 * mse_loss
```

Make it configurable:
```yaml
# configs/model/generative/repa_sit.yaml
loss_weights:
  norm_l2: 1.0
  cosine: 1.0
  mse: 0.1
```

## Debugging Tips

### 1. Fast Dev Run
```bash
python scripts/run_tta.py trainer.fast_dev_run=true
```
Runs 1 batch of train/val/test.

### 2. Single Task
```bash
python scripts/run_tta.py \
    data.corruptions='["gaussian_noise"]' \
    data.severities='[1]'
```

### 3. Profiling
```python
# Add to run_tta.py
from pytorch_lightning.profilers import SimpleProfiler

profiler = SimpleProfiler(dirpath=".", filename="profile")
trainer = pl.Trainer(..., profiler=profiler)
```

### 4. Check Gradients
```python
# In DUSATTAModule.training_step
for name, param in self.model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

## Best Practices

### 1. Config Organization
- Use experiments for full presets
- Use model/data/trainer for components
- Override via CLI for quick tweaks

### 2. Logging
- Log to W&B for remote monitoring
- Use TensorBoard for local analysis
- Add custom metrics in `training_step`

### 3. Reproducibility
- Set seed: `pl.seed_everything(42)`
- Use deterministic: `trainer.deterministic=true` (slower)
- Pin workers: `data.pin_memory=true`

### 4. Memory Management
- Use gradient accumulation for large batch sizes
- Enable gradient checkpointing for large models
- Profile memory with `torch.cuda.memory_summary()`

## Testing

### Unit Tests (TODO)
```python
# tests/test_models.py
def test_combined_model():
    model = create_combined_model(...)
    logits, features, loss = model(...)
    assert logits.shape == (B, num_classes)
```

### Integration Tests
```bash
# Test full pipeline on small data
python scripts/run_tta.py \
    data.corruptions='["gaussian_noise"]' \
    data.severities='[1]' \
    data.batch_size=2 \
    trainer.fast_dev_run=true
```

## Performance Optimization

### 1. Data Loading
- Increase `num_workers`
- Use `persistent_workers=true`
- Enable `pin_memory=true`

### 2. Mixed Precision
- Use `"16-mixed"` for A100/V100
- Use `"bf16-mixed"` for A100 (better stability)

### 3. Distributed Training
- Use DDP for multi-GPU single node
- Use FSDP for very large models
- Tune `batch_size` per GPU

### 4. Compilation (PyTorch 2.0+)
```python
# In create_model
model = torch.compile(model)
```

## Common Issues

### Issue: "No trainable parameters found"
**Solution**: Check `update_auxiliary` and `update_task_norm_only` settings.

### Issue: NaN loss
**Solution**: 
- Reduce learning rate
- Check gradient clipping
- Verify preprocessing

### Issue: Slow training
**Solution**:
- Increase `num_workers`
- Use mixed precision
- Profile to find bottleneck

### Issue: OOM on multi-GPU
**Solution**:
- Reduce `batch_size` per GPU
- Use gradient accumulation
- Try FSDP instead of DDP
