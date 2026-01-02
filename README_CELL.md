# Cell TTA - 细胞数据集测试时自适应

基于 DUSA (Diffusion-guided Unsupervised Self-Adaptation) 框架的细胞分类 TTA 项目。

## 模型架构

- **判别模型**: ViT-B/16 (输入尺寸 224x224)
- **生成模型**: SiT-B/4 (输入尺寸 512x512)

## 项目结构

```
├── configs/
│   ├── config_cell.yaml          # 主配置文件
│   ├── data/
│   │   └── cell.yaml             # 数据集配置
│   └── model/
│       └── combined_cell.yaml    # 模型配置
├── scripts/
│   └── run_cell_tta.py           # 运行脚本
├── src/
│   ├── data/
│   │   ├── cell_dataset.py       # 细胞数据集加载
│   │   └── transforms.py         # 数据变换
│   ├── models/
│   │   ├── discriminative.py     # 判别模型 (ViT-B/16)
│   │   ├── generative/
│   │   │   └── repa_sit.py       # 生成模型 (SiT-B/4)
│   │   └── combined.py           # 组合模型
│   └── tta/
│       └── module.py             # TTA 模块
└── cell_code/                    # 训练代码参考
    ├── ViT/                      # ViT 训练代码
    └── SiT/                      # SiT 训练代码
```

## 数据集格式

数据集需要按照 ImageFolder 格式组织：

```
dataset_root/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image1.jpg
│   └── ...
└── class_N/
    └── ...
```

## 配置

### 1. 数据路径配置

编辑 `configs/config_cell.yaml`:

```yaml
data_root: "/path/to/cell/dataset/"  # 设置为你的数据集路径
```

### 2. 模型权重配置

编辑 `configs/model/combined_cell.yaml`:

```yaml
discriminative:
  checkpoint_path: "/path/to/vit_checkpoint.pt"  # ViT-B/16 权重路径
  num_classes: 5  # 根据数据集类别数调整

generative:
  sit_checkpoint: "/path/to/sit_checkpoint.pt"  # SiT-B/4 权重路径
  num_classes: 5  # 必须与判别模型一致
```

### 3. 类别数配置

确保 `num_classes` 与你的数据集类别数一致。

## 运行

```bash
# 使用默认配置运行
python scripts/run_cell_tta.py

# 覆盖配置参数
python scripts/run_cell_tta.py data_root=/path/to/data \
    model.discriminative.checkpoint_path=/path/to/vit.pt \
    model.generative.sit_checkpoint=/path/to/sit.pt
```

## 训练模型

### 训练 ViT-B/16

使用 `cell_code/ViT/train.py`:

```bash
python cell_code/ViT/train.py --train_dir /path/to/train_data --epochs 30
```

### 训练 SiT-B/4

使用 `cell_code/SiT/train.py`:

```bash
torchrun --nproc_per_node=4 cell_code/SiT/train.py \
    --data-path /path/to/train_data \
    --model SiT-B/4 \
    --image-size 512 \
    --num-classes 5
```

## 关键参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.input_size` | 判别模型输入尺寸 | 224 |
| `model.generative.image_size` | 生成模型输入尺寸 | 512 |
| `model.generative.topk` | Top-k 类别采样 | 4 |
| `model.generative.rand_budget` | 随机采样数量 | 2 |
| `optimizer.learning_rate` | 学习率 | 4e-5 |
| `tta.forward_mode` | 前向模式 | normed_logits_with_logits |

## 注意事项

1. 生成模型使用 512x512 输入，判别模型使用 224x224 输入
2. VAE 会将图像从 512x512 编码到 64x64 的 latent space
3. 确保 GPU 显存足够（建议 16GB+）
