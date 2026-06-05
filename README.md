# DenseNet121 — ImageNette 全参数微调

使用 [timm](https://github.com/huggingface/pytorch-image-models) 库加载预训练的 **DenseNet121**，在 [ImageNette](https://github.com/fastai/imagenette) 数据集上进行全参数微调。

## 环境要求

| 依赖 | 版本 (示例) |
|------|------------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 1.10 |
| torchvision | ≥ 0.11 |
| timm | ≥ 0.9 |
| tensorboard | ≥ 2.9 |

安装依赖：

```bash
pip install torch torchvision timm tensorboard
```

## 数据集

使用 [ImageNette](https://s3.amazonaws.com/fastai-imageclas/imagenette2.tgz) — ImageNet 的子集，包含 10 个易于区分的类别。

数据集目录结构要求：

```
ImageNette/
├── train/
│   ├── n01440764/
│   ├── n02102040/
│   ├── n02979186/
│   ├── n03000684/
│   ├── n03028079/
│   ├── n03394916/
│   ├── n03417042/
│   ├── n03425413/
│   ├── n03445777/
│   └── n03888257/
└── val/
    ├── n01440764/
    ├── n02102040/
    ├── n02979186/
    ├── n03000684/
    ├── n03028079/
    ├── n03394916/
    ├── n03417042/
    ├── n03425413/
    ├── n03445777/
    └── n03888257/
```

> 数据集路径可在 `train.py` 的 `DATA_ROOT` 变量中修改。

## 快速开始

### 1. 训练

```bash
python train.py
```

### 2. 查看 TensorBoard

```bash
tensorboard --logdir runs --port 6006
```

打开浏览器访问 `http://localhost:6006`。

## 超参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `DATA_ROOT` | `/home/ivi/zqx/ImageNette` | 数据集根目录 |
| `EPOCHS` | 50 | 总训练轮数 |
| `WARMUP_EPOCHS` | 3 | 线性预热轮数 |
| `COSINE_EPOCHS` | 47 | 余弦退火轮数 |
| `BATCH_SIZE` | 64 | 批大小 |
| `LR` | 1e-4 | 峰值学习率 |
| `WEIGHT_DECAY` | 0.01 | AdamW 权重衰减 |
| `NUM_WORKERS` | 8 | DataLoader 子进程数 |

## 学习率调度策略

采用 **线性预热 + 余弦退火** 调度：

1. **前 3 epoch (预热期)**: 学习率从 0 线性增长到峰值 `1e-4`
2. **后 47 epoch (余弦退火期)**: 学习率按余弦曲线从 `1e-4` 平滑衰减到 `0`

```
lr = 0.5 * (1 + cos(π * progress)) * peak_lr
```

## 训练技巧

- **AMP (Automatic Mixed Precision)**: 使用 `torch.cuda.amp` 自动混合精度训练，在前向和反向传播中使用 FP16 加速，权重更新时保持 FP32 精度。
- **AdamW**: 使用 AdamW 优化器替代传统 Adam，将权重衰减与自适应学习率解耦。
- **全参数微调**: 不对任何层冻结，所有参数参与训练。
- **数据增强**: 随机裁剪 + 水平翻转，提升泛化能力。

## 输出文件

- `runs/` — TensorBoard 日志文件目录
- `best_model.pth` — 验证集上表现最佳的模型权重

## 结果

在 ImageNette 验证集上，DenseNet121 微调 50 epoch 的最佳验证准确率约为 **98.7%**。
