"""
DenseNet121 全参数微调训练脚本
==============================
使用 timm 库加载预训练的 DenseNet121，在 ImageNette 数据集上进行全参数微调。
支持 AMP 混合精度加速、TensorBoard 可视化、预热 + 余弦退火学习率调度。
"""

import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import math

# ============================
# 超参数配置
# ============================
DATA_ROOT = "/home/ivi/zqx/ImageNette"  # 数据集根目录，需包含 train/ 和 val/ 子目录
EPOCHS = 50                             # 总训练轮数
WARMUP_EPOCHS = 3                       # 线性预热轮数（前 3 epoch）
COSINE_EPOCHS = 47                      # 余弦退火轮数（后 47 epoch）
BATCH_SIZE = 64                         # 批大小
NUM_WORKERS = 8                         # DataLoader 子进程数
LR = 1e-4                               # 峰值学习率
WEIGHT_DECAY = 0.01                     # AdamW 权重衰减系数
PORT = 6006                             # TensorBoard 端口号

# ============================
# 数据增强与预处理
# ============================
# 训练集：随机裁剪 + 水平翻转 + 归一化
transform_train = T.Compose([
    T.RandomResizedCrop(224),           # 随机裁剪并缩放到 224x224
    T.RandomHorizontalFlip(),           # 随机水平翻转
    T.ToTensor(),                       # 转为 Tensor 并缩放到 [0,1]
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet 标准化
])

# 验证集：缩放 + 中心裁剪 + 归一化（无随机增强）
transform_val = T.Compose([
    T.Resize(256),                      # 短边缩放到 256
    T.CenterCrop(224),                  # 中心裁剪 224x224
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================
# 数据集与 DataLoader
# ============================
# ImageFolder 要求目录结构为: root/train/<class>/<image> 和 root/val/<class>/<image>
train_ds = ImageFolder(f"{DATA_ROOT}/train", transform=transform_train)
val_ds = ImageFolder(f"{DATA_ROOT}/val", transform=transform_val)

train_loader = DataLoader(
    train_ds, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)
val_loader = DataLoader(
    val_ds, BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

print(f"训练集样本数: {len(train_ds)}")
print(f"验证集样本数: {len(val_ds)}")
print(f"类别数: {len(train_ds.classes)}")
print(f"类别: {train_ds.classes}")

# ============================
# 模型构建
# ============================
# timm.create_model 自动下载预训练权重并替换分类头为 num_classes
model = timm.create_model("densenet121", pretrained=True, num_classes=len(train_ds.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备: {device}")

# ============================
# 损失函数 & 优化器
# ============================
criterion = nn.CrossEntropyLoss()                          # 多分类交叉熵损失
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # AdamW 优化器

# ============================
# 学习率调度: 线性预热 + 余弦退火
# ============================
total_steps = EPOCHS * len(train_loader)       # 总迭代步数
warmup_steps = WARMUP_EPOCHS * len(train_loader)  # 预热步数

def lr_lambda(current_step):
    """
    自定义学习率调度函数:
    - 前 warmup_steps 步: 从 0 线性增长到 1 (峰值 LR)
    - 之后: 余弦退火从 1 衰减到 0
    """
    if current_step < warmup_steps:
        # 线性预热: step / warmup_steps
        return float(current_step) / float(max(1, warmup_steps))
    # 余弦退火: 0.5 * (1 + cos(pi * progress))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda)

# ============================
# AMP (自动混合精度) 初始化
# ============================
scaler = GradScaler()          # 梯度缩放器，防止 FP16 下梯度下溢

# ============================
# TensorBoard 日志记录
# ============================
# 启动后在终端执行: tensorboard --logdir runs --port 6006
writer = SummaryWriter()

# ============================
# 训练循环
# ============================
best_acc = 0.0                 # 最佳验证准确率
global_step = 0                # 全局迭代步数

for epoch in range(1, EPOCHS + 1):
    # ---------- 训练阶段 ----------
    model.train()              # 启用 BatchNorm / Dropout 训练模式
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # 清空梯度

        # AMP 前向传播 + 损失计算 (FP16 加速)
        with autocast():
            outputs = model(images)                 # [B, num_classes]
            loss = criterion(outputs, labels)       # 标量

        # AMP 反向传播 (梯度缩放)
        scaler.scale(loss).backward()               # 缩放 loss 后反向传播
        scaler.step(optimizer)                      # 反缩放梯度并更新参数
        scaler.update()                             # 准备下一次缩放

        scheduler.step()                            # 更新学习率
        global_step += 1

        # 统计
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)               # 取最大 logit 对应的类别
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total

    # ---------- 验证阶段 ----------
    model.eval()               # 禁用 Dropout / 固定 BatchNorm
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():      # 验证阶段不计算梯度
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= val_total
    val_acc = val_correct / val_total

    # ---------- TensorBoard 日志 ----------
    writer.add_scalar("train/loss", train_loss, epoch)
    writer.add_scalar("train/acc", train_acc, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/acc", val_acc, epoch)
    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

    # ---------- 控制台输出 ----------
    print(
        f"Epoch {epoch:2d}/{EPOCHS}  "
        f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
        f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
        f"lr={optimizer.param_groups[0]['lr']:.2e}"
    )

    # ---------- 保存最佳模型 ----------
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> 保存最佳模型 (val_acc={val_acc:.4f})")

# ============================
# 训练完成
# ============================
writer.close()
print(f"\n训练完成! 最佳验证准确率: {best_acc:.4f}")
