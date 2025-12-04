import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
from tqdm import tqdm
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import glob

# ---------------------- 1. 配置参数 ----------------------
DATASET_ROOT = r"D:\Study\计算机视觉\第十节\dbs"  # 确保指向s1-s62的上级目录
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8  # 降低batch_size，适配更多GPU
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{DEVICE}")


# ---------------------- 2. 数据集扫描与验证 ----------------------
def scan_and_filter_dataset(root):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
    valid_class_paths = []

    for class_folder in glob.glob(os.path.join(root, "*")):
        if not os.path.isdir(class_folder):
            continue

        image_count = 0
        for ext in supported_extensions:
            image_count += len(glob.glob(os.path.join(class_folder, f"*{ext}"), recursive=False))

        if image_count > 0:
            class_name = os.path.basename(class_folder)
            valid_class_paths.append((class_folder, class_name, image_count))
            print(f"✅ 类别 '{class_name}'：{image_count} 张有效图像")
        else:
            class_name = os.path.basename(class_folder)
            print(f"❌ 跳过空类别 '{class_name}'")

    print(f"\n扫描完成：共找到 {len(valid_class_paths)} 个有效类别")
    return valid_class_paths


# 执行扫描
print(f"\n正在扫描数据集：{DATASET_ROOT}")
valid_class_info = scan_and_filter_dataset(DATASET_ROOT)
if len(valid_class_info) == 0:
    raise ValueError(f"错误：未找到有效类别！请检查数据集路径")


# ---------------------- 3. 自定义数据集 ----------------------
class CustomFaceDataset(Dataset):
    def __init__(self, valid_class_info, transform=None):
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        # 构建类别映射
        for idx, (class_path, class_name, _) in enumerate(valid_class_info):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            # 收集图像
            supported_extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            for ext in supported_extensions:
                image_paths = glob.glob(os.path.join(class_path, f"*{ext}"), recursive=False)
                for img_path in image_paths:
                    self.samples.append((img_path, idx))

        self.transform = transform
        print(f"\n数据集初始化完成：{len(self.samples)} 个样本，{len(self.classes)} 个类别")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        except (UnidentifiedImageError, OSError) as e:
            print(f"警告：跳过损坏图像 {path} -> {str(e)}")
            random_img = torch.randn(3, IMAGE_SIZE[0], IMAGE_SIZE[1])
            return random_img, target


# ---------------------- 4. 人脸预处理 ----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_preprocess(img):
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        x = max(0, x - 20)
        y = max(0, y - 30)
        w = min(img_cv.shape[1] - x, w + 40)
        h = min(img_cv.shape[0] - y, h + 60)
        face_roi = img_cv[y:y + h, x:x + w]
        face_processed = cv2.resize(face_roi, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    else:
        face_processed = cv2.resize(img_cv, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

    return Image.fromarray(cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB))


# ---------------------- 5. 数据预处理+增强 ----------------------
train_transform = transforms.Compose([
    transforms.Lambda(face_preprocess),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(face_preprocess),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------- 6. 数据集加载与划分 ----------------------
full_dataset = CustomFaceDataset(valid_class_info=valid_class_info, transform=None)
print(f"有效类别：{full_dataset.classes}")

# 划分训练集/验证集
train_size = int(0.7 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 应用变换
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# 数据加载器
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True, drop_last=False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True, drop_last=False
)

print(f"\n训练集：{len(train_dataset)} 个样本")
print(f"验证集：{len(val_dataset)} 个样本")

# ---------------------- 7. 模型定义（动态适配类别数） ----------------------
# 解决pretrained警告：用weights替代pretrained
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# 解冻最后3层backbone
for idx, m in enumerate(model.features):
    if idx >= len(model.features) - 3:
        for param in m.parameters():
            param.requires_grad = True
    else:
        for param in m.parameters():
            param.requires_grad = False

# 动态获取实际类别数
NUM_CLASSES_ACTUAL = len(full_dataset.classes)
# 替换分类头
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(model.last_channel, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES_ACTUAL)  # 适配实际类别数
)

model = model.to(DEVICE)
print(f"\n模型初始化完成：分类头适配 {NUM_CLASSES_ACTUAL} 个类别")

# ---------------------- 8. 训练配置（移除verbose参数） ----------------------
optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
criterion = nn.CrossEntropyLoss()

# 移除verbose=True（兼容低版本PyTorch）
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3,
    min_lr=1e-6
)

# ---------------------- 9. 训练循环 ----------------------
best_val_acc = 0.0
early_stop_patience = 8
early_stop_counter = 0

for epoch in range(NUM_EPOCHS):
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [训练]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (preds == labels).sum().item()

    train_avg_loss = train_loss / train_total
    train_acc = train_correct / train_total

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS} [验证]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_avg_loss = val_loss / val_total
    val_acc = val_correct / val_total

    # 学习率调度
    scheduler.step(val_avg_loss)

    # 保存最优模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        early_stop_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
            'class_names': full_dataset.classes,
            'num_classes': NUM_CLASSES_ACTUAL
        }, "best_face_model_no_dlib.pth")
        print(f"✅ 保存最优模型（验证准确率：{best_val_acc:.4f}）")
    else:
        early_stop_counter += 1
        print(f"⚠️  早停计数器：{early_stop_counter}/{early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print("❌ 验证准确率连续8轮无提升，触发早停")
            break

    # 打印日志
    print(f"""
    Epoch {epoch + 1}/{NUM_EPOCHS}
    ├─ 训练：损失={train_avg_loss:.4f} | 准确率={train_acc:.4f}
    └─ 验证：损失={val_avg_loss:.4f} | 准确率={val_acc:.4f}
    """)

print(f"\n训练完成！最佳验证准确率：{best_val_acc:.4f}")
print(f"最优模型已保存为：best_face_model_no_dlib.pth")
