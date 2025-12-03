# app.py
import os
import zipfile
import tempfile
import shutil
import glob
import time
import base64
import gzip
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image, UnidentifiedImageError
import qrcode
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# optional pyzbar for QR decode
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False

# ---------------------- 全局配置 ----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_PATH = "best_face_model.pth"

print(f"Device: {DEVICE}")

# Globals for loaded model
face_model = None
MODEL_OK = False
CLASS_NAMES = []

# ---------------------- 数据集辅助类 ----------------------
class SimpleImageFolderDataset(Dataset):
    """类似 ImageFolder，但更容错并允许自定义 transform"""
    def __init__(self, root, transform=None):
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        # scan subfolders
        folders = [d for d in sorted(glob.glob(os.path.join(root, "*"))) if os.path.isdir(d)]
        for idx, folder in enumerate(folders):
            class_name = os.path.basename(folder)
            # find images
            exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.ppm", "*.tif", "*.tiff", "*.webp")
            imgs = []
            for e in exts:
                imgs.extend(glob.glob(os.path.join(folder, e)))
            if len(imgs) == 0:
                continue
            self.class_to_idx[class_name] = idx
            self.classes.append(class_name)
            for p in imgs:
                self.samples.append((p, idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, label = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except (UnidentifiedImageError, OSError) as e:
            # fallback: random tensor
            tensor = torch.randn(3, 224, 224)
            return tensor, label

# ---------------------- 人脸检测预处理 ----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def face_preprocess_pil(pil_img, output_size=(224,224)):
    """Detect face and crop; fallback to resizing full image."""
    try:
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception:
        pil_img = pil_img.convert("RGB")
        img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        x,y,w,h = faces[0]
        x = max(0, x-20); y = max(0, y-30)
        w = min(img_cv.shape[1]-x, w+40); h = min(img_cv.shape[0]-y, h+60)
        roi = img_cv[y:y+h, x:x+w]
    else:
        roi = img_cv
    roi_resized = cv2.resize(roi, output_size, interpolation=cv2.INTER_CUBIC)
    roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
    return Image.fromarray(roi_rgb)

# ---------------------- 模型加载/构建 ----------------------
def build_model(num_classes):
    """Create MobileNetV2 and replace classifier head for num_classes."""
    try:
        # torchvision newer: weights param
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        try:
            model = models.mobilenet_v2(pretrained=True)
        except Exception:
            model = models.mobilenet_v2(weights=None)
    in_features = getattr(model, "last_channel", 1280)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def load_model_from_path(path):
    """Load checkpoint saved by our training function. Returns (model, class_names)."""
    global face_model, MODEL_OK, CLASS_NAMES
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    class_names = ckpt.get("class_names", None)
    num_classes = ckpt.get("num_classes", None)
    if class_names is None or num_classes is None:
        # try infer
        print("Warning: checkpoint missing class_names/num_classes, attempting best-effort load.")
        # if model_state_dict exists, try to load into a default head (may mismatch)
        num_classes = num_classes or 100
        class_names = class_names or [str(i) for i in range(num_classes)]
    model = build_model(num_classes)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model = model.to(DEVICE)
    model.eval()
    face_model = model
    MODEL_OK = True
    CLASS_NAMES = class_names
    print(f"Loaded model from {path}, classes={len(CLASS_NAMES)}")
    return True

# auto-load if default model exists
if os.path.exists(DEFAULT_MODEL_PATH):
    try:
        load_model_from_path(DEFAULT_MODEL_PATH)
    except Exception as e:
        print("Failed to auto-load existing model:", e)

# ---------------------- 训练函数 ----------------------
def train_on_uploaded_zip(zip_file, epochs=3, batch_size=8, lr=1e-4):
    """
    - zip_file: a gr.File object or path string pointing to a zip that contains class subfolders.
    - Trains a MobileNetV2 head, saves best checkpoint to DEFAULT_MODEL_PATH, and loads it.
    Returns string status and minimal metrics for display.
    """
    global face_model, MODEL_OK, CLASS_NAMES
    t0 = time.time()

    if zip_file is None:
        return "没有上传数据集(zip)。", ""

    # create temporary directory
    workdir = tempfile.mkdtemp(prefix="gradio_train_")
    try:
        # save uploaded file to disk if it's a SpooledTemporaryFile-like
        zip_path = os.path.join(workdir, "data.zip")
        if hasattr(zip_file, "name") and os.path.exists(zip_file.name):
            # gradio may give a local path
            shutil.copy(zip_file.name, zip_path)
        else:
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

        # extract
        extract_dir = os.path.join(workdir, "data")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # build dataset
        print("Scanning extracted folders...")
        dataset = SimpleImageFolderDataset(extract_dir, transform=None)
        if len(dataset) == 0:
            return "解压后未发现有效图像数据。请检查 zip 结构（应为每类子文件夹，如 s1, s2 ...）。", ""

        class_names = dataset.classes
        num_classes = len(class_names)
        # define transforms (same as earlier)
        train_transform = transforms.Compose([
            transforms.Lambda(lambda img: face_preprocess_pil(img, output_size=(224,224))),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.25, 0.25, 0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(lambda img: face_preprocess_pil(img, output_size=(224,224))),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

        # split
        total = len(dataset)
        train_count = int(0.7 * total)
        val_count = total - train_count
        train_ds, val_ds = random_split(dataset, [train_count, val_count], generator=torch.Generator().manual_seed(42))
        # set transforms (dataset is the underlying SimpleImageFolderDataset)
        train_ds.dataset.transform = train_transform
        val_ds.dataset.transform = val_transform

        train_loader = DataLoader(train_ds, batch_size=int(batch_size), shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=int(batch_size), shuffle=False, num_workers=0)

        # build model
        model = build_model(num_classes)
        model = model.to(DEVICE)

        # freeze backbone except last few layers (optional)
        # here we freeze all except classifier parameters for speed
        for name, p in model.named_parameters():
            if "classifier" not in name:
                p.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=float(lr), weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6)

        best_val_acc = 0.0
        best_ckpt = None
        history = []

        for epoch in range(int(epochs)):
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", leave=False)
            for imgs, labels in pbar:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                running_total += labels.size(0)
                running_correct += (preds == labels).sum().item()
                pbar.set_postfix(loss=running_loss/running_total if running_total>0 else 0,
                                 acc=running_correct/running_total if running_total>0 else 0)
            train_loss = running_loss / running_total if running_total>0 else 0
            train_acc = running_correct / running_total if running_total>0 else 0

            # validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}/{epochs}", leave=False):
                    imgs = imgs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (preds == labels).sum().item()
            val_loss_avg = val_loss / val_total if val_total>0 else 0
            val_acc = val_correct / val_total if val_total>0 else 0
            scheduler.step(val_loss_avg)

            history.append((train_loss, train_acc, val_loss_avg, val_acc))
            print(f"Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_loss={val_loss_avg:.4f} val_acc={val_acc:.4f}")

            # save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_ckpt = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "class_names": class_names,
                    "num_classes": num_classes,
                    "best_val_acc": best_val_acc
                }
                # save to DEFAULT_MODEL_PATH
                torch.save(best_ckpt, DEFAULT_MODEL_PATH)
                print(f"Saved best model to {DEFAULT_MODEL_PATH} (val_acc={best_val_acc:.4f})")

        # load back saved model to global
        try:
            load_model_from_path(DEFAULT_MODEL_PATH)
        except Exception as e:
            print("Failed to load after training:", e)

        elapsed = time.time() - t0
        summary = f"训练完成。best_val_acc={best_val_acc:.4f}，耗时 {elapsed:.1f}s。已保存并加载到 {DEFAULT_MODEL_PATH}。"
        return summary, f"{best_val_acc:.4f}"
    except Exception as e:
        return f"训练过程出现异常：{e}", ""
    finally:
        # cleanup
        try:
            shutil.rmtree(workdir)
        except Exception:
            pass

# ---------------------- 手动上传模型加载 ----------------------
def upload_and_load_model(model_file):
    """
    Accepts uploaded .pth file and loads it as current model.
    """
    if model_file is None:
        return "未选择文件。"
    try:
        # model_file may be a local path (gradio provides .name) or a file-like
        if hasattr(model_file, "name") and os.path.exists(model_file.name):
            path = model_file.name
            shutil.copy(path, DEFAULT_MODEL_PATH)
        else:
            data = model_file.read()
            with open(DEFAULT_MODEL_PATH, "wb") as f:
                f.write(data)
        load_model_from_path(DEFAULT_MODEL_PATH)
        return f"模型已加载并保存为 {DEFAULT_MODEL_PATH}，类别数={len(CLASS_NAMES)}"
    except Exception as e:
        return f"加载模型失败：{e}"

# ---------------------- QR 工具：生成/重试/解码/重构 ----------------------
def image_to_jpeg_bytes(pil_img, quality=75, max_side=None):
    img = pil_img
    if max_side is not None:
        w,h = img.size
        scale = max_side / max(w,h)
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()

def try_generate_qr_with_retries(pil_img, allow_gzip=True):
    """Try multiple scales & qualities and gzip option. Return (payload, qr_pil) or (None,None)."""
    error_levels = [qrcode.constants.ERROR_CORRECT_H, qrcode.constants.ERROR_CORRECT_M, qrcode.constants.ERROR_CORRECT_L]
    scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    qualities = [85, 70, 55, 40, 30, 20]
    w0,h0 = pil_img.size
    last_exc = None
    for err in error_levels:
        for scale in scale_factors:
            max_side = int(max(1, max(w0,h0) * scale))
            if max_side < 24:
                continue
            for q in qualities:
                try:
                    jpeg_bytes = image_to_jpeg_bytes(pil_img, quality=q, max_side=max_side)
                    payload = base64.b64encode(jpeg_bytes).decode("utf-8")
                    qr = qrcode.QRCode(error_correction=err, box_size=4, border=2)
                    qr.add_data(payload)
                    try:
                        qr.make(fit=True)
                        img = qr.make_image(fill_color="black", back_color="white")
                        if hasattr(img, "convert"):
                            img = img.convert("RGB")
                        return payload, img
                    except ValueError as ve:
                        last_exc = ve
                    if allow_gzip:
                        try:
                            gz = gzip.compress(jpeg_bytes)
                            payload_gz = "GZIP1:" + base64.b64encode(gz).decode("utf-8")
                            qr2 = qrcode.QRCode(error_correction=err, box_size=4, border=2)
                            qr2.add_data(payload_gz)
                            try:
                                qr2.make(fit=True)
                                img2 = qr2.make_image(fill_color="black", back_color="white")
                                if hasattr(img2, "convert"):
                                    img2 = img2.convert("RGB")
                                return payload_gz, img2
                            except ValueError as ve2:
                                last_exc = ve2
                        except Exception as e:
                            last_exc = e
                except Exception as e:
                    last_exc = e
    print("QR generate failed:", last_exc)
    return None, None

def decode_qr_from_pil(pil_qr):
    if pil_qr is None:
        return None
    if _HAS_PYZBAR:
        try:
            res = pyzbar_decode(pil_qr)
            if res:
                return res[0].data.decode("utf-8")
        except Exception:
            pass
    # fallback OpenCV
    try:
        arr = cv2.cvtColor(np.array(pil_qr), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, pts, _ = detector.detectAndDecode(arr)
        if data:
            return data
    except Exception:
        pass
    return None

def reconstruct_from_payload(payload):
    if not payload:
        return None
    try:
        if payload.startswith("GZIP1:"):
            b64 = payload[len("GZIP1:"):]
            gz = base64.b64decode(b64)
            raw = gzip.decompress(gz)
            return Image.open(BytesIO(raw)).convert("RGB")
        else:
            raw = base64.b64decode(payload)
            return Image.open(BytesIO(raw)).convert("RGB")
    except Exception as e:
        print("reconstruct failed:", e)
        return None

# ---------------------- 推理/接口函数 ----------------------
def infer_and_qr(upload_img):
    """
    Input: numpy array from gr.Image
    Outputs:
      - compressed image np
      - qr image np or None
      - reconstructed face np or None
      - decoded text str
      - cnn result str
    """
    # defaults
    compressed_np = None
    qr_np = None
    recon_np = None
    decoded_text = ""
    cnn_text = ""

    if upload_img is None:
        return None, None, None, "未上传图片", "模型未加载" if not MODEL_OK else "未上传图片"

    try:
        pil = Image.fromarray(upload_img).convert("RGB")
    except Exception as e:
        return None, None, None, f"无法读取上传图像：{e}", "模型未加载" if not MODEL_OK else "读取失败"

    # initial compress (keep reasonable)
    def auto_compress(img, max_side=800):
        w,h = img.size
        if max(w,h) <= max_side:
            return img
        scale = max_side / max(w,h)
        return img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

    compressed = auto_compress(pil, max_side=800)
    compressed_np = np.array(compressed)

    # try generate qr
    payload, qr_pil = try_generate_qr_with_retries(compressed)
    if qr_pil is None:
        # try progressive shrink
        img = compressed
        w,h = img.size
        while max(w,h) > 24:
            w = max(16, int(w*0.8)); h = max(16, int(h*0.8))
            img = img.resize((w,h), Image.LANCZOS)
            payload, qr_pil = try_generate_qr_with_retries(img)
            if qr_pil is not None:
                compressed = img
                compressed_np = np.array(compressed)
                break
    if qr_pil is None:
        decoded_text = "二维码生成失败（多次压缩仍无法生成）"
    else:
        qr_np = np.array(qr_pil)
        decoded = decode_qr_from_pil(qr_pil)
        if decoded:
            decoded_text = decoded
            recon = reconstruct_from_payload(decoded)
            recon_np = np.array(recon) if recon is not None else None
        else:
            decoded_text = "二维码已生成，但解码失败"

    # cnn recognition on original (or compressed) image if model loaded
    if MODEL_OK and face_model is not None:
        try:
            # preprocess same as training (detect face crop)
            inp = face_preprocess_pil(pil, output_size=(224,224))
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            t = transform(inp).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = face_model(t)
                probs = torch.softmax(out, dim=1)
                conf, idx = torch.max(probs, 1)
                idx = int(idx.item())
                conf = float(conf.item()*100.0)
                name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
                cnn_text = f"{name} ({conf:.2f}%)"
        except Exception as e:
            cnn_text = f"CNN 推理失败: {e}"
    else:
        cnn_text = "模型未加载，无法识别"

    return compressed_np, qr_np, recon_np, decoded_text, cnn_text

def decode_qr_upload(qr_img):
    """Decode an uploaded QR image and reconstruct."""
    if qr_img is None:
        return None, "未上传二维码图片"
    try:
        pil = Image.fromarray(qr_img).convert("RGB")
    except Exception as e:
        return None, f"读取二维码失败: {e}"
    decoded = decode_qr_from_pil(pil)
    if not decoded:
        return None, "二维码解码失败"
    recon = reconstruct_from_payload(decoded)
    if recon is None:
        return None, "解码成功，但重构图片失败"
    return np.array(recon), decoded

# ---------------------- Gradio UI ----------------------
with gr.Blocks(title="人脸识别 + 二维码系统（在线训练 + 推理）") as demo:
    gr.Markdown("# 人脸识别 + 二维码系统（支持在线上传数据集训练 & 推理）")

    with gr.Tab("模型训练 / 加载"):
        gr.Markdown("上传 zip 数据集（每个类一个文件夹，例如 s1, s2, ...），设置训练参数，然后点击 `开始训练`。训练结束后会自动保存并加载模型 `best_face_model.pth`。")
        with gr.Row():
            zip_input = gr.File(label="上传数据集 (ZIP)", file_count="single")
            model_upload = gr.File(label="或上传已有模型 (.pth)", file_count="single")
        with gr.Row():
            epochs_slider = gr.Number(value=3, precision=0, label="Epochs (建议小规模测试时用 1-3)")
            batch_input = gr.Number(value=8, precision=0, label="Batch size")
            lr_input = gr.Number(value=1e-4, label="Learning rate")
        with gr.Row():
            train_btn = gr.Button("开始训练")
            load_model_btn = gr.Button("加载上传模型")
        train_out = gr.Textbox(label="训练输出 / 状态")
        best_acc_out = gr.Textbox(label="最佳验证准确率（若有）")

        def _train_click(zip_file, epochs, batch_size, lr):
            return train_on_uploaded_zip(zip_file, epochs=epochs, batch_size=batch_size, lr=lr)
        train_btn.click(fn=_train_click, inputs=[zip_input, epochs_slider, batch_input, lr_input], outputs=[train_out, best_acc_out])

        def _load_model_click(mf):
            return upload_and_load_model(mf)
        load_model_btn.click(fn=_load_model_click, inputs=[model_upload], outputs=[train_out])

    with gr.Tab("推理 / 二维码"):
        gr.Markdown("上传人脸图片进行：自动压缩 → 生成二维码（必要时多次压缩重试）→ 解码并重构 → 使用模型识别")
        with gr.Row():
            img_in = gr.Image(label="上传人脸图片", type="numpy")
            with gr.Column():
                out_compressed = gr.Image(label="压缩后的人脸 (预览)")
                out_qr = gr.Image(label="生成的二维码")
                out_recon = gr.Image(label="从二维码重构的人脸")
        out_decoded = gr.Textbox(label="二维码解码内容")
        out_cnn = gr.Textbox(label="CNN 识别结果")
        run_btn = gr.Button("开始处理")
        run_btn.click(fn=infer_and_qr, inputs=[img_in], outputs=[out_compressed, out_qr, out_recon, out_decoded, out_cnn])

        gr.Markdown("或上传二维码图片直接解码并重构：")
        qr_upload = gr.Image(label="上传二维码图", type="numpy")
        qr_decode_btn = gr.Button("解码二维码")
        qr_recon_out = gr.Image(label="重构图像")
        qr_decoded_text = gr.Textbox(label="解码文本")
        qr_decode_btn.click(fn=decode_qr_upload, inputs=[qr_upload], outputs=[qr_recon_out, qr_decoded_text])

    with gr.Tab("工具 & 状态"):
        gr.Markdown("当前模型加载状态与类别信息")
        model_status = gr.Textbox(label="模型状态", value="已加载" if MODEL_OK else "未加载")
        model_classes = gr.Textbox(label="类别数/示例", value=f"{len(CLASS_NAMES)} / {CLASS_NAMES[:10]}")
        refresh_btn = gr.Button("刷新状态")
        def _refresh():
            return ("已加载" if MODEL_OK else "未加载", f"{len(CLASS_NAMES)} / {CLASS_NAMES[:10]}")
        refresh_btn.click(fn=_refresh, inputs=[], outputs=[model_status, model_classes])

# 启动界面
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        inbrowser=True
    )