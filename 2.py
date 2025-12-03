# -*- coding: utf-8 -*-
"""
最终整合版：基于 PyQt5 的 人脸 <-> 二维码 系统（含 CNN 识别）
保存为 2.py 并运行。
依赖：PyQt5, Pillow, qrcode, opencv-python, torch, torchvision, pyzbar (可选)
"""
import sys
import base64
import gzip
import numpy as np
import cv2
from PIL import Image
import qrcode
from io import BytesIO

# optional pyzbar
try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    _HAS_PYZBAR = True
except Exception:
    _HAS_PYZBAR = False

import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize

# ---------------------- 人脸识别模型推理类 ----------------------
class FaceRecognitionInfer:
    def __init__(self, model_path="D:\\Study\\计算机视觉\\第十节\\best_face_model_no_dlib.pth"):
        # 加载 checkpoint（容错）
        try:
            checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
        except Exception as e:
            raise FileNotFoundError(f"模型文件加载失败：{model_path}\n错误：{e}")

        if "model_state_dict" not in checkpoint:
            raise ValueError("checkpoint 中缺少 'model_state_dict'，无法恢复模型权重。")

        self.class_names = checkpoint.get("class_names", [str(i) for i in range(100)])
        self.num_classes = checkpoint.get("num_classes", len(self.class_names))

        # 创建 mobilenet_v2（兼容 torchvision 版本）
        try:
            self.model = models.mobilenet_v2(weights=None)
        except TypeError:
            self.model = models.mobilenet_v2(pretrained=False)

        in_features = getattr(self.model, "last_channel", None)
        if in_features is None:
            try:
                in_features = self.model.classifier[1].in_features
            except Exception:
                in_features = 1280

        # 重建 classifier 以匹配 num_classes
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )

        try:
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except Exception as e:
            raise RuntimeError(f"模型权重加载发生错误：{e}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        try:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception:
            self.face_cascade = None

        self.IMAGE_SIZE = (224, 224)
        self.transform = transforms.Compose([
            transforms.Lambda(self.face_preprocess),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def face_preprocess(self, img):
        """人脸检测+裁剪+统一尺寸，输入输出均为 PIL.Image"""
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        faces = ()
        if self.face_cascade is not None:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) > 0:
            x, y, w, h = faces[0]
            x = max(0, x - 20)
            y = max(0, y - 30)
            w = min(img_cv.shape[1] - x, w + 40)
            h = min(img_cv.shape[0] - y, h + 60)
            face_roi = img_cv[y:y + h, x:x + w]
            face_processed = cv2.resize(face_roi, self.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
        else:
            # 未检测到人脸，则把整图缩到模型输入尺寸
            face_processed = cv2.resize(img_cv, self.IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

        face_rgb = cv2.cvtColor(face_processed, cv2.COLOR_BGR2RGB)
        return Image.fromarray(face_rgb)

    def infer(self, img_pil):
        """识别人脸，返回 (class_name, confidence_percent)；保证返回有效值"""
        if not isinstance(img_pil, Image.Image):
            img_pil = Image.fromarray(np.array(img_pil)).convert("RGB")

        img_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            outputs = outputs.float()
            if not torch.isfinite(outputs).all():
                print("Warning: model outputs contain non-finite values:", outputs)
                return "Unknown", 0.0

            probs = torch.softmax(outputs, dim=1)
            max_prob, pred_idx = torch.max(probs, 1)

        idx = pred_idx.item()
        if idx < 0 or idx >= len(self.class_names):
            class_name = str(idx)
        else:
            class_name = self.class_names[idx]

        confidence = float(max_prob.item() * 100.0)
        if not np.isfinite(confidence):
            confidence = 0.0

        return class_name, confidence


# ---------------------- 主界面窗口 ----------------------
class FaceQRSystemWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的人脸二维码识别系统")
        self.setFixedSize(1050, 750)

        # 数据存储
        self.original_face = None    # PIL 原始
        self.compressed_face = None  # PIL 压缩后（用于编码）
        self.qr_code = None          # PIL 二维码图像
        self.qr_content = ""         # 解码后的字符串

        # 尝试加载模型（失败仍允许 QR 功能）
        try:
            self.face_infer = FaceRecognitionInfer()
            self._face_model_loaded = True
        except Exception as e:
            self.face_infer = None
            self._face_model_loaded = False
            QMessageBox.warning(self, "模型加载警告",
                                f"人脸识别模型加载失败：\n{e}\n系统仍可用于二维码编码/解码，但 CNN 识别不可用。")

        self.init_ui()
        self.apply_modern_style()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(25)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 左控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(18)
        control_layout.setContentsMargins(15, 15, 15, 15)

        control_title = QLabel("功能控制区")
        control_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(control_title)

        self.btn_select_face = QPushButton("选择人脸图像")
        self.btn_compress = QPushButton("人脸压缩")
        self.btn_gen_qr = QPushButton("二维码生成")
        self.btn_decode_qr = QPushButton("二维码解码")
        self.btn_recon_face = QPushButton("人脸重构")
        self.btn_cnn_recog = QPushButton("CNN识别")
        self.btn_exit = QPushButton("退出系统")

        # 初始按钮状态
        self.btn_compress.setEnabled(False)
        self.btn_gen_qr.setEnabled(False)
        self.btn_decode_qr.setEnabled(False)
        self.btn_recon_face.setEnabled(False)
        self.btn_cnn_recog.setEnabled(self._face_model_loaded)

        btn_size = QSize(160, 45)
        for btn in [self.btn_select_face, self.btn_compress, self.btn_gen_qr,
                    self.btn_decode_qr, self.btn_recon_face, self.btn_cnn_recog, self.btn_exit]:
            btn.setFixedSize(btn_size)
            btn.setCursor(Qt.PointingHandCursor)
            control_layout.addWidget(btn)

        control_layout.addStretch()
        main_layout.addWidget(control_panel, stretch=1)

        # 右显示区
        display_panel = QWidget()
        display_layout = QGridLayout(display_panel)
        display_layout.setSpacing(20)
        display_layout.setContentsMargins(15, 15, 15, 15)

        self.label_face = QLabel("人脸图像\n(选择后显示)")
        self.label_face.setAlignment(Qt.AlignCenter)
        self.label_face.setMinimumSize(320, 320)
        display_layout.addWidget(self.label_face, 0, 0)

        self.text_qr_content = QTextEdit()
        self.text_qr_content.setPlaceholderText("二维码解码内容将显示在这里...")
        self.text_qr_content.setReadOnly(True)
        display_layout.addWidget(self.text_qr_content, 0, 1)

        self.label_qr = QLabel("二维码图像\n(生成后显示)")
        self.label_qr.setAlignment(Qt.AlignCenter)
        self.label_qr.setMinimumSize(320, 320)
        display_layout.addWidget(self.label_qr, 1, 0)

        self.label_recon_face = QLabel("人脸重构结果\n(解码后显示)")
        self.label_recon_face.setAlignment(Qt.AlignCenter)
        self.label_recon_face.setMinimumSize(320, 320)
        display_layout.addWidget(self.label_recon_face, 1, 1)

        # CNN 结果文本
        self.label_cnn_result = QLabel("CNN识别结果：\n（未识别）")
        self.label_cnn_result.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.label_cnn_result.setMinimumSize(320, 80)
        display_layout.addWidget(self.label_cnn_result, 2, 0, 1, 2)

        main_layout.addWidget(display_panel, stretch=4)

        # 绑定事件（确保以下方法均存在）
        self.btn_select_face.clicked.connect(self.load_face_image)
        self.btn_compress.clicked.connect(self.compress_face_image)
        self.btn_gen_qr.clicked.connect(self.generate_qr_code)
        self.btn_decode_qr.clicked.connect(self.decode_qr_code)
        self.btn_recon_face.clicked.connect(self.reconstruct_face)
        self.btn_cnn_recog.clicked.connect(self.run_cnn_recognition)
        self.btn_exit.clicked.connect(self.close)

    def apply_modern_style(self):
        style_sheet = """
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                                           stop:0 #2C3E50, stop:0.5 #4CA1AF, stop:1 #2980B9);
            }
            QLabel {
                color: #FFFFFF;
                font-family: "微软雅黑";
                font-size: 20px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #27AE60;
                color: #FFFFFF;
                font-family: "微软雅黑";
                font-size: 15px;
                font-weight: bold;
                border: none;
                border-radius: 10px;
                padding: 8px 15px;
            }
            QPushButton:hover {
                background-color: #2ECC71;
            }
            QPushButton:pressed {
                background-color: #219E54;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
                color: #7F8C8D;
            }
            QLabel#display_label {
                border: 3px solid #3498DB;
                border-radius: 12px;
                background-color: rgba(255, 255, 255, 0.9);
                color: #2C3E50;
                font-size: 16px;
                font-family: "微软雅黑";
            }
            QTextEdit {
                border: 3px solid #E67E22;
                border-radius: 12px;
                background-color: rgba(255, 255, 255, 0.95);
                font-family: "Consolas";
                font-size: 14px;
                padding: 12px;
            }
        """
        self.setStyleSheet(style_sheet)
        self.label_face.setObjectName("display_label")
        self.label_qr.setObjectName("display_label")
        self.label_recon_face.setObjectName("display_label")

    # ---------------------- 辅助函数 ----------------------
    def pil_to_qpixmap(self, pil_img):
        """兼容 Pillow 版本：PIL -> QPixmap（优先 PNG bytes -> fromData，回退 numpy）"""
        try:
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            qimg = QImage.fromData(buffer.getvalue())
            if not qimg.isNull():
                return QPixmap.fromImage(qimg)
        except Exception:
            pass
        try:
            arr = np.array(pil_img.convert("RGB"))
            arr = np.ascontiguousarray(arr)
            h, w, ch = arr.shape
            bytes_per_line = ch * w
            qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
            if not qimg.isNull():
                return QPixmap.fromImage(qimg.copy())
        except Exception as e:
            print("pil_to_qpixmap fallback failed:", e)
        return QPixmap()

    def image_to_jpeg_bytes(self, pil_img, quality=75, max_side=None):
        """将 PIL 图像保存为 JPEG bytes，可指定质量与最长边尺寸（不破坏原图）"""
        img = pil_img
        if max_side is not None:
            w, h = img.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=int(quality), optimize=True)
        data = buffer.getvalue()
        buffer.close()
        return data

    def auto_compress_image(self, pil_img, max_side=1024, step=0.85, min_side=64):
        """
        自动把大图压缩到 max_side 以下，按 step 缩放，最小到 min_side 为止
        返回压缩后的 PIL 图像
        """
        img = pil_img
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        # 逐步缩放直到达到阈值
        while max(w, h) > max_side and max(w, h) > min_side:
            w = max(min_side, int(w * step))
            h = max(min_side, int(h * step))
            img = img.resize((w, h), resample=Image.LANCZOS)
        return img

    # ---------------------- 选择 / 加载图片 ----------------------
    def load_face_image(self):
        """选择并加载人脸图像，自动初步压缩并显示"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择人脸图像", "", "支持的图像格式 (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        try:
            img = Image.open(file_path).convert("RGB")
        except Exception as e:
            QMessageBox.critical(self, "加载失败", f"图像加载失败：{e}")
            return

        # 保存原始（可以用于 CNN 识别）
        self.original_face = img

        # 初步自动压缩，防止非常大图直接导致二维码失败
        self.compressed_face = self.auto_compress_image(img, max_side=800, step=0.85, min_side=64)

        # 显示在 label_face
        qpix = self.pil_to_qpixmap(self.compressed_face)
        if not qpix.isNull():
            scaled = qpix.scaled(self.label_face.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_face.setPixmap(scaled)
            self.label_face.setText("")
        else:
            QMessageBox.warning(self, "显示失败", "无法将图像显示为 Qt pixmap。")

        # 启用后续按钮
        self.btn_compress.setEnabled(True)
        self.btn_gen_qr.setEnabled(True)  # allow immediate qr attempt
        if self._face_model_loaded:
            self.btn_cnn_recog.setEnabled(True)

    def compress_face_image(self):
        """手动触发一次压缩（压为原尺寸的50%）"""
        if not self.original_face:
            QMessageBox.warning(self, "提示", "请先选择人脸图像！")
            return
        try:
            resample_filter = Image.Resampling.LANCZOS
        except Exception:
            resample_filter = Image.LANCZOS
        w, h = self.original_face.size
        new_w, new_h = max(1, w // 2), max(1, h // 2)
        self.compressed_face = self.original_face.resize((new_w, new_h), resample=resample_filter)
        QMessageBox.information(self, "压缩完成", "人脸已压缩为原尺寸的50%！")
        # 更新显示
        qpix = self.pil_to_qpixmap(self.compressed_face)
        if not qpix.isNull():
            scaled = qpix.scaled(self.label_face.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label_face.setPixmap(scaled)
            self.label_face.setText("")
        self.btn_gen_qr.setEnabled(True)

    # ---------------------- 自动多轮尝试生成二维码（核心） ----------------------
    def try_generate_qr(self, pil_img):
        """
        在不同参数下尝试生成二维码。
        返回 (payload_string, qr_pil_image) 或 (None, None)
        """
        error_levels = [
            qrcode.constants.ERROR_CORRECT_H,
            qrcode.constants.ERROR_CORRECT_M,
            qrcode.constants.ERROR_CORRECT_L
        ]
        scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # 相对原始 pil_img 大小
        qualities = [85, 70, 55, 40, 30, 20]
        allow_gzip = True
        w0, h0 = pil_img.size
        min_side = 24
        last_exc = None

        for err_level in error_levels:
            for scale in scale_factors:
                max_side = int(max(1, max(w0, h0) * scale))
                if max_side < min_side:
                    continue
                for q in qualities:
                    try:
                        jpeg_bytes = self.image_to_jpeg_bytes(pil_img, quality=q, max_side=max_side)
                        payload_raw = base64.b64encode(jpeg_bytes).decode("utf-8")
                        qr = qrcode.QRCode(error_correction=err_level, box_size=4, border=2)
                        qr.add_data(payload_raw)
                        try:
                            qr.make(fit=True)
                            img = qr.make_image(fill_color="#2C3E50", back_color="white")
                            if hasattr(img, "convert"):
                                img = img.convert("RGB")
                            return payload_raw, img
                        except ValueError as ve:
                            last_exc = ve
                        if allow_gzip:
                            try:
                                gz = gzip.compress(jpeg_bytes)
                                payload_gz = "GZIP1:" + base64.b64encode(gz).decode("utf-8")
                                qr2 = qrcode.QRCode(error_correction=err_level, box_size=4, border=2)
                                qr2.add_data(payload_gz)
                                try:
                                    qr2.make(fit=True)
                                    img2 = qr2.make_image(fill_color="#2C3E50", back_color="white")
                                    if hasattr(img2, "convert"):
                                        img2 = img2.convert("RGB")
                                    return payload_gz, img2
                                except ValueError as ve2:
                                    last_exc = ve2
                            except Exception as e:
                                last_exc = e
                    except Exception as e:
                        last_exc = e
        print("try_generate_qr failed, last_exc:", last_exc)
        return None, None

    def generate_qr_code(self):
        """主按钮：尝试使用 compressed_face 生成二维码，失败则自动多轮压缩后继续尝试"""
        if not self.compressed_face:
            QMessageBox.warning(self, "提示", "请先选择或压缩人脸图像！")
            return

        # 先用当前 compressed_face 尝试
        payload, qr_img = self.try_generate_qr(self.compressed_face)
        if payload is not None:
            self.qr_content = payload
            self.qr_code = qr_img
            qpix = self.pil_to_qpixmap(self.qr_code)
            if not qpix.isNull():
                scaled = qpix.scaled(self.label_qr.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_qr.setPixmap(scaled)
                self.label_qr.setText("")
            self.btn_decode_qr.setEnabled(True)
            return

        # 若失败，进入更激进的自动压缩循环
        img = self.compressed_face.copy()
        w, h = img.size
        min_side = 24
        attempt = 0
        success = False

        while max(w, h) > min_side:
            attempt += 1
            new_w = max(min_side, int(w * 0.8))
            new_h = max(min_side, int(h * 0.8))
            try:
                img = img.resize((new_w, new_h), resample=Image.LANCZOS)
            except Exception:
                img = img.copy().resize((new_w, new_h))
            w, h = img.size
            print(f"[generate_qr_code] second-phase compress attempt {attempt}, size={img.size}")

            payload, qr_img = self.try_generate_qr(img)
            if payload is not None:
                self.qr_content = payload
                self.qr_code = qr_img
                self.compressed_face = img  # update compressed version
                qpix = self.pil_to_qpixmap(self.qr_code)
                if not qpix.isNull():
                    scaled = qpix.scaled(self.label_qr.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.label_qr.setPixmap(scaled)
                    self.label_qr.setText("")
                self.btn_decode_qr.setEnabled(True)
                success = True
                break

        if not success:
            QMessageBox.critical(self, "生成失败",
                                 "二维码生成失败：图片过大且多次压缩后仍无法编码。\n建议使用更小的源图片或进一步降低质量。")
        else:
            print("[generate_qr_code] success after attempts:", attempt)

    # ---------------------- 解码 / 重构 ----------------------
    def decode_qr_code(self):
        if not self.qr_code:
            QMessageBox.warning(self, "提示", "请先生成二维码！")
            return

        decoded_text = None
        try:
            if _HAS_PYZBAR:
                try:
                    decoded_objs = pyzbar_decode(self.qr_code)
                    if decoded_objs:
                        decoded_text = decoded_objs[0].data.decode("utf-8")
                except Exception:
                    decoded_text = None
            if decoded_text is None:
                qr_cv = cv2.cvtColor(np.array(self.qr_code), cv2.COLOR_RGB2BGR)
                detector = cv2.QRCodeDetector()
                data, points, _ = detector.detectAndDecode(qr_cv)
                if data:
                    decoded_text = data
        except Exception as e:
            QMessageBox.warning(self, "解码异常", f"解码时发生异常：{e}")
            decoded_text = None

        if not decoded_text:
            QMessageBox.warning(self, "解码失败", "未识别到二维码内容或数据格式不支持。")
            return

        self.qr_content = decoded_text
        if len(self.qr_content) > 1000:
            display_content = f"二维码内容（前1000字符）：\n{self.qr_content[:1000]}..."
        else:
            display_content = f"二维码内容：\n{self.qr_content}"
        self.text_qr_content.setText(display_content)
        self.btn_recon_face.setEnabled(True)

    def reconstruct_face(self):
        if not self.qr_content:
            QMessageBox.warning(self, "提示", "请先解码二维码！")
            return
        try:
            if self.qr_content.startswith("GZIP1:"):
                b64 = self.qr_content[len("GZIP1:"):]
                gz_bytes = base64.b64decode(b64)
                face_bytes = gzip.decompress(gz_bytes)
            else:
                face_bytes = base64.b64decode(self.qr_content)
            recon_face = Image.open(BytesIO(face_bytes)).convert("RGB")
            qpix = self.pil_to_qpixmap(recon_face)
            if not qpix.isNull():
                scaled = qpix.scaled(self.label_recon_face.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_recon_face.setPixmap(scaled)
                self.label_recon_face.setText("")
            else:
                QMessageBox.warning(self, "显示失败", "重构的人脸无法显示。")
        except Exception as e:
            QMessageBox.critical(self, "重构失败", f"错误：{e}")

    # ---------------------- CNN 识别 ----------------------
    def run_cnn_recognition(self):
        if not self.original_face:
            QMessageBox.warning(self, "提示", "请先选择人脸图像！")
            return
        if not self._face_model_loaded or self.face_infer is None:
            QMessageBox.warning(self, "模型不可用", "人脸识别模型未加载，无法执行识别。")
            return
        try:
            class_name, confidence = self.face_infer.infer(self.original_face)
            if class_name is None:
                class_name = "Unknown"
            try:
                confidence_val = float(confidence)
                if not np.isfinite(confidence_val):
                    confidence_val = 0.0
            except Exception:
                confidence_val = 0.0
            result_text = f"识别类别：{class_name}\n置信度：{confidence_val:.2f}%"
            # 在界面上显示结果并弹窗提示
            self.label_cnn_result.setText("CNN识别结果：\n" + result_text)
            QMessageBox.information(self, "CNN识别结果", result_text)
        except Exception as e:
            print("run_cnn_recognition error:", e)
            QMessageBox.critical(self, "识别失败", f"模型调用错误：{e}\n请检查模型文件与输入图像。")

# ---------------------- 主程序入口 ----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FaceQRSystemWindow()
    window.show()
    sys.exit(app.exec_())
