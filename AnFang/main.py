import sys
import os
import cv2
import numpy as np
import time
import urllib.request
import random
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QListWidget, QComboBox, QSlider, QProgressBar,
                             QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import ssl

# 修复SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ==================== 多镜像源配置 ====================
MIRRORS = [
    'https://mirrors.tuna.tsinghua.edu.cn/pytorch/vision/models/',  # 清华大学镜像源
    'https://mirrors.aliyun.com/pytorch/vision/models/',            # 阿里云镜像源
    'https://mirrors.ustc.edu.cn/pytorch/vision/models/'           # 中国科技大学镜像源
]


def setup_download_environment():
    """配置下载环境"""
    # 1. 设置模型缓存目录
    os.environ['TORCH_HOME'] = './models'

    # 2. 随机选择镜像源
    current_mirror = random.choice(MIRRORS)
    os.environ['TORCHVISION_MODEL_ZOO'] = current_mirror
    print(f"使用镜像源: {current_mirror}")


# 在加载模型前调用
setup_download_environment()

# ==================== 模型定义 ====================
# COCO类别标签
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
    'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

# 我们关心的类别
TARGET_CLASSES = ['person', 'car', 'bus', 'truck']


# ==================== 模型加载工具 ====================
def load_model():
    """加载预训练模型"""
    # 检查是否有可用的GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载轻量级模型
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    model.eval()
    model.to(device)

    print("模型加载完成，使用设备:", device)
    return model, device


# ==================== 核心功能实现 ====================
class SecuritySystem:
    def __init__(self):
        # 加载深度学习模型
        self.detection_model, self.device = load_model()

        # 加载中文字体
        self.chinese_font = self.load_chinese_font()

        # 初始化区域和状态
        self.zone_pts = None
        self.prev_frame = None
        self.alerts = []

        # 添加历史帧缓存减少闪烁
        self.prev_detections = []
        self.detection_history_size = 5  # 缓存帧数

        # 事故检测相关变量
        self.accident_threshold = 0.5  # 事故检测阈值
        self.prev_vehicle_positions = {}  # 存储前一帧车辆位置
        self.accident_frames = 0  # 连续检测到事故的帧数

    def load_chinese_font(self):
        """加载中文字体"""
        try:
            # 尝试加载系统字体
            font_path = "simhei.ttf"
            if not os.path.exists(font_path):
                # 如果系统字体不存在，尝试下载
                download_file("https://github.com/Python3WebSpider/SimHei/raw/master/simhei.ttf", font_path)

            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, 20)
            else:
                print("警告：无法加载中文字体文件，将使用默认字体")
                return None
        except Exception as e:
            print(f"加载中文字体失败: {str(e)}")
            return None

    def draw_chinese_text(self, frame, text, position, color=(255, 255, 255)):
        """在图像上绘制中文文本"""
        if self.chinese_font is None:
            # 如果没有中文字体，使用OpenCV的英文文本
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return frame

        try:
            # 将OpenCV图像转换为PIL图像
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)

            # 绘制中文文本
            draw.text(position, text, font=self.chinese_font, fill=color)

            # 将PIL图像转换回OpenCV格式
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            return frame
        except Exception as e:
            print(f"绘制中文失败: {str(e)}")
            # 失败时使用英文
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            return frame

    def preprocess_image(self, image, target_size=320):
        """预处理图像：转换为tensor并归一化"""
        # 将BGR转换为RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 调整图像大小以加快处理速度
        h, w = image.shape[:2]
        scale = target_size / min(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

        # 转换为PIL图像
        image = Image.fromarray(image)
        # 转换为tensor
        image_tensor = F.to_tensor(image)
        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0)
        return image_tensor.to(self.device), scale

    def detect_objects(self, frame, threshold=0.5):
        """使用深度学习模型检测目标"""
        # 预处理图像
        image_tensor, scale = self.preprocess_image(frame)

        # 进行预测
        with torch.no_grad():
            predictions = self.detection_model(image_tensor)

        # 获取预测结果
        pred_boxes = predictions[0]['boxes'].cpu().numpy() / scale  # 还原到原始尺寸
        pred_labels = predictions[0]['labels'].cpu().numpy()
        pred_scores = predictions[0]['scores'].cpu().numpy()

        # 筛选出置信度高于阈值的目标
        keep = pred_scores > threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        # 当前帧的检测结果
        current_detections = []

        # 绘制检测框
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]

            # 只处理我们关心的类别
            if class_name in TARGET_CLASSES:
                # 设置不同类别的颜色
                if class_name == 'person':
                    color = (255, 0, 0)  # 蓝色代表人
                elif class_name in ['car', 'bus', 'truck']:
                    color = (0, 0, 255)  # 红色代表车辆
                else:
                    color = (0, 255, 0)  # 绿色为其他

                # 转换为整数坐标
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 添加标签和置信度
                text = f"{class_name}: {score:.2f}"
                self.draw_chinese_text(frame, text, (x1, y1 - 30), color)

                # 保存当前检测结果
                current_detections.append({
                    'class': class_name,
                    'box': (x1, y1, x2, y2),
                    'score': score,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })

        return frame, current_detections

    def detect_zone_intrusion(self, frame):
        """区域入侵检测（稳定性优化）"""
        if self.zone_pts is None or len(self.zone_pts) < 3:
            return frame

        # 绘制监控区域
        cv2.polylines(frame, [np.array(self.zone_pts)], True, (0, 165, 255), 2)

        # 运动检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.prev_frame is None:
            self.prev_frame = blurred
            return frame

        # 计算帧差
        diff = cv2.absdiff(self.prev_frame, blurred)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        self.prev_frame = blurred

        # 检测轮廓
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        intrusion = False

        for c in contours:
            if cv2.contourArea(c) < 1000:
                continue

            x, y, w, h = cv2.boundingRect(c)
            center = (x + w // 2, y + h // 2)

            # 检测是否在区域内
            if cv2.pointPolygonTest(np.array(self.zone_pts), center, False) >= 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                intrusion = True

        if intrusion:
            self.draw_chinese_text(frame, "区域入侵!", (30, 60), (0, 0, 255))
            self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "Zone Intrusion"})
        return frame

    def detect_abnormal_behavior(self, frame):
        """异常行为检测（跌倒）"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fall_detected = False

        for c in contours:
            if cv2.contourArea(c) < 2000:
                continue

            x, y, w, h = cv2.boundingRect(c)

            # 通过宽高比检测跌倒
            aspect_ratio = w / float(h)
            if aspect_ratio > 1.8:  # 宽远大于高
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                self.draw_chinese_text(frame, "跌倒检测!", (x, y - 30), (255, 0, 255))
                fall_detected = True

        if fall_detected:
            self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "Fall Detected"})
        return frame

    def detect_accidents(self, frame, detections):
        """车辆事故检测"""
        # 简化版事故检测逻辑（实际应用中应使用训练好的模型）
        accident_detected = False

        # 存储当前帧车辆位置
        current_vehicle_positions = {}

        # 首先收集当前帧的车辆位置
        for det in detections:
            if det['class'] in ['car', 'bus', 'truck']:
                box = det['box']
                center = det['center']
                current_vehicle_positions[id(det)] = {
                    'center': center,
                    'box': box
                }

        # 检查车辆运动状态是否异常（例如突然停止）
        for det_id, current_vehicle in current_vehicle_positions.items():
            current_center = current_vehicle['center']
            current_box = current_vehicle['box']

            # 检查前一帧中是否有相同车辆
            if det_id in self.prev_vehicle_positions:
                prev_center = self.prev_vehicle_positions[det_id]['center']

                # 计算车辆移动距离
                distance = np.sqrt((current_center[0] - prev_center[0]) ** 2 +
                                   (current_center[1] - prev_center[1]) ** 2)

                # 如果两帧之间车辆移动距离很小（突然停止）
                w = current_box[2] - current_box[0]
                h = current_box[3] - current_box[1]

                if distance < max(w, h) * 0.1:
                    # 标记为事故
                    x1, y1, x2, y2 = current_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    self.draw_chinese_text(frame, "事故!", (x1, y1 - 60), (0, 0, 255))
                    accident_detected = True
                    self.accident_frames += 1
                else:
                    self.accident_frames = 0
            else:
                # 新出现的车辆
                self.prev_vehicle_positions[det_id] = {
                    'center': current_center,
                    'box': current_box
                }

        # 更新前一帧车辆位置
        self.prev_vehicle_positions = current_vehicle_positions

        # 如果连续多帧检测到事故，则触发警报
        if self.accident_frames > 5 and accident_detected:
            self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "Vehicle Accident"})
            self.accident_frames = 0  # 重置计数器

        return frame

    def process_frame(self, frame):
        """处理单帧图像并应用所有检测功能"""
        if frame is None:
            return frame

        # 使用深度学习模型检测目标
        frame, detections = self.detect_objects(frame)

        # 添加历史帧检测结果减少闪烁
        self.prev_detections.append(detections)
        if len(self.prev_detections) > self.detection_history_size:
            self.prev_detections.pop(0)

        # 应用其他检测功能
        frame = self.detect_zone_intrusion(frame)
        frame = self.detect_abnormal_behavior(frame)
        frame = self.detect_accidents(frame, detections)

        # 添加时间戳
        height, width, _ = frame.shape
        self.draw_chinese_text(frame, time.strftime("%Y-%m-%d %H:%M:%S"),
                               (10, height - 30), (0, 255, 0))
        return frame


# ==================== 模型下载工具 ====================
def download_file(url, save_path, timeout=120):
    """下载文件并添加国内镜像源"""
    if not os.path.exists(save_path):
        print(f"下载文件: {url}")

        # 国内镜像源列表
        mirrors = [
            "https://ghproxy.com/",  # GitHub代理
            "https://mirror.ghproxy.com/",
            "https://hub.fastgit.org/"
        ]

        for mirror in mirrors:
            try:
                mirror_url = mirror + url.replace("https://", "")
                print(f"尝试镜像源: {mirror_url}")
                urllib.request.urlretrieve(mirror_url, save_path)
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    print(f"通过镜像源下载成功: {mirror}")
                    return True
            except Exception as e:
                print(f"镜像源 {mirror} 失败: {str(e)}")

        # 镜像源都失败时尝试原始链接
        try:
            print("尝试原始链接下载...")
            urllib.request.urlretrieve(url, save_path)
            return True
        except Exception as e:
            print(f"原始链接下载失败: {str(e)}")
            return False
    return True


# ==================== 视频处理线程 ====================
class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    finished = pyqtSignal()
    progress_updated = pyqtSignal(int)

    def __init__(self, security_system):
        super().__init__()
        self.security_system = security_system
        self.video_path = ""
        self.is_running = False
        self.cap = None
        self.frame_count = 0
        self.total_frames = 0
        self.frame_delay = 0  # 帧间延迟（毫秒）
        self.last_frame_time = 0  # 上一帧显示时间

    def run(self):
        self.is_running = True
        self.cap = cv2.VideoCapture(self.video_path)

        # 获取视频属性
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 计算帧间延迟（毫秒）
        self.frame_delay = max(1, int(1000 / fps)) if fps > 0 else 33  # 默认30fps

        print(f"视频帧率: {fps} FPS, 帧延迟: {self.frame_delay}ms")

        self.last_frame_time = time.time() * 1000  # 当前时间(毫秒)

        while self.is_running:
            current_time = time.time() * 1000  # 当前时间(毫秒)

            # 控制帧率：确保帧间隔时间
            if current_time - self.last_frame_time < self.frame_delay:
                QThread.msleep(1)  # 短暂休眠1ms
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            # 处理帧
            processed_frame = self.security_system.process_frame(frame)

            # 转换为RGB格式
            rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            # 发送信号更新UI
            self.change_pixmap.emit(qt_image)

            # 更新计数器和时间戳
            self.frame_count += 1
            self.last_frame_time = current_time

            # 更新进度
            if self.total_frames > 0:
                progress = int((self.frame_count / self.total_frames) * 100)
                self.progress_updated.emit(progress)

        self.cap.release()
        self.finished.emit()

    def stop(self):
        self.is_running = False


# ==================== GUI主界面 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能安防监控系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化安防系统
        self.security_system = SecuritySystem()
        self.video_thread = None

        # 创建UI组件
        self.create_ui()

    def create_ui(self):
        """创建用户界面"""
        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout()

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(QLabel("处理进度:"))
        control_layout.addWidget(self.progress_bar)

        # 帧率显示
        self.fps_label = QLabel("视频帧率: -- FPS")
        control_layout.addWidget(self.fps_label)

        # 视频选择按钮
        self.btn_open = QPushButton("选择监控视频")
        self.btn_open.clicked.connect(self.open_video)
        control_layout.addWidget(self.btn_open)

        # 功能选择
        control_layout.addWidget(QLabel("目标检测:"))
        self.cb_detection = QComboBox()
        self.cb_detection.addItems(["开启", "关闭"])
        self.cb_detection.setCurrentIndex(0)
        control_layout.addWidget(self.cb_detection)

        control_layout.addWidget(QLabel("区域入侵检测:"))
        self.cb_intrusion = QComboBox()
        self.cb_intrusion.addItems(["开启", "关闭"])
        self.cb_intrusion.setCurrentIndex(0)
        control_layout.addWidget(self.cb_intrusion)

        control_layout.addWidget(QLabel("跌倒检测:"))
        self.cb_fall = QComboBox()
        self.cb_fall.addItems(["开启", "关闭"])
        self.cb_fall.setCurrentIndex(0)
        control_layout.addWidget(self.cb_fall)

        control_layout.addWidget(QLabel("事故检测:"))
        self.cb_accident = QComboBox()
        self.cb_accident.addItems(["开启", "关闭"])
        self.cb_accident.setCurrentIndex(0)
        control_layout.addWidget(self.cb_accident)

        # 报警区域
        control_layout.addWidget(QLabel("报警记录:"))
        self.alert_list = QListWidget()
        control_layout.addWidget(self.alert_list)

        # 添加清空按钮
        self.btn_clear = QPushButton("清空报警记录")
        self.btn_clear.clicked.connect(self.clear_alerts)
        control_layout.addWidget(self.btn_clear)

        # 右侧视频显示
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")

        # 添加组件到主布局
        control_panel.setLayout(control_layout)
        control_panel.setMaximumWidth(350)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.video_label)

        # 设置主布局
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def clear_alerts(self):
        """清空报警记录"""
        self.security_system.alerts = []
        self.alert_list.clear()

    def open_video(self):
        """打开视频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择监控视频", "", "视频文件 (*.mp4 *.avi *.mov)")

        if file_path:
            self.security_system.alerts = []
            self.security_system.zone_pts = [(300, 200), (800, 200), (800, 500), (300, 500)]
            self.alert_list.clear()
            self.progress_bar.setValue(0)

            # 重置检测历史
            self.security_system.prev_detections = []
            self.security_system.prev_vehicle_positions = {}

            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait()

            self.video_thread = VideoThread(self.security_system)
            self.video_thread.video_path = file_path
            self.video_thread.change_pixmap.connect(self.update_video)
            self.video_thread.finished.connect(self.video_finished)
            self.video_thread.progress_updated.connect(self.update_progress)
            self.video_thread.start()

    def update_video(self, image):
        """更新视频显示"""
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.KeepAspectRatio))

        # 更新报警列表
        self.alert_list.clear()
        for alert in self.security_system.alerts[-10:]:
            # 将英文报警类型转换为中文
            alert_type = alert['type']
            if alert_type == "Zone Intrusion":
                alert_type = "区域入侵"
            elif alert_type == "Fall Detected":
                alert_type = "跌倒检测"
            elif alert_type == "Vehicle Accident":
                alert_type = "车辆事故"

            self.alert_list.addItem(f"{alert['time']} - {alert_type}")

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def video_finished(self):
        """视频处理完成"""
        self.video_label.clear()
        self.video_label.setText("视频处理完成")

        # 显示完成消息框
        QMessageBox.information(self, "完成", "视频分析完成！", QMessageBox.Ok)

    def closeEvent(self, event):
        """关闭窗口时停止视频处理"""
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        event.accept()


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())