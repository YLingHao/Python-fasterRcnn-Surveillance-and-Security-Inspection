import sys
import os
import cv2
import numpy as np
import time
import urllib.request
import random
import torch
from PIL import Image
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.transforms import functional as F
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QListWidget, QComboBox, QSlider, QProgressBar,
                             QMessageBox, QCheckBox, QLineEdit, QDialog,
                             QGridLayout, QGroupBox, QFormLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import ssl
from collections import deque
from math import sqrt, atan2, degrees

# 修复SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# ==================== 多镜像源配置 ====================
MIRRORS = [
    'https://mirrors.tuna.tsinghua.edu.cn/pytorch/vision/models/',  # 清华大学镜像源
    'https://mirrors.aliyun.com/pytorch/vision/models/',  # 阿里云镜像源
    'https://mirrors.ustc.edu.cn/pytorch/vision/models/'  # 中国科技大学镜像源
]


def setup_download_environment():
    """配置下载环境"""
    os.environ['TORCH_HOME'] = './models'
    current_mirror = random.choice(MIRRORS)
    os.environ['TORCHVISION_MODEL_ZOO'] = current_mirror
    print(f"使用镜像源: {current_mirror}")


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
TARGET_CLASSES = ['person', 'car', 'bus', 'truck', 'bicycle', 'motorcycle']

# 物品类别（用于遗留物品检测）
ITEM_CLASSES = ['backpack', 'handbag', 'suitcase', 'bottle', 'laptop', 'cell phone', 'book', 'umbrella']


# ==================== 模型加载工具 ====================
def load_model():
    """加载预训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
    model.eval()
    model.to(device)
    print("模型加载完成，使用设备:", device)
    return model, device


# ==================== 红外增强功能 ====================
class InfraredEnhancer:
    def __init__(self):
        self.clip_limit = 2.0
        self.tile_size = (8, 8)
        self.thermal_palette = cv2.COLORMAP_JET
        self.min_temp = 20  # 最低温度阈值(℃)
        self.max_temp = 40  # 最高温度阈值(℃)

    def apply_clahe(self, gray_frame):
        """应用CLAHE增强红外图像对比度"""
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_size)
        return clahe.apply(gray_frame)

    def convert_to_thermal(self, gray_frame):
        """将灰度红外图像转换为热成像伪彩色图像"""
        # 归一化温度范围
        normalized = cv2.normalize(gray_frame, None, 0, 255, cv2.NORM_MINMAX)
        # 应用热成像调色板
        thermal = cv2.applyColorMap(normalized, self.thermal_palette)
        return thermal

    def detect_hotspots(self, gray_frame):
        """检测高温区域(潜在人体)"""
        # 应用自适应阈值
        _, thresh = cv2.threshold(gray_frame, 220, 255, cv2.THRESH_BINARY)
        # 形态学操作增强热点
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh

    def enhance_infrared_frame(self, frame):
        """增强红外图像处理流程"""
        # 保存原始尺寸
        original_height, original_width = frame.shape[:2]

        # 转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 应用CLAHE增强对比度
        enhanced = self.apply_clahe(gray)
        # 转换为热成像伪彩色
        thermal = self.convert_to_thermal(enhanced)
        # 检测热点区域
        hotspots = self.detect_hotspots(enhanced)

        # 确保输出尺寸与输入一致
        thermal = cv2.resize(thermal, (original_width, original_height))
        hotspots = cv2.resize(hotspots, (original_width, original_height))

        return thermal, hotspots


# ==================== 图表窗口 ====================
class StatsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("人流统计图表")
        self.setGeometry(200, 200, 800, 600)

        # 创建布局
        layout = QVBoxLayout()

        # 创建图表
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)

        # 将画布添加到布局中
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def update_charts(self, density_history, total_count_history):
        """更新图表数据"""
        self.figure.clear()

        # 确保有数据可绘制
        if len(density_history) == 0 or len(total_count_history) == 0:
            # 显示无数据提示
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, '无可用数据', ha='center', va='center', fontsize=15)
            ax.set_title('数据不足')
            self.canvas.draw()
            return

        # 创建人流密度图表
        ax1 = self.figure.add_subplot(211)
        ax1.plot(density_history, 'b-')
        ax1.set_title('Crowd Density Trend')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Density')
        ax1.grid(True)

        # 创建总人数图表
        ax2 = self.figure.add_subplot(212)
        ax2.plot(total_count_history, 'g-')
        ax2.set_title('Total People Count')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Count')
        ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()


# ==================== 核心功能实现 ====================
class SecuritySystem:
    def __init__(self):
        # 加载深度学习模型
        self.detection_model, self.device = load_model()

        # 新增红外增强器
        self.infrared_enhancer = InfraredEnhancer()
        self.infrared_mode = False  # 红外模式标志

        # 初始化区域和状态
        self.zone_pts = None
        self.prev_frame = None
        self.alerts = []
        self.active_visual_alerts = []  # 用于多帧显示的警报

        # 添加历史帧缓存减少闪烁
        self.prev_detections = deque(maxlen=5)  # 使用deque代替list

        # 事故检测相关变量
        self.vehicle_tracks = {}  # 车辆跟踪信息
        self.track_id_counter = 0  # 跟踪ID计数器
        self.accident_cooldown = {}  # 事故冷却计时器
        self.accident_cooldown_time = 2.0  # 冷却时间（秒）
        self.alert_duration = 14  # 车辆事故警报显示帧数（默认值）
        self.fall_alert_duration = 13  # 跌倒报警显示帧数（默认值）
        # 新增车辆速度阈值设置
        self.vehicle_speed_threshold = 1.0  # 车辆移动速度阈值（像素/帧）
        self.relative_speed_threshold = 18  # 相对速度阈值（像素/帧）

        # 跌倒检测相关变量
        self.fall_history = {}  # 跌倒检测历史记录
        self.fall_cooldown_time = 1.5  # 跌倒报警冷却时间
        self.person_tracks = {}  # 人员跟踪信息

        # 人流统计相关变量
        self.total_person_count = 0  # 总人数
        self.crowd_density = 0.0  # 人流密度
        self.warning_line = 0.1  # 默认人流密度预警线
        self.density_history = deque(maxlen=100)  # 存储历史密度数据
        self.total_count_history = deque(maxlen=100)  # 存储历史总人数数据

        # 遗留物品检测相关变量
        self.abandoned_object_threshold = 2.5  # 物品无人看管的时间阈值（秒），默认2秒
        self.abandoned_object_distance = 240  # 物品与人的最大距离（像素），默认100像素
        self.abandoned_objects = {}  # 跟踪遗留物品
        self.abandoned_id_counter = 0  # 遗留物品ID计数器
        self.abandoned_enabled = False  # 是否启用遗留物品检测
        self.abandoned_alert_duration = 25  # 遗留物品警报显示帧数，默认15帧

        # 打架斗殴检测相关变量
        self.fight_enabled = True  # 默认启用打架检测
        self.fight_params = {
            'min_overlap': 0.4,  # 最小重叠面积比例（相对于较小边界框的面积）
            'max_distance': 210,  # 两个人中心点的最大距离（像素）
            'min_interaction_frames': 14,  # 最小互动帧数
            'min_close_count': 3,  # 最小靠近次数
            'min_speed': 6.0,  # 最小速度（像素/帧）
            'min_aspect_change': 0.5,  # 最小宽高比变化阈值
        }
        self.fight_tracks = {}  # 打架行为跟踪
        self.fight_cooldown = {}  # 打架报警冷却
        self.fight_cooldown_time = 4.0  # 冷却时间（秒）
        self.fight_alert_duration = 25  # 打架警报显示帧数

    def compute_iou(self, box1, box2):
        """计算两个边界框的IoU（交并比）"""
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2

        # 计算交集区域坐标
        xx1 = max(x1, x3)
        yy1 = max(y1, y3)
        xx2 = min(x2, x4)
        yy2 = min(y2, y4)

        # 计算交集区域面积
        inter_area = max(0, xx2 - xx1) * max(0, yy2 - yy1)

        # 计算两个边界框面积
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x4 - x3) * (y4 - y3)

        # 计算并集面积
        union_area = area1 + area2 - inter_area

        # 计算IoU
        iou = inter_area / union_area if union_area > 0 else 0.0
        return iou

    def draw_text_with_background(self, frame, text, position, font_scale=0.8, thickness=2, color=(0, 255, 0),
                                  bg_color=(0, 0, 0)):
        """绘制带背景的文字，防止闪烁"""
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        # 背景矩形
        padding = 5
        cv2.rectangle(frame,
                      (position[0] - padding, position[1] - text_height - padding),
                      (position[0] + text_width + padding, position[1] + padding),
                      bg_color, -1)  # 填充背景

        # 绘制文字
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
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
        # 延迟加载模型
        if not hasattr(self, 'detection_model') or self.detection_model is None:
            self.detection_model, self.device = load_model()
        # 在红外模式下使用专门的目标检测逻辑
        if self.infrared_mode:
            return self.detect_infrared_objects(frame, threshold)

        # 预处理图像
        image_tensor, scale = self.preprocess_image(frame)

        # 进行预测
        with torch.no_grad():
            predictions = self.detection_model(image_tensor)
            # 显存清理
            torch.cuda.empty_cache()

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

        # 重置总人数
        self.total_person_count = 0

        # 绘制检测框
        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            class_name = COCO_INSTANCE_CATEGORY_NAMES[label]

            # 只处理我们关心的类别
            if class_name in TARGET_CLASSES + ITEM_CLASSES:
                # 统计人数：只有类别为'person'才计数
                if class_name == 'person':
                    self.total_person_count += 1

                # 设置不同类别的颜色
                if class_name == 'person':
                    color = (255, 0, 0)  # 蓝色代表人
                elif class_name in ['car', 'bus', 'truck']:
                    color = (0, 0, 255)  # 红色代表车辆
                elif class_name in ['bicycle', 'motorcycle']:
                    color = (0, 165, 255)  # 橙色代表两轮车
                elif class_name in ITEM_CLASSES:
                    color = (0, 255, 255)  # 黄色代表物品
                else:
                    color = (0, 255, 0)  # 绿色为其他

                # 转换为整数坐标
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                # 绘制矩形框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # 添加标签和置信度
                text = f"{class_name}: {score:.2f}"
                cv2.putText(frame, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # 保存当前检测结果
                current_detections.append({
                    'class': class_name,
                    'box': (x1, y1, x2, y2),
                    'score': score,
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                    'aspect_ratio': (x2 - x1) / (y2 - y1)  # 宽高比
                })

        return frame, current_detections

    def detect_infrared_objects(self, frame, threshold=0.5):
        """红外模式下的目标检测"""
        # 转换为HSV色彩空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 根据热成像颜色定义温度范围 (红色表示高温)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 120, 70])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        # 合并红色区域
        mask = cv2.bitwise_or(mask1, mask2)

        # 形态学操作增强目标区域
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检测结果
        current_detections = []

        # 重置总人数
        self.total_person_count = 0

        for contour in contours:
            if cv2.contourArea(contour) > 500:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(contour)

                # 绘制边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                # 添加"热源"标签
                cv2.putText(frame, "Heat Source", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                # 保存检测结果
                current_detections.append({
                    'class': 'heat_source',
                    'box': (x, y, x + w, y + h),
                    'score': 0.9,  # 红外检测置信度较高
                    'center': (x + w // 2, y + h // 2),
                    'aspect_ratio': w / h  # 宽高比
                })

                # 统计人数（假设每个热源是一个人）
                self.total_person_count += 1

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

        # 修复尺寸不匹配问题
        if self.prev_frame is not None:
            if (gray.shape[0] != self.prev_frame.shape[0] or
                    gray.shape[1] != self.prev_frame.shape[1]):
                # 尺寸不匹配，重置前一帧
                self.prev_frame = None
                return frame

        if self.prev_frame is None:
            self.prev_frame = blurred
            return frame

        try:
            # 计算帧差
            diff = cv2.absdiff(self.prev_frame, blurred)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            self.prev_frame = blurred
        except cv2.error as e:
            print(f"OpenCV错误: {str(e)}")
            self.prev_frame = blurred
            return frame

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
            cv2.putText(frame, "区域入侵!", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "区域入侵"})
        return frame

    def detect_abnormal_behavior(self, frame, detections):
        """综合跌倒检测：保留原有方法，整合下滑和蜷缩姿态检测"""
        person_detections = [d for d in detections if d['class'] == 'person']
        fall_detected = False

        for det in person_detections:
            # 获取边界框信息
            x1, y1, x2, y2 = det['box']
            w, h = x2 - x1, y2 - y1
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

            # 创建人员唯一ID
            person_id = f"person_{center_x}_{center_y}"

            if person_id not in self.person_tracks:
                self.person_tracks[person_id] = {
                    'positions': deque(maxlen=15),
                    'heights': deque(maxlen=15),
                    'last_fall_time': 0,
                    'wall_proximity': False,  # 新增：墙体接近标记
                    'slide_history': deque(maxlen=8),  # 新增：滑动历史记录
                    'max_height': 0,  # 新增：最大高度记录
                }

            # 更新最大高度记录
            if h > self.person_tracks[person_id]['max_height']:
                self.person_tracks[person_id]['max_height'] = h

            # 记录当前帧信息
            self.person_tracks[person_id]['positions'].append((center_x, center_y))
            self.person_tracks[person_id]['heights'].append(h)

            # ===== 墙体接近检测 =====
            # 判断人员是否靠近边缘(距离图像边缘<5%视为靠近墙体)
            img_h, img_w = frame.shape[:2]
            is_near_wall = any([
                center_x < img_w * 0.05,
                center_x > img_w * 0.95,
                center_y > img_h * 0.95
            ])
            self.person_tracks[person_id]['wall_proximity'] = is_near_wall

            # ===== 记录滑动向量 =====
            if len(self.person_tracks[person_id]['positions']) > 1:
                last_x, last_y = self.person_tracks[person_id]['positions'][-2]
                slide_vector = (center_x - last_x, center_y - last_y)
                # 只记录显著移动（防止小抖动）
                if abs(slide_vector[0]) > 1 or abs(slide_vector[1]) > 1:
                    self.person_tracks[person_id]['slide_history'].append(slide_vector)

            # ===== 方法1：宽高比检测（保留原有方法） =====
            aspect_fall = (w / h) > 1.4

            # ===== 方法2：垂直变化检测（保留原有方法） =====
            vertical_fall = False
            if len(self.person_tracks[person_id]['positions']) > 2:
                positions = self.person_tracks[person_id]['positions']
                heights = self.person_tracks[person_id]['heights']

                current_y = positions[-1][1]
                prev_y = positions[-2][1]
                vertical_move = current_y - prev_y  # 向下移动为正

                if heights[-2] > 1:  # 有效高度阈值
                    height_drop = (heights[-2] - heights[-1]) / heights[-2]
                else:
                    height_drop = 0

                vertical_fall = vertical_move > 15 and height_drop > 0.25

            # ===== 方法3：骤停+身高降低检测（保留原有方法） =====
            sudden_stop = False
            if len(self.person_tracks[person_id]['positions']) > 4:
                positions = list(self.person_tracks[person_id]['positions'])
                heights = list(self.person_tracks[person_id]['heights'])

                # 1. 检测移动骤停
                moved_recently = any(
                    np.sqrt((positions[i][0] - positions[i - 1][0]) ** 2 +
                            (positions[i][1] - positions[i - 1][1]) ** 2) > 10
                    for i in range(1, len(positions) - 1)
                )

                current_move = np.sqrt((positions[-1][0] - positions[-2][0]) ** 2 +
                                       (positions[-1][1] - positions[-2][1]) ** 2)
                movement_stop = current_move < 5  # 移动几乎停止

                # 2. 检测身高降低
                if heights[-4] > 1:
                    height_drop_sudden = (heights[-4] - heights[-1]) / heights[-4] > 0.2
                else:
                    height_drop_sudden = False

                sudden_stop = moved_recently and movement_stop and height_drop_sudden

            # ===== 方法4：滑动姿态检测（针对墙体下滑场景） =====
            sliding_fall = False
            slide_history = self.person_tracks[person_id]['slide_history']

            if len(slide_history) >= 5:  # 需要足够的帧数分析
                # 1. 验证是连续下滑：最后5帧都是向下移动（Y轴正方向）
                all_down = all(slide[1] > 0 for slide in list(slide_history)[-5:])

                # 2. 分析滑动方向：向下和水平移动比例
                if all_down:
                    avg_vector = (
                        sum(s[0] for s in list(slide_history)[-5:]) / 5,
                        sum(s[1] for s in list(slide_history)[-5:]) / 5
                    )

                    # 计算水平移动与垂直移动的比例（防止水平移动过大）
                    horizontal_ratio = abs(avg_vector[0]) / max(abs(avg_vector[1]), 1e-5)
                    sliding_fall = horizontal_ratio < 0.8  # 垂直移动占主导

            # ===== 方法5：蜷缩姿态检测（针对最终蜷缩场景） =====
            crouching_fall = False
            if len(self.person_tracks[person_id]['heights']) >= 5:
                heights = list(self.person_tracks[person_id]['heights'])

                # 计算最近5帧的高度变化
                current_height = heights[-1]
                if len(heights) > 5:
                    recent_drop = heights[-5] - current_height
                else:
                    recent_drop = heights[0] - current_height

                # 计算高度下降比率（当前高度对比历史最高）
                max_height = self.person_tracks[person_id]['max_height']
                height_drop_ratio = (max_height - current_height) / max_height

                # 蜷缩姿态条件：
                # 1. 靠近墙体（是墙体下滑的最后状态）
                # 2. 当前高度明显低于历史最高（高度下降超过30%）
                # 3. 最近5帧仍有下降趋势
                crouching_fall = (
                        is_near_wall and
                        height_drop_ratio > 0.3 and
                        recent_drop > 0
                )

            # ===== 触发跌倒的条件 =====
            # 原有三种方法+新增两种方法
            if any([aspect_fall, vertical_fall, sudden_stop, sliding_fall, crouching_fall]):
                current_time = time.time()
                if current_time - self.person_tracks[person_id]['last_fall_time'] > self.fall_cooldown_time:
                    # 绘制醒目警报
                    alert_position = (x1, y1 - 50) if y1 > 50 else (x1, 20)
                    self.active_visual_alerts.append({
                        'text': "发生跌倒!",
                        'position': alert_position,
                        'duration': self.fall_alert_duration,
                        'color': (0, 0, 255)
                    })
                    # 用红色框标记跌倒人员
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                    fall_detected = True

        return frame

    def track_objects(self, detections, frame_time):
        """跟踪检测到的物体 - 增强稳定性"""
        updated_tracks = {}

        for det in detections:
            center = det['center']
            box = det['box']
            class_name = det['class']
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]

            # 计算边界框对角线长度作为尺寸度量
            size = sqrt(box_width ** 2 + box_height ** 2)

            # 设置不同类别的跟踪阈值（基于物体大小）
            if class_name in ['person', 'bicycle', 'motorcycle', 'heat_source']:
                tracking_threshold = size * 0.6  # 小物体较高阈值
            else:
                tracking_threshold = size * 0.4  # 车辆较低阈值

            # 尝试匹配现有轨迹
            matched = False
            best_match_id = None
            best_match_distance = float('inf')

            for track_id, track_info in self.vehicle_tracks.items():
                if track_info['class'] != class_name:
                    continue

                last_center = track_info['last_center']
                last_time = track_info['last_time']

                # 计算距离
                distance = sqrt((center[0] - last_center[0]) ** 2 +
                                (center[1] - last_center[1]) ** 2)

                # 时间差（秒）
                time_diff = frame_time - last_time

                # 匹配条件：距离小于阈值
                if distance < tracking_threshold and distance < best_match_distance:
                    best_match_distance = distance
                    best_match_id = track_id

            if best_match_id is not None:
                # 更新轨迹
                track_info = self.vehicle_tracks[best_match_id]
                track_info['last_center'] = center
                track_info['last_time'] = frame_time
                track_info['speed'] = best_match_distance / time_diff if time_diff > 0 else 0
                track_info['positions'].append(center)
                track_info['box'] = box  # 更新边界框

                # 记录长宽比变化
                aspect_ratio = det.get('aspect_ratio', box_width / box_height)
                if 'aspect_ratios' not in track_info:
                    track_info['aspect_ratios'] = deque(maxlen=10)
                track_info['aspect_ratios'].append(aspect_ratio)

                # 记录高度变化
                if 'heights' not in track_info:
                    track_info['heights'] = deque(maxlen=10)
                track_info['heights'].append(box_height)

                # 保留最近10个位置（使用deque自动处理）
                # 不再需要手动popleft()

                updated_tracks[best_match_id] = track_info
                matched = True

            # 如果没有匹配，创建新轨迹
            if not matched:
                track_id = self.track_id_counter
                self.track_id_counter += 1

                aspect_ratio = det.get('aspect_ratio', box_width / box_height)
                updated_tracks[track_id] = {
                    'class': class_name,
                    'last_center': center,
                    'last_time': float(frame_time),  # 确保是浮点数
                    'speed': 0,
                    'positions': deque([center], maxlen=10),  # 使用deque
                    'box': box,
                    'aspect_ratios': deque([aspect_ratio], maxlen=10),  # 使用deque
                    'heights': deque([float(box_height)], maxlen=10),  # 使用deque，并转换为浮点数
                    'stable_count': 0,  # 新增：稳定计数
                    'accident_state': False,  # 新增：事故状态
                    'last_alert_time': 0.0  # 新增：最后报警时间，初始化为0.0
                }

        # 更新轨迹字典
        self.vehicle_tracks = updated_tracks

        return self.vehicle_tracks

    def detect_accidents(self, frame, frame_time):
        """优化后的车辆碰撞检测算法 - 增加速度阈值判断"""
        accident_detected = False
        vehicle_tracks = [t for t in self.vehicle_tracks.values() if t['class'] in ['car', 'bus', 'truck']]
        non_vehicle_tracks = [t for t in self.vehicle_tracks.values() if
                              t['class'] in ['person', 'bicycle', 'motorcycle']]

        # 1. 单车辆异常检测：急转弯或极刹车
        for track in vehicle_tracks:
            # 只有位置历史足够时才能判断
            if len(track['positions']) < 3:
                continue

            positions = list(track['positions'])

            # 计算最近3帧的运动向量
            vector1 = (positions[-1][0] - positions[-2][0], positions[-1][1] - positions[-2][1])
            vector2 = (positions[-2][0] - positions[-3][0], positions[-2][1] - positions[-3][1])

            # 计算角度变化（度）
            dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            magnitude1 = max(1e-5, sqrt(vector1[0] ** 2 + vector1[1] ** 2))  # 防零值
            magnitude2 = max(1e-5, sqrt(vector2[0] ** 2 + vector2[1] ** 2))

            # 修复1：限制cos_angle范围
            cos_angle = dot_product / (magnitude1 * magnitude2)
            cos_angle = max(-1.0, min(1.0, cos_angle))

            # 修复2：安全计算角度变化
            angle_change = degrees(atan2(sqrt(max(0, 1 - cos_angle ** 2)), cos_angle))

            # 急转弯检测（角度变化>60度且速度突然降低）
            speed_change = (magnitude1 / magnitude2) if magnitude2 > 0 else 0

            if angle_change > 60 and speed_change < 0.5:
                # 必须连续2帧满足条件才报警（减少误报）
                track['stable_count'] = track.get('stable_count', 0) + 1
                # 确保last_alert_time是数值
                last_alert = track.get('last_alert_time', 0.0)
                if isinstance(last_alert, dict):
                    last_alert = last_alert.get('time', 0.0)
                if track['stable_count'] >= 2 and (frame_time - last_alert) > self.accident_cooldown_time:
                    x1, y1, x2, y2 = track['box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 3)  # 橙色标记

                    # 添加醒目的事故警报（显示多帧）
                    self.active_visual_alerts.append({
                        'text': "急转弯!",
                        'position': (x1, y1 - 50),
                        'duration': self.alert_duration,
                        'color': (0, 165, 255)
                    })

                    self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "车辆急转弯"})
                    track['last_alert_time'] = frame_time
                    accident_detected = True

            # 重置稳定计数
            else:
                track['stable_count'] = 0

        # 2. 车辆间碰撞检测
        for i, track1 in enumerate(vehicle_tracks):
            for j in range(i + 1, len(vehicle_tracks)):
                track2 = vehicle_tracks[j]

                # 生成唯一pair_key
                # 使用字符串表示位置和类别来生成唯一键
                pair_key = tuple(sorted([
                    f"{track1['class']}_{track1['last_center'][0]}_{track1['last_center'][1]}",
                    f"{track2['class']}_{track2['last_center'][0]}_{track2['last_center'][1]}"
                ]))

                # 检查冷却期
                if pair_key in self.accident_cooldown:
                    cool_down_value = self.accident_cooldown[pair_key]
                    # 确保比较的是数值类型
                    if isinstance(cool_down_value, dict):
                        cool_down_value = cool_down_value.get('time', 0)
                    if frame_time - cool_down_value < self.accident_cooldown_time:
                        continue

                # 计算中心距离
                center1 = track1['last_center']
                center2 = track2['last_center']
                distance = sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

                # 计算碰撞危险指数
                collision_risk = 0
                relative_speed = 0

                # 只有轨迹足够长时才能计算相对速度
                if len(track1['positions']) > 1 and len(track2['positions']) > 1:
                    # 使用list转换deque
                    track1_positions = list(track1['positions'])
                    track2_positions = list(track2['positions'])

                    v1 = (track1_positions[-1][0] - track1_positions[-2][0],
                          track1_positions[-1][1] - track1_positions[-2][1])
                    v2 = (track2_positions[-1][0] - track2_positions[-2][0],
                          track2_positions[-1][1] - track2_positions[-2][1])

                    # 计算相对速度向量
                    relative_vector = (v1[0] - v2[0], v1[1] - v2[1])

                    # 计算方向夹角
                    direction_vector = (center2[0] - center1[0], center2[1] - center1[1])
                    dot_product = relative_vector[0] * direction_vector[0] + relative_vector[1] * direction_vector[1]
                    magnitudes = sqrt(relative_vector[0] ** 2 + relative_vector[1] ** 2) * sqrt(
                        direction_vector[0] ** 2 + direction_vector[1] ** 2)
                    if magnitudes > 0:
                        cos_angle = dot_product / magnitudes
                        # 角度越小说明越接近对向运动
                        if cos_angle > 0.7:  # 夹角小于45度
                            relative_speed = min(sqrt(relative_vector[0] ** 2 + relative_vector[1] ** 2), 50)  # 限制最大速度
                            collision_risk = (relative_speed * 10) / max(distance, 1)  # 速度高且距离近则危险

                # 如果有碰撞风险
                if collision_risk > 1.0 or (distance < 80 and relative_speed > 15):
                    # 标记车辆
                    x1, y1, x2, y2 = track1['box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    x1, y1, x2, y2 = track2['box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # 在中间位置绘制警告标志
                    mid_x = (center1[0] + center2[0]) // 2
                    mid_y = (center1[1] + center2[1]) // 2
                    cv2.circle(frame, (mid_x, mid_y), 30, (0, 0, 255), -1)
                    cv2.putText(frame, "!", (mid_x - 10, mid_y + 10), cv2.FONT_HERSHEY_SIMPLEX,
                                1.2, (255, 255, 255), 3)

                    # 添加醒目的事故警报
                    self.active_visual_alerts.append({
                        'text': "车辆碰撞!",
                        'position': (mid_x - 100, mid_y - 50),
                        'duration': self.alert_duration,
                        'color': (0, 0, 255)
                    })

                    self.accident_cooldown[pair_key] = frame_time  # 存储数值时间戳
                    self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "车辆碰撞预警"})
                    accident_detected = True

        # 3. 车辆与非机动物体碰撞检测（添加速度阈值判断）
        for vehicle_track in vehicle_tracks:
            # 检查车辆速度是否达到阈值
            if vehicle_track.get('speed', 0) < self.vehicle_speed_threshold:
                continue  # 车辆速度低于阈值，不检测碰撞

            for non_vehicle_track in non_vehicle_tracks:
                # 计算中心距离
                vehicle_center = vehicle_track['last_center']
                non_vehicle_center = non_vehicle_track['last_center']
                distance = sqrt((vehicle_center[0] - non_vehicle_center[0]) ** 2 +
                                (vehicle_center[1] - non_vehicle_center[1]) ** 2)

                # 车辆尺寸
                vw = vehicle_track['box'][2] - vehicle_track['box'][0]
                vh = vehicle_track['box'][3] - vehicle_track['box'][1]
                vehicle_size = max(vw, vh)

                # 危险距离阈值（车辆尺寸的一半）
                danger_threshold = vehicle_size * 0.5

                # 计算接近速度
                relative_speed = 0
                if len(vehicle_track['positions']) > 1 and len(non_vehicle_track['positions']) > 1:
                    # 转换deque为list
                    vehicle_positions = list(vehicle_track['positions'])
                    non_vehicle_positions = list(non_vehicle_track['positions'])

                    # 车辆最后两帧移动向量
                    v_vector = (vehicle_positions[-1][0] - vehicle_positions[-2][0],
                                vehicle_positions[-1][1] - vehicle_positions[-2][1])
                    # 非机动车最后两帧移动向量
                    nv_vector = (non_vehicle_positions[-1][0] - non_vehicle_positions[-2][0],
                                 non_vehicle_positions[-1][1] - non_vehicle_positions[-2][1])

                    # 相对速度向量
                    relative_vector = (v_vector[0] - nv_vector[0], v_vector[1] - nv_vector[1])
                    relative_speed = sqrt(relative_vector[0] ** 2 + relative_vector[1] ** 2)

                # 碰撞检测条件：距离过近且相对速度高
                if distance < danger_threshold and relative_speed > self.relative_speed_threshold:
                    # 标记车辆和非机动车
                    x1, y1, x2, y2 = vehicle_track['box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    x1, y1, x2, y2 = non_vehicle_track['box']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # 添加醒目的事故警报
                    self.active_visual_alerts.append({
                        'text': "人车碰撞!",
                        'position': (non_vehicle_center[0] - 100, non_vehicle_center[1]),
                        'duration': self.alert_duration,
                        'color': (0, 0, 255)
                    })

                    self.alerts.append({"time": time.strftime("%H:%M:%S"), "type": "行人碰撞预警"})
                    accident_detected = True

        return frame


    def detect_abandoned_objects(self, frame, detections, frame_time):
        """智能遗留物品检测算法

        基于以下逻辑：
        1. 只检测物品类别（ITEM_CLASSES）
        2. 排除人、车辆和热源
        3. 判断物品周围一定距离内是否有人
        4. 如果没有人且超过时间阈值，则判定为遗留物品
        """
        if not self.abandoned_enabled:
            return frame

        # 只检测物品类别
        item_detections = [d for d in detections if d['class'] in ITEM_CLASSES]
        person_detections = [d for d in detections if d['class'] in ['person', 'heat_source']]

        # 更新遗留物品跟踪
        updated_objects = {}
        for item in item_detections:
            item_center = item['center']
            matched = False

            # 尝试匹配已有遗留物品
            for obj_id, obj_info in self.abandoned_objects.items():
                last_center = obj_info['center']
                distance = sqrt((item_center[0] - last_center[0]) ** 2 + (item_center[1] - last_center[1]) ** 2)

                # 如果距离小于阈值，认为是同一物体
                if distance < self.abandoned_object_distance * 0.5:
                    # 更新物体信息
                    obj_info['center'] = item_center
                    obj_info['last_seen'] = frame_time
                    obj_info['duration'] = frame_time - obj_info['first_seen']
                    obj_info['class'] = item['class']
                    obj_info['box'] = item['box']

                    # 检查是否有人靠近
                    obj_info['has_nearby_person'] = False
                    for person in person_detections:
                        person_center = person['center']
                        person_distance = sqrt((item_center[0] - person_center[0]) ** 2 +
                                               (item_center[1] - person_center[1]) ** 2)

                        # 如果有人在一定距离内
                        if person_distance < self.abandoned_object_distance:
                            obj_info['has_nearby_person'] = True
                            obj_info['last_person_time'] = frame_time
                            break

                    updated_objects[obj_id] = obj_info
                    matched = True
                    break

            # 如果没有匹配，添加新物品
            if not matched:
                new_id = self.abandoned_id_counter
                self.abandoned_id_counter += 1

                # 初始状态：检查是否有人靠近
                has_nearby_person = False
                for person in person_detections:
                    person_center = person['center']
                    person_distance = sqrt((item_center[0] - person_center[0]) ** 2 +
                                           (item_center[1] - person_center[1]) ** 2)
                    if person_distance < self.abandoned_object_distance:
                        has_nearby_person = True
                        break

                updated_objects[new_id] = {
                    'id': new_id,
                    'center': item_center,
                    'first_seen': frame_time,
                    'last_seen': frame_time,
                    'duration': 0.0,
                    'class': item['class'],
                    'box': item['box'],
                    'has_nearby_person': has_nearby_person,
                    'last_person_time': frame_time if has_nearby_person else 0,
                    'alert_start_time': 0  # 新增：报警开始时间
                }

        # 更新遗留物品字典
        self.abandoned_objects = updated_objects

        # 检测遗留物品并报警
        for obj_id, obj_info in self.abandoned_objects.items():
            # 计算无人看管的时间
            unattended_time = frame_time - obj_info['last_person_time']

            # 如果物品无人看管时间超过阈值
            if unattended_time > self.abandoned_object_threshold:
                # 绘制遗留物品边界框
                x1, y1, x2, y2 = obj_info['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 黄色框

                # 触发报警（避免重复报警）
                if not obj_info.get('alert_triggered', False):
                    # 标记已报警并记录报警开始时间
                    obj_info['alert_triggered'] = True
                    obj_info['alert_start_time'] = frame_time  # 记录报警开始时间

                    self.alerts.append({
                        "time": time.strftime("%H:%M:%S"),
                        "type": f"遗留物品: {obj_info['class']}"
                    })

                    # 添加视觉警报
                    self.active_visual_alerts.append({
                        'text': "物品遗留!",
                        'position': (x1, y1 - 60),
                        'duration': self.abandoned_alert_duration,
                        'color': (0, 255, 255)
                    })

                # 只在报警后开始计时显示
                if obj_info.get('alert_triggered', False):
                    # 计算报警持续时间（从报警开始到当前时间）
                    alert_duration = frame_time - obj_info['alert_start_time']

                    # 显示报警持续时间
                    text = f"Left item: {alert_duration:.1f}s"
                    self.draw_text_with_background(frame, text, (x1, y1 - 30),
                                                   font_scale=0.7, color=(0, 255, 255))

        return frame

    def detect_fighting(self, frame, detections, frame_time):
        """优化打架斗殴检测算法（排除车辆，只检测人与人之间）"""
        if not self.fight_enabled:
            return frame

        # 只处理人的检测结果
        person_detections = [d for d in detections if d['class'] == 'person']
        num_persons = len(person_detections)

        # 如果没有至少两个人，直接返回
        if num_persons < 2:
            return frame

        # 更新打架行为跟踪
        updated_fights = {}

        for i in range(num_persons):
            for j in range(i + 1, num_persons):
                det1 = person_detections[i]
                det2 = person_detections[j]

                # 获取边界框和中心点
                box1 = det1['box']
                box2 = det2['box']
                center1 = det1['center']
                center2 = det2['center']

                # 计算中心点距离
                distance = sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

                # 如果距离过大，跳过检测
                if distance > self.fight_params['max_distance']:
                    continue

                # 计算边界框的交集面积
                x1_inter = max(box1[0], box2[0])
                y1_inter = max(box1[1], box2[1])
                x2_inter = min(box1[2], box2[2])
                y2_inter = min(box1[3], box2[3])

                inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
                min_area = min((box1[2] - box1[0]) * (box1[3] - box1[1]),
                               (box2[2] - box2[0]) * (box2[3] - box2[1]))

                # 计算重叠比例
                overlap_ratio = inter_area / min_area if min_area > 0 else 0

                # 生成唯一键（避免重复）
                pair_key = tuple(sorted([i, j]))

                # 初始化或更新跟踪状态
                if pair_key not in self.fight_tracks:
                    self.fight_tracks[pair_key] = {
                        'start_time': frame_time,
                        'last_interaction': frame_time,
                        'close_count': 0,
                        'interaction_frames': 0,
                        'max_overlap': 0.0,
                        'state': 'separate',  # 状态：close（靠近）或 separate（分开）
                        'aspect_change': 0.0,  # 宽高比变化
                        'last_aspect1': det1['aspect_ratio'],
                        'last_aspect2': det2['aspect_ratio'],
                        'speed': 0.0  # 互动速度
                    }

                pair_state = self.fight_tracks[pair_key]

                # 更新互动状态
                pair_state['last_interaction'] = frame_time

                # 计算宽高比变化（用于检测大幅动作）
                aspect_change1 = abs(det1['aspect_ratio'] - pair_state['last_aspect1'])
                aspect_change2 = abs(det2['aspect_ratio'] - pair_state['last_aspect2'])
                aspect_change = max(aspect_change1, aspect_change2)
                pair_state['aspect_change'] = aspect_change

                # 更新宽高比记录
                pair_state['last_aspect1'] = det1['aspect_ratio']
                pair_state['last_aspect2'] = det2['aspect_ratio']

                # 计算互动速度（中心点距离变化）
                if 'last_distance' in pair_state:
                    distance_change = abs(distance - pair_state['last_distance'])
                    time_diff = frame_time - pair_state['last_time']
                    if time_diff > 0:
                        pair_state['speed'] = distance_change / time_diff
                pair_state['last_distance'] = distance
                pair_state['last_time'] = frame_time

                # 判断是否满足靠近条件
                if overlap_ratio > self.fight_params['min_overlap']:
                    # 更新最大重叠
                    if overlap_ratio > pair_state['max_overlap']:
                        pair_state['max_overlap'] = overlap_ratio

                    # 更新状态
                    if pair_state['state'] == 'separate':
                        pair_state['close_count'] += 1  # 靠近次数增加
                        pair_state['state'] = 'close'

                    # 增加互动帧数
                    pair_state['interaction_frames'] += 1

                    # 检查是否满足打架条件
                    if (pair_state['interaction_frames'] >= self.fight_params['min_interaction_frames'] and
                            pair_state['close_count'] >= self.fight_params['min_close_count'] and
                            pair_state['speed'] > self.fight_params['min_speed'] and
                            pair_state['aspect_change'] > self.fight_params['min_aspect_change']):

                        # 检查冷却时间
                        if pair_key in self.fight_cooldown:
                            if frame_time - self.fight_cooldown[pair_key] < self.fight_cooldown_time:
                                continue

                        # 触发打架报警
                        self.fight_cooldown[pair_key] = frame_time

                        # 添加报警记录
                        self.alerts.append({
                            "time": time.strftime("%H:%M:%S"),
                            "type": "打架斗殴"
                        })

                        # 添加视觉警报
                        mid_x = (center1[0] + center2[0]) // 2
                        mid_y = (center1[1] + center2[1]) // 2
                        self.active_visual_alerts.append({
                            'text': "打架斗殴!",
                            'position': (mid_x - 100, mid_y - 50),
                            'duration': self.fight_alert_duration,
                            'color': (0, 0, 255)
                        })

                        # 绘制打架框
                        cv2.rectangle(frame, (box1[0], box1[1]), (box1[2], box1[3]), (0, 0, 255), 3)
                        cv2.rectangle(frame, (box2[0], box2[1]), (box2[2], box2[3]), (0, 0, 255), 3)

                        # 绘制连线
                        cv2.line(frame, center1, center2, (0, 0, 255), 2)

                        # 绘制警告标志
                        cv2.putText(frame, "FIGHT", (mid_x - 30, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                else:
                    # 如果重叠比例小于阈值，更新为分开状态
                    pair_state['state'] = 'separate'

                updated_fights[pair_key] = pair_state

        # 清理长时间无互动的跟踪
        current_time = time.time()
        for pair_key in list(self.fight_tracks.keys()):
            if current_time - self.fight_tracks[pair_key]['last_interaction'] > self.fight_cooldown_time * 2:
                del self.fight_tracks[pair_key]

        return frame

    def process_frame(self, frame):
        """处理单帧图像并应用所有检测功能"""
        if frame is None:
            return frame

        # 帧计数器
        if hasattr(self, 'frame_counter'):
            self.frame_counter += 1
        else:
            self.frame_counter = 0

        # 新增：红外模式处理
        if self.infrared_mode:
            # 增强红外图像
            thermal_frame, hotspots = self.infrared_enhancer.enhance_infrared_frame(frame)
            # 使用增强后的热成像图像进行后续处理
            frame = thermal_frame

        # 获取当前时间
        current_time = time.time()

        # 使用深度学习模型检测目标
        frame, detections = self.detect_objects(frame)

        # 计算人流密度（人数/图像面积）
        height, width = frame.shape[:2]
        total_pixels = width * height
        self.crowd_density = self.total_person_count / (total_pixels / 10000)  # 每10000像素的人数

        # 保存历史数据
        self.density_history.append(self.crowd_density)
        self.total_count_history.append(self.total_person_count)

        # 添加历史帧检测结果减少闪烁（使用deque）
        self.prev_detections.append(detections)

        # 跟踪物体
        self.track_objects(detections, current_time)

        # 应用其他检测功能
        frame = self.detect_zone_intrusion(frame)
        frame = self.detect_abnormal_behavior(frame, detections)
        frame = self.detect_accidents(frame, current_time)

        # 使用新的打架检测算法
        frame = self.detect_fighting(frame, detections, current_time)

        # 新增：遗留物品检测
        frame = self.detect_abandoned_objects(frame, detections, current_time)

        # 添加时间戳（带背景）
        height, width, _ = frame.shape
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.draw_text_with_background(frame, timestamp, (10, height - 30),
                                       color=(0, 255, 0), bg_color=(0, 0, 0))

        # 添加人流统计信息（带背景）
        total_text = f"总人数: {self.total_person_count}"
        self.draw_text_with_background(frame, total_text, (10, 120),
                                       color=(0, 255, 0), bg_color=(0, 0, 0))

        density_text = f"人流密度: {self.crowd_density:.2f}"
        self.draw_text_with_background(frame, density_text, (10, 150),
                                       color=(0, 255, 0), bg_color=(0, 0, 0))

        # 添加预警线信息（带背景）
        warning_text = f"人流密度预警线: {self.warning_line:.2f}"
        self.draw_text_with_background(frame, warning_text, (width - 250, 30),
                                       color=(0, 0, 255), bg_color=(0, 0, 0))

        # 检查是否超过预警线
        if self.crowd_density > self.warning_line:
            warning_alert = "人流过密!"
            self.draw_text_with_background(frame, warning_alert, (width - 250, 60),
                                           color=(0, 0, 255), bg_color=(0, 0, 0))
            self.alerts.append({
                "time": time.strftime("%H:%M:%S"),
                "type": f"人流密度预警 ({self.crowd_density:.2f} > {self.warning_line:.2f})"
            })

        # 处理活动的视觉警报
        for i in range(len(self.active_visual_alerts) - 1, -1, -1):
            alert = self.active_visual_alerts[i]
            # 使用draw_text_with_background方法绘制警告文字
            self.draw_text_with_background(frame, alert['text'], alert['position'],
                                           font_scale=1.2, thickness=3, color=alert['color'])
            alert['duration'] -= 1

            # 移除过期的警报
            if alert['duration'] <= 0:
                self.active_visual_alerts.pop(i)

        return frame


# ==================== 视频处理线程 ====================
class VideoThread(QThread):
    change_pixmap = pyqtSignal(QImage)
    finished = pyqtSignal()
    progress_updated = pyqtSignal(int)
    fps_updated = pyqtSignal(float)  # 新增：FPS更新信号

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
        self.current_frame = None
        self.fps = 0  # 存储视频帧率
        self.playback_speed = 1.0  # 倍速播放因子（默认为1倍速）

    def run(self):
        self.is_running = True
        try:
            # 确保模型在设备之间传输时正确处理
            if self.security_system.detection_model is None:
                self.security_system.detection_model, self.security_system.device = load_model()

            # 将模型设置为评估模式
            self.security_system.detection_model.eval()

            # 延迟加载模型到设备
            if next(self.security_system.detection_model.parameters()).device != self.security_system.device:
                self.security_system.detection_model.to(self.security_system.device)
                print(f"已将模型传输到设备: {self.security_system.device}")

            # 释放未使用的内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("已释放GPU内存")

            # 打开视频文件
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                print(f"错误：无法打开视频 {self.video_path}")
                self.finished.emit()
                return

            # 获取视频属性
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 发送FPS信号
            self.fps_updated.emit(self.fps)

            # 计算帧间延迟（毫秒）
            base_frame_delay = 1000.0 / self.fps if self.fps > 0 else 33.0  # 默认30fps
            self.frame_delay = base_frame_delay / self.playback_speed

            print(f"视频帧率: {self.fps} FPS, 倍速: {self.playback_speed}x, 帧延迟: {self.frame_delay:.1f}ms")

            self.last_frame_time = time.time() * 1000  # 当前时间(毫秒)

            # 跳帧计数器
            frame_skip_counter = 0
            skip_frames = 0  # 根据速度计算要跳过的帧数

            while self.is_running:
                current_time = time.time() * 1000  # 当前时间(毫秒)

                # 计算实际应该跳过的帧数
                if self.playback_speed > 1.0:
                    skip_frames = int(self.playback_speed) - 1
                else:
                    skip_frames = 0

                # 控制帧率：确保帧间隔正确
                if current_time - self.last_frame_time < self.frame_delay:
                    QThread.msleep(1)  # 短暂休眠1ms
                    continue

                # 如果需要跳过帧
                if skip_frames > 0 and frame_skip_counter < skip_frames:
                    # 抓取但不处理这一帧
                    _ = self.cap.grab()
                    frame_skip_counter += 1
                    continue

                # 重置跳帧计数器
                frame_skip_counter = 0

                ret, frame = self.cap.read()
                if not ret:
                    break

                # 处理帧
                try:
                    processed_frame = self.security_system.process_frame(frame)
                    self.current_frame = processed_frame
                except Exception as e:
                    print(f"处理帧时出错: {str(e)}")
                    processed_frame = frame

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

                # 更新进度 - 考虑加速播放时实际处理的帧更少
                if self.total_frames > 0:
                    actual_frame = int(self.frame_count * self.playback_speed)
                    progress = min(100, int((actual_frame / self.total_frames) * 100))
                    self.progress_updated.emit(progress)

        except Exception as e:
            print(f"视频处理线程发生错误: {str(e)}")
            # 清理资源
            if self.cap:
                self.cap.release()
            # 处理内存错误
            if "CUDA" in str(e):
                print("CUDA内存不足，尝试回退到CPU")
                # 回退到CPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.security_system.device = torch.device('cpu')
                if self.security_system.detection_model:
                    self.security_system.detection_model.to(self.security_system.device)
                self.run()  # 重新尝试
            else:
                # 显示错误消息
                error_msg = QMessageBox()
                error_msg.setIcon(QMessageBox.Critical)
                error_msg.setText("视频处理错误")
                error_msg.setInformativeText(str(e))
                error_msg.setWindowTitle("错误")
                error_msg.exec_()
        finally:
            if self.cap:
                self.cap.release()
            self.finished.emit()

    def stop(self):
        self.is_running = False
        self.wait(500)  # 等待线程结束，最多500ms

    def set_playback_speed(self, speed_factor):
        """设置倍速播放因子"""
        self.playback_speed = speed_factor
        if self.cap and self.fps > 0:
            base_frame_delay = 1000.0 / self.fps
            self.frame_delay = base_frame_delay / self.playback_speed
            print(f"已设置倍速为: {self.playback_speed}x, 帧延迟: {self.frame_delay:.1f}ms")


# ==================== GUI主界面 ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能安防监控系统")
        self.setGeometry(100, 100, 1200, 800)

        # 初始化安防系统
        self.security_system = SecuritySystem()
        self.video_thread = None
        self.stats_dialog = None  # 统计图表对话框

        # 创建UI组件
        self.create_ui()

    def create_ui(self):
        """创建用户界面"""
        # 主布局
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        # 左侧控制面板
        control_panel = QWidget()
        control_layout = QGridLayout()  # 改为网格布局实现两列

        # 添加进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        control_layout.addWidget(QLabel("处理进度:"), 0, 0)
        control_layout.addWidget(self.progress_bar, 0, 1, 1, 2)

        # 帧率显示
        self.fps_label = QLabel("视频帧率: -- FPS")
        control_layout.addWidget(self.fps_label, 1, 0, 1, 3)

        # 倍速控制
        self.speed_label = QLabel("播放倍速:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x 慢速", "1.0x 正常", "2.0x 快速", "4.0x 极速"])
        self.speed_combo.setCurrentIndex(1)  # 默认正常速度
        self.speed_combo.currentIndexChanged.connect(self.change_speed)
        control_layout.addWidget(self.speed_label, 2, 0)
        control_layout.addWidget(self.speed_combo, 2, 1, 1, 2)

        # 视频选择按钮
        self.btn_open = QPushButton("选择监控视频")
        self.btn_open.clicked.connect(self.open_video)
        control_layout.addWidget(self.btn_open, 3, 0, 1, 3)

        # ========== 检测功能设置组 ==========
        detection_group = QGroupBox("检测功能设置")
        detection_layout = QGridLayout()

        # 第一列
        self.cb_fall = QCheckBox("跌倒检测")
        self.cb_fall.setChecked(True)
        self.cb_fall.stateChanged.connect(self.toggle_fall_detection)
        detection_layout.addWidget(self.cb_fall, 0, 0)

        self.cb_accident = QCheckBox("事故检测")
        self.cb_accident.setChecked(True)
        self.cb_accident.stateChanged.connect(self.toggle_accident_detection)
        detection_layout.addWidget(self.cb_accident, 1, 0)

        self.cb_abandoned = QCheckBox("物品遗留检测")
        self.cb_abandoned.stateChanged.connect(self.toggle_abandoned_detection)
        detection_layout.addWidget(self.cb_abandoned, 2, 0)

        # 第二列
        self.cb_infrared = QCheckBox("红外夜视模式")
        self.cb_infrared.stateChanged.connect(self.toggle_infrared_mode)
        detection_layout.addWidget(self.cb_infrared, 0, 1)

        self.cb_intrusion = QCheckBox("区域入侵检测")
        self.cb_intrusion.setChecked(True)
        self.cb_intrusion.stateChanged.connect(self.toggle_intrusion_detection)
        detection_layout.addWidget(self.cb_intrusion, 1, 1)

        self.cb_fight = QCheckBox("打架斗殴检测")
        self.cb_fight.setChecked(True)  # 默认勾选
        self.cb_fight.stateChanged.connect(self.toggle_fight_detection)
        detection_layout.addWidget(self.cb_fight, 2, 1)

        detection_group.setLayout(detection_layout)
        control_layout.addWidget(detection_group, 4, 0, 1, 2)

        # ========== 遗留物品检测设置组 ==========
        abandoned_group = QGroupBox("遗留物品检测设置")
        abandoned_layout = QFormLayout()

        # 无人看管时间阈值
        self.abandoned_time_label = QLabel("无人看管时间阈值(秒):")
        self.abandoned_time_input = QLineEdit("2.5")
        self.abandoned_time_input.textChanged.connect(self.update_abandoned_time)
        abandoned_layout.addRow(self.abandoned_time_label, self.abandoned_time_input)

        # 物品与人距离阈值
        self.abandoned_distance_label = QLabel("物品与人距离阈值(像素):")
        self.abandoned_distance_input = QLineEdit("240")
        self.abandoned_distance_input.textChanged.connect(self.update_abandoned_distance)
        abandoned_layout.addRow(self.abandoned_distance_label, self.abandoned_distance_input)

        abandoned_group.setLayout(abandoned_layout)
        control_layout.addWidget(abandoned_group, 5, 0, 1, 2)

        # ========== 车辆碰撞检测设置组 ==========
        vehicle_group = QGroupBox("车辆碰撞检测设置")
        vehicle_layout = QFormLayout()

        # 车辆速度阈值设置
        self.vehicle_speed_label = QLabel("车辆移动速度阈值(像素/帧):")
        self.vehicle_speed_input = QLineEdit("1.0")
        self.vehicle_speed_input.textChanged.connect(self.update_vehicle_speed_threshold)
        vehicle_layout.addRow(self.vehicle_speed_label, self.vehicle_speed_input)

        # 相对速度阈值设置
        self.relative_speed_label = QLabel("相对速度阈值(像素/帧):")
        self.relative_speed_input = QLineEdit("18")
        self.relative_speed_input.textChanged.connect(self.update_relative_speed_threshold)
        vehicle_layout.addRow(self.relative_speed_label, self.relative_speed_input)

        vehicle_group.setLayout(vehicle_layout)
        control_layout.addWidget(vehicle_group, 6, 0, 1, 2)  # 添加到控制面板中

        # ========== 打架斗殴检测设置组 ==========
        fight_group = QGroupBox("打架斗殴检测设置")
        fight_layout = QFormLayout()

        # 动作幅度阈值
        self.fight_intensity_label = QLabel("动作幅度(碰撞程度):")
        self.fight_intensity_slider = QSlider(Qt.Horizontal)
        self.fight_intensity_slider.setMinimum(1)
        self.fight_intensity_slider.setMaximum(10)
        self.fight_intensity_slider.setValue(5)
        self.fight_intensity_slider.valueChanged.connect(self.update_fight_intensity)
        fight_layout.addRow(self.fight_intensity_label, self.fight_intensity_slider)

        fight_group.setLayout(fight_layout)
        control_layout.addWidget(fight_group, 7, 0, 1, 2)

        # ========== 报警持续帧数设置组 ==========
        alert_group = QGroupBox("报警持续帧数设置")
        alert_layout = QFormLayout()

        # 跌倒报警持续帧数
        self.fall_alert_label = QLabel("跌倒报警帧数:")
        self.fall_alert_input = QLineEdit("13")
        self.fall_alert_input.textChanged.connect(self.update_fall_alert_duration)
        alert_layout.addRow(self.fall_alert_label, self.fall_alert_input)

        # 车辆事故报警持续帧数
        self.accident_alert_label = QLabel("车辆事故报警帧数:")
        self.accident_alert_input = QLineEdit("14")
        self.accident_alert_input.textChanged.connect(self.update_accident_alert_duration)
        alert_layout.addRow(self.accident_alert_label, self.accident_alert_input)

        # 遗留物品报警持续帧数
        self.abandoned_alert_label = QLabel("遗留物品报警帧数:")
        self.abandoned_alert_input = QLineEdit("25")
        self.abandoned_alert_input.textChanged.connect(self.update_abandoned_alert_duration)
        alert_layout.addRow(self.abandoned_alert_label, self.abandoned_alert_input)

        # 打架斗殴报警持续帧数
        self.fight_alert_label = QLabel("打架斗殴报警帧数:")
        self.fight_alert_input = QLineEdit("25")
        self.fight_alert_input.textChanged.connect(self.update_fight_alert_duration)
        alert_layout.addRow(self.fight_alert_label, self.fight_alert_input)

        alert_group.setLayout(alert_layout)
        control_layout.addWidget(alert_group, 8, 0, 1, 2)

        # ========== 人流统计设置组 ==========
        crowd_group = QGroupBox("人流统计设置")
        crowd_layout = QFormLayout()

        # 人流密度预警线
        self.warning_line_label = QLabel("人流密度预警线:")
        self.warning_line_input = QLineEdit("0.1")
        self.warning_line_input.textChanged.connect(self.update_warning_line)
        crowd_layout.addRow(self.warning_line_label, self.warning_line_input)

        # 显示统计图表按钮
        self.btn_show_stats = QPushButton("显示统计图表")
        self.btn_show_stats.clicked.connect(self.show_stats)
        crowd_layout.addRow(self.btn_show_stats)

        crowd_group.setLayout(crowd_layout)
        control_layout.addWidget(crowd_group, 9, 0, 1, 2)

        # ========== 报警记录 ==========
        # 报警记录
        alert_list_group = QGroupBox("报警记录")
        alert_list_layout = QVBoxLayout()

        self.alert_list = QListWidget()
        self.alert_list.setMinimumHeight(200)
        alert_list_layout.addWidget(self.alert_list)

        # 清空按钮
        self.btn_clear = QPushButton("清空报警记录")
        self.btn_clear.clicked.connect(self.clear_alerts)
        alert_list_layout.addWidget(self.btn_clear)

        alert_list_group.setLayout(alert_list_layout)
        control_layout.addWidget(alert_list_group, 10, 0, 1, 2)

        # 设置控制面板布局
        control_panel.setLayout(control_layout)
        control_layout.addWidget(alert_list_group, 10, 0, 1, 2)

        # ========== 视频显示区域 ==========
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color: black;")

        # 添加组件到主布局
        control_panel.setMaximumWidth(380)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(self.video_label)

        # 设置主布局
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    # ========== 连接函数 ==========
    def toggle_fall_detection(self, state):
        """切换跌倒检测功能"""
        self.security_system.fall_detection_enabled = (state == Qt.Checked)
        print(f"跌倒检测: {'开启' if self.security_system.fall_detection_enabled else '关闭'}")

    def toggle_accident_detection(self, state):
        """切换事故检测功能"""
        self.security_system.accident_detection_enabled = (state == Qt.Checked)
        print(f"事故检测: {'开启' if self.security_system.accident_detection_enabled else '关闭'}")

    def toggle_abandoned_detection(self, state):
        """切换遗留物品检测功能"""
        self.security_system.abandoned_enabled = (state == Qt.Checked)
        print(f"遗留物品检测: {'开启' if self.security_system.abandoned_enabled else '关闭'}")

    def toggle_infrared_mode(self, state):
        """切换红外模式"""
        self.security_system.infrared_mode = (state == Qt.Checked)
        print(f"红外模式: {'开启' if self.security_system.infrared_mode else '关闭'}")

    def toggle_intrusion_detection(self, state):
        """切换区域入侵检测功能"""
        self.security_system.intrusion_detection_enabled = (state == Qt.Checked)
        print(f"区域入侵检测: {'开启' if self.security_system.intrusion_detection_enabled else '关闭'}")

    def toggle_fight_detection(self, state):
        """切换打架斗殴检测功能"""
        self.security_system.fight_enabled = (state == Qt.Checked)
        print(f"打架斗殴检测: {'开启' if self.security_system.fight_enabled else '关闭'}")

    def update_abandoned_time(self):
        """更新无人看管时间阈值"""
        try:
            time_threshold = float(self.abandoned_time_input.text())
            self.security_system.abandoned_object_threshold = time_threshold
            print(f"无人看管时间阈值更新为: {time_threshold}秒")
        except ValueError:
            pass

    def update_abandoned_distance(self):
        """更新物品与人距离阈值"""
        try:
            distance = int(self.abandoned_distance_input.text())
            self.security_system.abandoned_object_distance = distance
            print(f"物品与人距离阈值更新为: {distance}像素")
        except ValueError:
            pass

    def update_fight_intensity(self, value):
        """更新打架动作幅度阈值"""
        # 将滑动条值(1-10)映射到阈值0.1-1.0
        intensity_threshold = value / 10.0
        self.security_system.fight_params['min_aspect_change'] = intensity_threshold
        print(f"打架动作幅度阈值更新为: {intensity_threshold:.1f}")

    def update_fall_alert_duration(self):
        """更新跌倒报警持续帧数"""
        try:
            duration = int(self.fall_alert_input.text())
            self.security_system.fall_alert_duration = duration
            print(f"跌倒报警持续帧数更新为: {duration}")
        except ValueError:
            pass

    def update_accident_alert_duration(self):
        """更新车辆事故报警持续帧数"""
        try:
            duration = int(self.accident_alert_input.text())
            self.security_system.alert_duration = duration
            print(f"车辆事故报警持续帧数更新为: {duration}")
        except ValueError:
            pass

    def update_abandoned_alert_duration(self):
        """更新遗留物品报警持续帧数"""
        try:
            duration = int(self.abandoned_alert_input.text())
            self.security_system.abandoned_alert_duration = duration
            print(f"遗留物品报警持续帧数更新为: {duration}")
        except ValueError:
            pass

    def update_fight_alert_duration(self):
        """更新打架斗殴报警持续帧数"""
        try:
            duration = int(self.fight_alert_input.text())
            self.security_system.fight_alert_duration = duration
            print(f"打架斗殴报警持续帧数更新为: {duration}")
        except ValueError:
            pass

    def update_warning_line(self):
        """更新人流密度预警线"""
        try:
            warning_line = float(self.warning_line_input.text())
            self.security_system.warning_line = warning_line
            print(f"人流密度预警线更新为: {warning_line:.2f}")
        except ValueError:
            pass

    def show_stats(self):
        """显示统计图表"""
        if not self.stats_dialog:
            self.stats_dialog = StatsDialog(self)

        # 检查数据是否为空
        if (len(self.security_system.density_history) == 0 or
                len(self.security_system.total_count_history) == 0):
            QMessageBox.warning(self, "数据不足", "没有可用的统计数据，请先分析视频")
            return

        # 更新图表数据
        self.stats_dialog.update_charts(
            list(self.security_system.density_history),
            list(self.security_system.total_count_history)
        )

        self.stats_dialog.show()

    # 新增方法：更新车辆速度阈值
    def update_vehicle_speed_threshold(self):
        """更新车辆移动速度阈值"""
        try:
            threshold = float(self.vehicle_speed_input.text())
            self.security_system.vehicle_speed_threshold = threshold
            print(f"车辆移动速度阈值更新为: {threshold} 像素/帧")
        except ValueError:
            pass

    # 新增方法：更新相对速度阈值
    def update_relative_speed_threshold(self):
        """更新相对速度阈值"""
        try:
            threshold = float(self.relative_speed_input.text())
            self.security_system.relative_speed_threshold = threshold
            print(f"相对速度阈值更新为: {threshold} 像素/帧")
        except ValueError:
            pass

    def update_fps_display(self, fps):
        """更新FPS显示"""
        self.fps_label.setText(f"视频帧率: {fps:.2f} FPS")

    def clear_alerts(self):
        """清空报警记录"""
        self.security_system.alerts = []
        self.alert_list.clear()

    def change_speed(self, index):
        """切换视频播放速度"""
        speed_options = [0.5, 1.0, 2.0, 4.0]
        selected_speed = speed_options[index]

        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.set_playback_speed(selected_speed)
            print(f"已切换至 {selected_speed}x 倍速")

    def open_video(self):
        """安全打开视频文件（支持切换视频）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择监控视频", "", "视频文件 (*.mp4 *.avi *.mov)")

        if not file_path:
            return

        # 安全停止当前视频线程
        if self.video_thread and self.video_thread.isRunning():
            # 先停止线程
            self.video_thread.stop()
            # 等待线程结束（最多等待1秒）
            if not self.video_thread.wait(1000):
                # 强制终止
                self.video_thread.terminate()
                self.video_thread.wait()

        # 确保释放CUDA内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已释放GPU内存")

        # 重置系统状态
        self.security_system.alerts = []
        self.security_system.zone_pts = [(300, 200), (800, 200), (800, 500), (300, 500)]
        self.security_system.prev_frame = None  # 重置前一帧缓存
        self.security_system.active_visual_alerts = []  # 重置视觉警报
        self.security_system.abandoned_objects = {}  # 重置遗留物品跟踪
        self.security_system.fight_tracks = {}  # 重置打架行为跟踪
        self.alert_list.clear()
        self.progress_bar.setValue(0)
        self.fps_label.setText("视频帧率: -- FPS")  # 重置FPS显示

        # 重置历史数据
        self.security_system.density_history = deque(maxlen=100)
        self.security_system.total_count_history = deque(maxlen=100)

        # 重置检测历史
        self.security_system.prev_detections = deque(maxlen=5)
        self.security_system.vehicle_tracks = {}
        self.security_system.person_tracks = {}
        self.security_system.track_id_counter = 0
        self.security_system.accident_cooldown = {}
        self.security_system.fall_history = {}
        self.security_system.fight_cooldown = {}  # 重置打架冷却

        # 创建新的视频线程
        self.video_thread = VideoThread(self.security_system)
        self.video_thread.video_path = file_path

        # 连接信号
        self.video_thread.change_pixmap.connect(self.update_video)
        self.video_thread.finished.connect(self.video_finished)
        self.video_thread.progress_updated.connect(self.update_progress)
        self.video_thread.fps_updated.connect(self.update_fps_display)

        # 设置初始倍速
        current_speed = [0.5, 1.0, 2.0, 4.0][self.speed_combo.currentIndex()]
        self.video_thread.set_playback_speed(current_speed)

        # 启动线程
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
            self.alert_list.addItem(f"{alert['time']} - {alert['type']}")
            # 滚动到最后一条
            self.alert_list.scrollToBottom()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def video_finished(self):
        """视频处理完成"""
        self.video_label.clear()
        self.video_label.setText("视频处理完成")

        # 释放模型资源
        self.security_system.detection_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 显示完成消息框
        QMessageBox.information(self, "完成", "视频分析完成！", QMessageBox.Ok)

    def closeEvent(self, event):
        """关闭窗口时停止视频处理"""
        # 停止视频线程
        if self.video_thread and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()

        # 确保释放CUDA内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 释放模型
        if hasattr(self.security_system, 'detection_model') and self.security_system.detection_model:
            del self.security_system.detection_model
            self.security_system.detection_model = None

        # 关闭统计窗口
        if self.stats_dialog:
            self.stats_dialog.close()

        event.accept()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())