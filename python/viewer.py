import sys
import os
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QPushButton, QFileDialog, QVBoxLayout, QWidget
)
from PyQt5.QtGui import QImage, QPixmap, QPen, QPainter, QMouseEvent
from PyQt5.QtCore import Qt, QPointF

def cv_imread(file_path, dtype=np.uint8):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def cv_imwrite(file_path, frame):
    cv2.imencode('.bmp', frame)[1].tofile(file_path)

import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QFileDialog
from PyQt5.QtGui import QPainter, QPixmap, QPen, QColor, QIcon, QPainterPath
from PyQt5.QtCore import Qt, QPointF


class AOISelector(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap = QPixmap()
        self.scale = 1.0
        self.offset = QPointF(0, 0)

        # 中键拖动
        self._middle_dragging = False
        self._last_mouse_pos = None

        # AOI 数据
        self.aois = []  # 已完成 AOI 列表
        self.current_aoi = None  # 当前绘制的 AOI
        self.dragging_point_index = None
        self.dragging_aoi = None  # 当前拖动的 AOI

        # 选中点（用于删除/N键插入点）
        self.selected_point_index = None
        self.selected_aoi = None

        # 当前鼠标位置（用于 N 键插入点）
        self._mouse_pos_scene = QPointF(0, 0)

        # 绘制样式
        self.pen_line = QPen(Qt.red, 1)
        self.pen_cross = QPen(QColor(0, 255, 0), 1)

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # ================= 图片 ==================
    def set_image(self, pixmap: QPixmap):
        self.pixmap = pixmap
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self.clear_all_aois()
        self.update()

    # ================= 绘制 ==================
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)

        if not self.pixmap.isNull():
            painter.save()
            painter.translate(self.offset)
            painter.scale(self.scale, self.scale)
            painter.drawPixmap(0, 0, self.pixmap)
            painter.restore()

        all_aois = self.aois + ([self.current_aoi] if self.current_aoi else [])
        for aoi in all_aois:
            if not aoi:
                continue

            painter.save()
            painter.translate(self.offset)
            painter.scale(self.scale, self.scale)

            pts = aoi["points"]

            # ============ 绘制凸多边形填充 ============
            if len(pts) >= 2:  # 至少2个点才能绘制线或多边形
                path = QPainterPath()
                path.moveTo(pts[0])
                for pt in pts[1:]:
                    path.lineTo(pt)

                # 如果 AOI 已完成，则闭合路径
                if aoi.get("finished", False) or len(pts) >= 3:
                    path.closeSubpath()

                # 填充颜色
                color_fill = QColor(0, 255, 0, 51)  # 绿色半透明
                painter.fillPath(path, color_fill)

            # ============ 绘制轮廓 ============
            painter.setPen(self.pen_line)
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            # 实时连线
            if aoi == self.current_aoi and len(pts) >= 1 and getattr(self, "_temp_pos", None):
                painter.drawLine(pts[-1], self._temp_pos)
            # 闭合轮廓
            if aoi.get("finished", False) and len(pts) >= 3:
                painter.drawLine(pts[-1], pts[0])

            # ============ 绘制小十字点 ============
            painter.setPen(self.pen_cross)
            for pt in pts:
                self._draw_cross(painter, pt, size=4)

            painter.restore()



    def _draw_cross(self, painter, pt: QPointF, size=4):
        x, y = pt.x(), pt.y()
        painter.drawLine(QPointF(x - size, y), QPointF(x + size, y))
        painter.drawLine(QPointF(x, y - size), QPointF(x, y + size))

    # ================= 鼠标事件 ==================
    def mousePressEvent(self, event):
        pos = (event.pos() - self.offset) / self.scale
        self._temp_pos = QPointF(pos.x(), pos.y())
        self._mouse_pos_scene = QPointF(pos.x(), pos.y())
        x, y = pos.x(), pos.y()

        # 中键拖动
        if event.button() == Qt.MiddleButton:
            self._middle_dragging = True
            self._last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return

        # 左键：先判断是否点击已有点（拖动或选中）
        if event.button() == Qt.LeftButton:
            all_aois = self.aois + ([self.current_aoi] if self.current_aoi else [])
            for aoi in all_aois:
                for i, pt in enumerate(aoi["points"]):
                    if (pt - pos).manhattanLength() < 6:
                        self.dragging_point_index = i
                        self.dragging_aoi = aoi
                        self.selected_point_index = i
                        self.selected_aoi = aoi
                        return

            # 没点击点，则新增点
            if self.current_aoi is None:
                self.current_aoi = {"points": [], "finished": False, "mask": None}
            self.current_aoi["points"].append(QPointF(x, y))
            self.update()

        # 右键闭合 AOI
        elif event.button() == Qt.RightButton:
            if self.current_aoi and len(self.current_aoi["points"]) >= 3:
                self.current_aoi["finished"] = True
                self._generate_mask(self.current_aoi)
                self.aois.append(self.current_aoi)
                self.current_aoi = None
                self.update()

    def mouseMoveEvent(self, event):
        pos = (event.pos() - self.offset) / self.scale
        self._temp_pos = QPointF(pos.x(), pos.y())
        self._mouse_pos_scene = QPointF(pos.x(), pos.y())

        # 中键拖动
        if self._middle_dragging:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.pos()
            self.offset += delta
            self.update()
            return

        # 拖动点
        if self.dragging_point_index is not None and self.dragging_aoi:
            self.dragging_aoi["points"][self.dragging_point_index] = QPointF(pos.x(), pos.y())
            if self.dragging_aoi.get("finished", False):
                self._generate_mask(self.dragging_aoi)
            self.update()
            return

        # 刷新实时连线
        if self.current_aoi and not self.current_aoi.get("finished", False):
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._middle_dragging = False
            self.setCursor(Qt.ArrowCursor)
        elif event.button() == Qt.LeftButton:
            self.dragging_point_index = None
            self.dragging_aoi = None

    def wheelEvent(self, event):
        if self.pixmap.isNull():
            return
        factor = 1.1 if event.angleDelta().y() > 0 else 0.9
        mouse_pos = event.position()
        before = (mouse_pos - self.offset) / self.scale
        self.scale *= factor
        self.scale = max(0.1, min(self.scale, 15.0))
        after = before * self.scale + self.offset
        self.offset += mouse_pos - after
        self.update()

    # ================= AOI Mask ==================
    def _generate_mask(self, aoi):
        if len(aoi["points"]) < 3 or self.pixmap.isNull():
            return
        h, w = self.pixmap.height(), self.pixmap.width()
        pts = np.array([[int(p.x()), int(p.y())] for p in aoi["points"]], np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        aoi["mask"] = mask

    def save_masks(self):
        if not self.aois:
            return
        h, w = self.pixmap.height(), self.pixmap.width()
        merged_mask = np.zeros((h, w), np.uint8)
        for aoi in self.aois:
            if aoi.get("mask") is None:
                self._generate_mask(aoi)
            merged_mask = np.maximum(merged_mask, aoi["mask"])
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存整合 AOI", "aoi.png", "BMP (*.bmp);;PNG (*.png);;NPY (*.npy)"
        )
        if filename:
            if filename.endswith(".npy"):
                np.save(filename, merged_mask)
            else:
                cv2.imwrite(filename, merged_mask * 255)

    def clear_all_aois(self):
        self.aois.clear()
        self.current_aoi = None
        self.dragging_point_index = None
        self.dragging_aoi = None
        self.selected_point_index = None
        self.selected_aoi = None
        self.update()

    # ================== 键盘事件 ==================
    def keyPressEvent(self, event):
        # 删除点
        if event.key() == Qt.Key_Delete:
            if self.selected_point_index is not None and self.selected_aoi:
                idx = self.selected_point_index
                aoi = self.selected_aoi
                del aoi["points"][idx]
                if len(aoi["points"]) < 3:
                    if aoi == self.current_aoi:
                        self.current_aoi = None
                    elif aoi in self.aois:
                        self.aois.remove(aoi)
                else:
                    if aoi.get("finished", False):
                        self._generate_mask(aoi)
                self.selected_point_index = None
                self.selected_aoi = None
                self.update()

        # N 键 → 在尾部增加一个新点（跟随鼠标）
        elif event.key() == Qt.Key_I:
            target_aoi = self.selected_aoi or self.current_aoi
            if target_aoi:
                new_pt = QPointF(self._mouse_pos_scene.x(), self._mouse_pos_scene.y())
                target_aoi["points"].append(new_pt)
                if target_aoi.get("finished", False):
                    self._generate_mask(target_aoi)
                self.update()


from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AOI 编辑器")

        self.viewer = AOISelector()

        # 按钮
        self.btn_load = QPushButton("导入背景图")
        self.btn_save = QPushButton("导出 AOI Mask")
        self.btn_clear = QPushButton("清除 AOI")

        self.btn_load.clicked.connect(self.load_image)
        self.btn_save.clicked.connect(self.viewer.save_masks)
        self.btn_clear.clicked.connect(self.viewer.clear_all_aois)

        layout = QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addWidget(self.btn_load)
        layout.addWidget(self.btn_save)
        layout.addWidget(self.btn_clear)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.resize(1200, 800)

    def load_image(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "加载图片", "", "图片 (*.png *.jpg *.bmp)"
        )
        if filename:
            pix = QPixmap(filename)
            self.viewer.set_image(pix)

import base64

from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import QByteArray

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 创建窗口
    win = MainWindow()

    with open("icon.png", "rb") as f:
        icon_b64 = base64.b64encode(f.read()).decode()

    # 从 base64 加载图标
    icon_bytes = base64.b64decode(icon_b64)
    pixmap = QPixmap()
    pixmap.loadFromData(icon_bytes)
    win.setWindowIcon(QIcon(pixmap))

    win.show()
    sys.exit(app.exec_())