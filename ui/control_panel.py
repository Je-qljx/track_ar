import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from pipeline.main_pipeline import TrackARPipeline
from calibration.coords import WorldCoord


@dataclass
class ControlState:
    overlay_enabled: bool = True
    show_bboxes: bool = True
    show_anchors: bool = True
    offset_ahead: float = 2.0
    offset_behind: float = 1.0
    opacity: float = 0.8
    paused: bool = False


class ControlPanel:
    def __init__(self, pipeline: TrackARPipeline):
        self.pipeline = pipeline
        self.state = ControlState()
        self.window_name = "TrackAR Control"
        self.mouse_pos: tuple[int, int] = (0, 0)
        self.selected_lane: int = -1
        self.click_callback: Callable | None = None
        self.on_reset: Callable | None = None

    def set_click_callback(self, callback: Callable):
        self.click_callback = callback

    def mouse_callback(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.click_callback:
            self.click_callback(x, y)

    def create_trackbars(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.createTrackbar("Opacity %", self.window_name, int(self.state.opacity * 100), 100, self._on_opacity)
        cv2.createTrackbar("Offset Ahead (m*10)", self.window_name, int(self.state.offset_ahead * 10), 100, self._on_offset_ahead)
        cv2.createTrackbar("Offset Behind (m*10)", self.window_name, int(self.state.offset_behind * 10), 50, self._on_offset_behind)

    def _on_opacity(self, val):
        self.state.opacity = val / 100.0

    def _on_offset_ahead(self, val):
        self.state.offset_ahead = val / 10.0

    def _on_offset_behind(self, val):
        self.state.offset_behind = val / 10.0

    def draw_controls(self, canvas: np.ndarray):
        overlay = canvas.copy()
        h, w = canvas.shape[:2]
        panel_w = 250
        cv2.rectangle(overlay, (0, 0), (panel_w, h), (0, 0, 0, 180), -1)
        y = 60
        cv2.putText(overlay, "TRACK AR CONTROL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(overlay, f"Paused: {self.state.paused}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 30
        cv2.putText(overlay, f"Overlay: {'ON' if self.state.overlay_enabled else 'OFF'}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.state.overlay_enabled else (0, 0, 255), 1)
        y += 30
        cv2.putText(overlay, f"Offset Ahead: {self.state.offset_ahead:.1f}m", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 30
        cv2.putText(overlay, f"Offset Behind: {self.state.offset_behind:.1f}m", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        y += 30
        cv2.putText(overlay, f"FPS: {self.pipeline.fps:.1f}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        y += 30
        follow = self.pipeline.dynamic_camera.follow_mode
        cv2.putText(overlay, f"Follow: {'ON' if follow else 'OFF'}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if follow else (150, 150, 150), 1)
        y += 30
        cv2.putText(overlay, "Keys: [p] pause [b] bbox [o] overlay [f] follow [r] reset", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

    def handle_key(self, key: int) -> bool:
        if key == ord('p'):
            self.state.paused = not self.state.paused
        elif key == ord('b'):
            self.state.show_bboxes = not self.state.show_bboxes
            self.pipeline.debug_overlay.show_bboxes = self.state.show_bboxes
        elif key == ord('o'):
            self.state.overlay_enabled = not self.state.overlay_enabled
        elif key == ord('f'):
            self.pipeline.dynamic_camera.follow_mode = not self.pipeline.dynamic_camera.follow_mode
            if not self.pipeline.dynamic_camera.follow_mode:
                self.pipeline.projector.look_at(WorldCoord(50.0, 4.88, 0.0))
        elif key == ord('r'):
            self.pipeline.reset()
            self.pipeline.dynamic_camera.prev_look_x = None
            if not self.pipeline.dynamic_camera.follow_mode:
                self.pipeline.projector.look_at(WorldCoord(50.0, 4.88, 0.0))
            if self.on_reset:
                self.on_reset()
        elif key == ord('q'):
            return False
        return True
