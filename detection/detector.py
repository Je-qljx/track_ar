import numpy as np
from dataclasses import dataclass


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0

    @property
    def u_center(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def v_center(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


class BaseDetector:
    def __init__(self, conf_threshold: float = 0.5, input_size: int = 640):
        self.conf_threshold = conf_threshold
        self.input_size = input_size

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas

    def postprocess(self, raw_outputs, frame_shape: tuple[int, int]) -> list[Detection]:
        raise NotImplementedError

    def detect(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


try:
    from ultralytics import YOLO

    class YOLODetector(BaseDetector):
        def __init__(self, model_path: str = "", conf_threshold: float = 0.5, input_size: int = 640):
            super().__init__(conf_threshold, input_size)
            import torch
            import os
            if not model_path:
                model_path = os.path.join(os.path.dirname(__file__), "..", "yolov8s.pt")
            self.model = YOLO(model_path)
            if torch.cuda.is_available():
                self.model.to('cuda')

        def detect(self, frame: np.ndarray) -> list[Detection]:
            results = self.model(frame, conf=self.conf_threshold, imgsz=self.input_size, classes=[0], verbose=False)
            detections = []
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append(Detection(bbox=tuple(xyxy), confidence=conf))
            return detections

except ImportError:
    YOLO = None

    class YOLODetector(BaseDetector):
        def __init__(self, *args, **kwargs):
            raise ImportError("ultralytics is not installed. Install with: pip install ultralytics")


import cv2


class DummyDetector(BaseDetector):
    def __init__(self, num_athletes: int = 8, conf_threshold: float = 0.5):
        super().__init__(conf_threshold)
        self.num_athletes = num_athletes

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        detections = []
        lane_h = h / self.num_athletes
        for i in range(self.num_athletes):
            cy = int((i + 0.5) * lane_h)
            cx = int(w * 0.3 + (i * 20) % (w // 3))
            box_w, box_h = 60, 120
            x1 = max(0, cx - box_w // 2)
            y1 = max(0, cy - box_h // 2)
            x2 = min(w, cx + box_w // 2)
            y2 = min(h, cy + box_h // 2)
            detections.append(Detection(bbox=(x1, y1, x2, y2), confidence=0.95))
        return detections
