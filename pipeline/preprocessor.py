import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class PreprocessedFrame:
    original: np.ndarray
    scaled: np.ndarray
    timestamp: float
    frame_id: int


class Preprocessor:
    def __init__(self, target_size: int = 640, enable_undistortion: bool = True):
        self.target_size = target_size
        self.enable_undistortion = enable_undistortion
        self.frame_count = 0

    def process(self, frame: np.ndarray, timestamp: float) -> PreprocessedFrame:
        self.frame_count += 1
        scaled = self._resize_to_square(frame)
        return PreprocessedFrame(
            original=frame,
            scaled=scaled,
            timestamp=timestamp,
            frame_id=self.frame_count,
        )

    def _resize_to_square(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.target_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))
        canvas = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas
