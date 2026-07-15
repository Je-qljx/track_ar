import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Callable
from threading import Thread, Lock
from collections import deque


@dataclass
class VideoSource:
    uri: str
    width: int = 1920
    height: int = 1080
    fps: float = 60.0
    is_ndi: bool = False


class VideoInput:
    def __init__(self, buffer_size: int = 30):
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.lock = Lock()
        self.thread: Optional[Thread] = None
        self.running = False
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self, source: VideoSource | str):
        uri = source if isinstance(source, str) else source.uri
        self.cap = cv2.VideoCapture(uri)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {uri}")
        return self

    def start(self):
        self.running = True
        self.thread = Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            with self.lock:
                self.buffer.append(frame)

    def read(self) -> Optional[np.ndarray]:
        with self.lock:
            if self.buffer:
                return self.buffer.popleft()
        return None

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()


class VideoOutput:
    def __init__(self):
        self.writer: Optional[cv2.VideoWriter] = None

    def open(self, path: str, fps: float = 60.0, size: tuple[int, int] = (1920, 1080)):
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(path, fourcc, fps, size)
        return self

    def write(self, frame: np.ndarray):
        if self.writer:
            self.writer.write(frame)

    def close(self):
        if self.writer:
            self.writer.release()
            self.writer = None
