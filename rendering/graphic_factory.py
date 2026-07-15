import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class GraphicContent:
    rank: int
    time_str: str
    name: str = ""
    lane: int = 0

    def render_texture(self, width: int = 256, height: int = 128) -> np.ndarray:
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (0, 0, 0, 0), -1)
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (20, 20, 20, 200), -1)
        cv2.rectangle(overlay, (0, 0), (width - 1, height - 1), (255, 255, 255, 60), 2)
        rank_text = f"#{self.rank}"
        rank_size = cv2.getTextSize(rank_text, cv2.FONT_HERSHEY_DUPLEX, 1.5, 3)[0]
        rx = (width - rank_size[0]) // 2
        ry = 35
        cv2.putText(overlay, rank_text, (rx, ry), cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255, 255), 3)
        time_size = cv2.getTextSize(self.time_str, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        tx = (width - time_size[0]) // 2
        ty = 70
        cv2.putText(overlay, self.time_str, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 255, 200, 255), 2)
        if self.name:
            cv2.putText(overlay, self.name, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200, 255), 1)
        lane_colors = {
            1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 255, 255),
            4: (0, 255, 0), 5: (255, 255, 0), 6: (255, 165, 0),
            7: (255, 0, 0), 8: (128, 0, 128),
        }
        color = lane_colors.get(self.lane, (255, 255, 255))
        cv2.rectangle(overlay, (2, 2), (width - 3, height - 3), (int(color[2]), int(color[1]), int(color[0]), 100), 3)
        canvas = cv2.addWeighted(canvas, 0, overlay, 1, 0)
        return canvas
