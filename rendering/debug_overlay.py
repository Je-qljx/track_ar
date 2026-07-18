import numpy as np
import cv2
from typing import Optional

from rendering.occlusion_guard import GraphicAnchor
from tracking.lane_assigner import AthleteState


class DebugOverlay:
    def __init__(self, show_bboxes: bool = True, show_anchors: bool = True, show_depth: bool = False):
        self.show_bboxes = show_bboxes
        self.show_anchors = show_anchors
        self.show_depth = show_depth

    def draw(self, canvas: np.ndarray, athletes: dict[int, AthleteState],
             anchors: dict[int, GraphicAnchor] | None = None,
             frame_count: int = 0, fps: float = 0.0):
        if self.show_bboxes:
            for lane, athlete in athletes.items():
                if athlete.detection and athlete.frames_tracked > 2:
                    x1, y1, x2, y2 = map(int, athlete.detection.bbox)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(canvas, f"L{lane}#{athlete.athlete_id}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.show_anchors and anchors:
            for lane, anchor in anchors.items():
                pt = (int(anchor.image.u), int(anchor.image.v))
                color = (0, 255, 255) if anchor.placement_mode == "ahead" else (255, 0, 255)
                cv2.circle(canvas, pt, 8, color, -1)
                cv2.putText(canvas, f"L{lane} {anchor.placement_mode}",
                            (pt[0] + 12, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(canvas, f"Frame: {frame_count} | FPS: {fps:.1f}",
                    (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
