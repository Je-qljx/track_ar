import numpy as np
import cv2
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SynthAthlete:
    lane: int
    d_m: float
    speed: float


class SynthVideoGenerator:
    def __init__(self, duration_sec: float = 12.0, fps: float = 60.0, width: int = 1920, height: int = 1080):
        self.duration_sec = duration_sec
        self.fps = fps
        self.width = width
        self.height = height
        self.total_frames = int(duration_sec * fps)
        self.lane_h = height / 8

    def generate(self, output_path: str, ground_truth_path: str | None = None):
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), self.fps, (self.width, self.height))
        gt_records = []
        speeds = [9.5, 9.8, 10.2, 9.0, 8.5, 9.3, 8.8, 10.5]
        for frame_idx in range(self.total_frames):
            t = frame_idx / self.fps
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 40
            # draw track lanes
            for i in range(8):
                y0 = int(i * self.lane_h)
                cv2.rectangle(canvas, (0, y0), (self.width, y0 + int(self.lane_h)), (50, 50, 50), 1)
            # draw start/finish
            cv2.line(canvas, (150, 0), (150, self.height), (0, 0, 255), 3)
            cv2.putText(canvas, "START", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            finish_x = int(self.width * 0.85)
            cv2.line(canvas, (finish_x, 0), (finish_x, self.height), (0, 255, 0), 3)
            cv2.putText(canvas, "FINISH", (finish_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame_gt = []
            for lane in range(8):
                d_m = max(0, min(t * speeds[lane], 100))
                cx = int(150 + (d_m / 100.0) * (finish_x - 150))
                cy = int((lane + 0.5) * self.lane_h)
                color = (
                    int(50 + lane * 25) % 256,
                    int(100 + lane * 15) % 256,
                    int(200 - lane * 20) % 256,
                )
                cv2.circle(canvas, (cx, cy), 20, color, -1)
                cv2.rectangle(canvas, (cx - 15, cy - 30), (cx + 15, cy + 30), color, 2)
                cv2.putText(canvas, f"L{lane+1}", (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                frame_gt.append(SynthAthlete(lane=lane + 1, d_m=d_m, speed=speeds[lane]))
            writer.write(canvas)
            gt_records.append(frame_gt)
        writer.release()
        if ground_truth_path:
            self._save_ground_truth(ground_truth_path, gt_records)
        print(f"Synth video saved to {output_path} ({self.total_frames} frames)")

    def _save_ground_truth(self, path: str, records: list[list[SynthAthlete]]):
        with open(path, 'w') as f:
            f.write("frame,lane,d_m,speed\n")
            for frame_idx, athletes in enumerate(records):
                for a in athletes:
                    f.write(f"{frame_idx},{a.lane},{a.d_m:.3f},{a.speed:.3f}\n")
        print(f"Ground truth saved to {path}")
