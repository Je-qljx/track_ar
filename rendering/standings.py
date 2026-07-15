import numpy as np
import cv2
from pipeline.timing import RaceTimer
from pipeline.ranking import RankEntry
from tracking.position_estimator import AthletePosition


class StandingsPanel:
    PANEL_X = 20
    PANEL_Y = 60
    PANEL_W = 440
    ROW_H = 28
    HEADER_H = 24
    TIMER_H = 40

    def __init__(self):
        self.finish_times: dict[int, float] = {}

    def reset(self):
        self.finish_times.clear()

    def draw(self, canvas: np.ndarray, ranks: list[RankEntry],
             positions: list[AthletePosition], timer: RaceTimer,
             athlete_names: dict[int, str], geometry_length: float = 100.0,
             video_timestamp: float | None = None,
             finish_distances: dict[int, float] | None = None):
        h, w = canvas.shape[:2]
        for pos in positions:
            fd = finish_distances.get(pos.lane, geometry_length) if finish_distances is not None else geometry_length
            if pos.d_m >= fd - 0.5 and pos.lane not in self.finish_times and timer.race_started and pos.confidence > 0.0:
                self.finish_times[pos.lane] = timer.get_elapsed(video_timestamp)

        elapsed = timer.get_elapsed(video_timestamp) if timer.race_started else 0.0
        timer_text = f"Race Timer:  {timer.format_time(elapsed)}" if timer.race_started else "Race Timer:  --.---s"
        all_finished = len(self.finish_times) >= 8 or timer.race_finished

        n_entries = len(ranks)
        panel_h = self.TIMER_H + self.HEADER_H + n_entries * self.ROW_H + 16
        x1, y1 = self.PANEL_X, self.PANEL_Y
        x2 = min(x1 + self.PANEL_W, w - 10)
        y2 = min(y1 + panel_h, h - 10)
        if x2 <= x1 or y2 <= y1:
            return

        sub = canvas[y1:y2, x1:x2]
        overlay = np.full_like(sub, (10, 10, 10), dtype=np.uint8)
        cv2.addWeighted(sub, 0.85, overlay, 0.15, 0, dst=sub)

        yy = y1 + 8
        cv2.putText(canvas, timer_text, (x1 + 12, yy + 26),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (255, 255, 200), 2)

        yy += self.TIMER_H
        cv2.rectangle(canvas, (x1 + 8, yy), (x2 - 8, yy), (100, 100, 100), 1)

        lane_colors = {
            1: (0, 0, 255), 2: (0, 165, 255), 3: (0, 255, 255),
            4: (0, 255, 0), 5: (255, 255, 0), 6: (255, 165, 0),
            7: (255, 0, 0), 8: (128, 0, 128),
        }

        yy += 6
        cv2.putText(canvas, f"{'#':<3} {'Lane':<5} {'Athlete':<14} {'Dist':<8} {'Time'}", (x1 + 10, yy + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        yy += self.HEADER_H - 4
        for entry in ranks:
            name = athlete_names.get(entry.lane, f"Lane {entry.lane}")
            color = lane_colors.get(entry.lane, (255, 255, 255))

            is_finished = entry.lane in self.finish_times
            is_tracked = entry.confidence > 0.0
            if is_finished:
                time_str = f"{self.finish_times[entry.lane]:06.3f}s"
                dist_str = f"{geometry_length:.0f}.0m"
            elif is_tracked and timer.race_started:
                time_str = f"{entry.time:.3f}s"
                dist_str = f"{entry.d_m:5.1f}m"
            elif is_tracked:
                time_str = "--.---s"
                dist_str = f"{entry.d_m:5.1f}m"
            else:
                time_str = "--.---s"
                dist_str = "---.-m"

            rank_text = f"#{entry.rank}" if entry.rank > 0 else " - "
            entry_line = f"{rank_text:<3} L{entry.lane:<3} {name:<14} {dist_str:<8} {time_str}"

            if all_finished and is_finished:
                fg = (100, 255, 100)
            elif is_finished:
                fg = (200, 255, 200)
            elif is_tracked:
                fg = (220, 220, 220)
            else:
                fg = (100, 100, 100)
                color = (60, 60, 60)

            cv2.putText(canvas, entry_line, (x1 + 10, yy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1)
            cv2.rectangle(canvas, (x1 + 6, yy), (x2, yy + self.ROW_H),
                          (int(color[2]), int(color[1]), int(color[0])), 2)
            yy += self.ROW_H

        if all_finished:
            msg = "RACE COMPLETE"
            msg_w = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0][0]
            mx = (x1 + x2 - msg_w) // 2
            cv2.putText(canvas, msg, (mx, y2 - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
