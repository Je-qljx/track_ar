import time
import csv
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class MetricRecord:
    timestamp: float
    frame_id: int
    fps: float
    pipeline_latency_ms: float
    num_athletes: int
    num_alerts: int


class MetricsLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.start_time = time.time()
        self.csv_writer: Optional[csv.writer] = None
        self.csv_file: Optional[Path] = None

    def start_session(self, session_name: Optional[str] = None):
        if session_name is None:
            session_name = f"session_{int(self.start_time)}"
        self.csv_file = self.log_dir / f"{session_name}.csv"
        f = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(f)
        self.csv_writer.writerow([
            "timestamp", "frame_id", "fps", "pipeline_latency_ms",
            "num_athletes", "num_alerts",
        ])

    def log(self, record: MetricRecord):
        if self.csv_writer:
            self.csv_writer.writerow([
                f"{record.timestamp:.3f}", record.frame_id,
                f"{record.fps:.1f}", f"{record.pipeline_latency_ms:.2f}",
                record.num_athletes, record.num_alerts,
            ])

    def close(self):
        pass
