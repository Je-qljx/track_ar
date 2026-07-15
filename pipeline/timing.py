from dataclasses import dataclass, field


class RaceTimer:
    def __init__(self):
        self.t0: float | None = None
        self.race_started: bool = False
        self.race_finished: bool = False
        self.finish_time: float = 0.0

    def start_race(self, timestamp: float | None = None):
        if not self.race_started:
            self.t0 = timestamp if timestamp is not None else 0.0
            self.race_started = True
            self.race_finished = False

    def get_elapsed(self, timestamp: float | None = None) -> float:
        if self.t0 is None:
            return 0.0
        if self.race_finished:
            return self.finish_time - self.t0
        now = timestamp if timestamp is not None else 0.0
        return now - self.t0

    def format_time(self, elapsed: float | None = None, timestamp: float | None = None) -> str:
        if elapsed is None:
            elapsed = self.get_elapsed(timestamp)
        if elapsed >= 60:
            minutes = int(elapsed) // 60
            secs = elapsed % 60
            return f"{minutes}:{secs:06.3f}"
        return f"{elapsed:06.3f}s"

    def finish_race(self, timestamp: float | None = None):
        if not self.race_finished:
            self.finish_time = timestamp if timestamp is not None else 0.0
            self.race_finished = True

    def reset(self):
        self.t0 = None
        self.race_started = False
        self.race_finished = False
        self.finish_time = 0.0
