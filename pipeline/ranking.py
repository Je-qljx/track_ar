from dataclasses import dataclass
from tracking.position_estimator import AthletePosition


@dataclass
class RankEntry:
    rank: int
    athlete_id: int
    lane: int
    d_m: float = 0.0
    time: float = 0.0
    confidence: float = 1.0


class RankingCalculator:
    MIN_CONFIDENCE_FOR_RANK = 0.1

    def __init__(self):
        self.previous_ranks: dict[int, int] = {}

    def compute(self, positions: list[AthletePosition], current_time: float) -> list[RankEntry]:
        active = [p for p in positions if p.confidence >= self.MIN_CONFIDENCE_FOR_RANK]
        entries = []
        if active:
            sorted_positions = sorted(active, key=lambda p: p.d_m, reverse=True)
            for i, pos in enumerate(sorted_positions):
                rank = i + 1
                self.previous_ranks[pos.athlete_id] = rank
                entries.append(RankEntry(
                    rank=rank,
                    athlete_id=pos.athlete_id,
                    lane=pos.lane,
                    d_m=pos.d_m,
                    time=current_time,
                    confidence=pos.confidence,
                ))
        # Add untracked lanes at the end (rank=0 signals "not tracked")
        all_lanes = {e.lane for e in entries}
        for lane in range(1, 9):
            if lane not in all_lanes:
                entries.append(RankEntry(
                    rank=0,
                    athlete_id=lane,
                    lane=lane,
                    d_m=0.0,
                    time=current_time,
                    confidence=0.0,
                ))
        return entries
