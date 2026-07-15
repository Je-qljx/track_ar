import numpy as np
from collections import OrderedDict
from rendering.occlusion_guard import GraphicAnchor
from calibration.coords import WorldCoord


class PositionSmoother:
    def __init__(self, window: int = 3, max_window: int = 7):
        self.base_window = window
        self.max_window = max_window
        self.history: OrderedDict[int, list[float]] = OrderedDict()

    def reset(self):
        self.history.clear()

    def smooth(self, lane: int, value: float, confidence: float = 1.0) -> float:
        if lane not in self.history:
            self.history[lane] = []
        window = self.base_window + int((1.0 - confidence) * (self.max_window - self.base_window))
        window = min(window, self.max_window)
        self.history[lane].append(value)
        while len(self.history[lane]) > window:
            self.history[lane].pop(0)
        weights = np.linspace(0.5, 1.0, len(self.history[lane]))
        weights = weights / weights.sum()
        return float(np.average(self.history[lane], weights=weights))

    def smooth_anchor(self, lane: int, anchor: GraphicAnchor, confidence: float = 1.0) -> GraphicAnchor:
        smoothed_x = self.smooth(lane, anchor.world.x, confidence)
        smoothed_y = self.smooth(f"y_{lane}", anchor.world.y, confidence)
        return GraphicAnchor(
            world=WorldCoord(smoothed_x, smoothed_y, anchor.world.z),
            image=anchor.image,
            offset_ahead=anchor.offset_ahead,
            placement_mode=anchor.placement_mode,
            collision_resolved=anchor.collision_resolved,
        )
