import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional

from calibration.coords import WorldCoord, ImageCoord
from calibration.projector import Projector
from rendering.graphic_factory import GraphicContent
from rendering.occlusion_guard import GraphicAnchor, compute_graphic_bbox


@dataclass
class DecalInstance:
    graphic: Optional[GraphicContent] = None
    anchor: GraphicAnchor = None
    texture: np.ndarray = field(default_factory=lambda: np.zeros((128, 256, 4), dtype=np.uint8))


class DecalRenderer:
    def __init__(self, projector: Projector):
        self.projector = projector

    def render_decal(self, canvas: np.ndarray, instance: DecalInstance):
        g_bbox = compute_graphic_bbox(
            instance.anchor.world, self.projector,
            graphic_width_m=0.8 * 1.22, graphic_height_m=0.4,
        )
        u1, v1, u2, v2 = map(int, g_bbox)
        u1 = max(0, min(u1, canvas.shape[1] - 2))
        v1 = max(0, min(v1, canvas.shape[0] - 2))
        u2 = max(u1 + 1, min(u2, canvas.shape[1]))
        v2 = max(v1 + 1, min(v2, canvas.shape[0]))
        if u2 - u1 < 4 or v2 - v1 < 4:
            return
        overlay = cv2.resize(instance.texture, (u2 - u1, v2 - v1), interpolation=cv2.INTER_LINEAR)
        if overlay.shape[2] == 4:
            a = overlay[:, :, 3:4].astype(np.uint16)
            fg = overlay[:, :, :3].astype(np.uint16)
            bg = canvas[v1:v2, u1:u2].astype(np.uint16)
            blended = ((fg * a + bg * (255 - a)) // 255).astype(np.uint8)
            canvas[v1:v2, u1:u2] = blended
        else:
            canvas[v1:v2, u1:u2] = overlay
