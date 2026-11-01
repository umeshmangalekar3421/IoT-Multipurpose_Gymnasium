"""Configuration loading with sane defaults and dotted-key access."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict, Optional

try:  # PyYAML is the only third-party requirement for config files
    import yaml
except Exception:  # pragma: no cover - json fallback for minimal edge images
    yaml = None
    import json


DEFAULTS: Dict[str, Any] = {
    "sport": "badminton",
    "match": {
        "format": "singles",          # singles | doubles
        "best_of": 3,
        "players": ["P1", "P2"],
        "first_server": "P1",
        "switch_sides_at_half": True,
    },
    "video": {
        "fps": None,                  # auto-detect
        "max_frames": None,
        "stride": 1,
    },
    "camera": {
        "court_corners_img": None,    # [[x,y] * 4] image points of court corners
        "homography": None,           # 3x3 nested list image->court
        "auto_calibrate": True,
        "corner_order": "tl,tr,br,bl",
    },
    "detector": {
        "ball": "classical",          # classical | yolo | none
        "weights": None,
        "conf": 0.25,
        "device": "cpu",
        "imgsz": 640,
        "players": "none",            # none | yolo | color
        "player_weights": None,
    },
    "tracking": {
        "gate_px": 90.0,
        "max_coast_frames": 12,
        "max_speed_px_s": 6000.0,
    },
    "events": {
        "model": "rules",             # rules | mlp
        "model_path": None,
        "min_bounce_prominence_px": 8.0,
        "min_bounce_spacing_frames": 4,
        "line_margin_m": 0.03,        # ball radius tolerance for line calls
        "rally_gap_frames": 120,      # frames without a contact => rally over (4 s @30fps)
        "return_window_frames": 40,
        "floor_contact_ends_rally": False,  # true = a floor touch always ends the rally
        "confirm_landing_frames": 10,  # window to revoke a false bounce when a return appears
        "on_track_loss": "opponent",
        "net_band_px": 18,
    },
    "evidence": {
        "enabled": True,
        "out_dir": "runs/match",
        "pre_frames": 15,
        "post_frames": 40,
        "jpeg_quality": 90,
        "annotate": True,
        "save_scoreboard_png": True,
    },
    "output": {
        "timeline_json": None,
        "annotated_video": None,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config(dict):
    """dict with ``cfg["a.b.c"]`` and ``cfg.get_path("a.b")`` helpers."""

    def get_path(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Load YAML/(JSON) config merged over :data:`DEFAULTS`.

    Unknown keys are preserved (they are usually sport-specific extras) and
    every default is always present, so downstream code never has to guard.
    """
    user: Dict[str, Any] = {}
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"config not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            if yaml is not None:
                user = yaml.safe_load(fh) or {}
            else:  # pragma: no cover
                user = json.load(fh)
    merged = _deep_merge(DEFAULTS, user)
    if overrides:
        merged = _deep_merge(merged, overrides)
    return Config(merged)
