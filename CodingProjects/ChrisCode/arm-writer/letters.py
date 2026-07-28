"""
Uppercase letter stroke definitions and name → 3-D waypoint conversion.

Each letter is a list of strokes.
Each stroke is a list of (x, y) points with x, y ∈ [0, 1]:
  x = 0 (left edge) … 1 (right edge)
  y = 0 (bottom)    … 1 (top)

Strokes are drawn continuously (pen-down); between strokes the arm
lifts (pen-up).  A "PEN_UP" sentinel is inserted by name_to_waypoints.
"""

import numpy as np
from config import CANVAS_X, CANVAS_Y, CANVAS_Z

# ── 26 capital letters ────────────────────────────────────────────────────────
GLYPHS = {
    'A': [[(0.0,0.0),(0.5,1.0),(1.0,0.0)],
          [(0.2,0.4),(0.8,0.4)]],
    'B': [[(0.0,0.0),(0.0,1.0)],
          [(0.0,1.0),(0.6,1.0),(0.8,0.8),(0.8,0.6),(0.6,0.5)],
          [(0.0,0.5),(0.6,0.5),(0.9,0.3),(0.9,0.1),(0.6,0.0),(0.0,0.0)]],
    'C': [[(0.9,0.85),(0.5,1.0),(0.1,0.8),(0.0,0.5),(0.1,0.2),(0.5,0.0),(0.9,0.15)]],
    'D': [[(0.0,0.0),(0.0,1.0)],
          [(0.0,1.0),(0.5,1.0),(0.9,0.7),(0.9,0.3),(0.5,0.0),(0.0,0.0)]],
    'E': [[(0.8,0.0),(0.0,0.0),(0.0,1.0),(0.8,1.0)],
          [(0.0,0.5),(0.6,0.5)]],
    'F': [[(0.0,0.0),(0.0,1.0),(0.8,1.0)],
          [(0.0,0.5),(0.6,0.5)]],
    'G': [[(0.9,0.85),(0.5,1.0),(0.1,0.8),(0.0,0.5),(0.1,0.2),(0.5,0.0),
           (0.9,0.15),(0.9,0.5),(0.5,0.5)]],
    'H': [[(0.0,0.0),(0.0,1.0)],
          [(1.0,0.0),(1.0,1.0)],
          [(0.0,0.5),(1.0,0.5)]],
    'I': [[(0.2,0.0),(0.8,0.0)],
          [(0.5,0.0),(0.5,1.0)],
          [(0.2,1.0),(0.8,1.0)]],
    'J': [[(0.7,1.0),(0.7,0.2),(0.5,0.0),(0.2,0.1)]],
    'K': [[(0.0,0.0),(0.0,1.0)],
          [(0.9,1.0),(0.0,0.5),(0.9,0.0)]],
    'L': [[(0.0,1.0),(0.0,0.0),(0.8,0.0)]],
    'M': [[(0.0,0.0),(0.0,1.0),(0.5,0.4),(1.0,1.0),(1.0,0.0)]],
    'N': [[(0.0,0.0),(0.0,1.0),(1.0,0.0),(1.0,1.0)]],
    'O': [[(0.5,1.0),(0.1,0.8),(0.0,0.5),(0.1,0.2),(0.5,0.0),
           (0.9,0.2),(1.0,0.5),(0.9,0.8),(0.5,1.0)]],
    'P': [[(0.0,0.0),(0.0,1.0)],
          [(0.0,1.0),(0.7,1.0),(0.9,0.8),(0.9,0.6),(0.7,0.5),(0.0,0.5)]],
    'Q': [[(0.5,1.0),(0.1,0.8),(0.0,0.5),(0.1,0.2),(0.5,0.0),
           (0.9,0.2),(1.0,0.5),(0.9,0.8),(0.5,1.0)],
          [(0.6,0.3),(1.0,0.0)]],
    'R': [[(0.0,0.0),(0.0,1.0)],
          [(0.0,1.0),(0.7,1.0),(0.9,0.8),(0.9,0.6),(0.7,0.5),(0.0,0.5)],
          [(0.5,0.5),(0.9,0.0)]],
    'S': [[(0.8,0.9),(0.4,1.0),(0.1,0.8),(0.2,0.6),(0.7,0.4),
           (0.9,0.2),(0.7,0.0),(0.2,0.05)]],
    'T': [[(0.0,1.0),(1.0,1.0)],
          [(0.5,1.0),(0.5,0.0)]],
    'U': [[(0.0,1.0),(0.0,0.2),(0.2,0.0),(0.8,0.0),(1.0,0.2),(1.0,1.0)]],
    'V': [[(0.0,1.0),(0.5,0.0),(1.0,1.0)]],
    'W': [[(0.0,1.0),(0.2,0.0),(0.5,0.5),(0.8,0.0),(1.0,1.0)]],
    'X': [[(0.0,1.0),(1.0,0.0)],
          [(0.0,0.0),(1.0,1.0)]],
    'Y': [[(0.0,1.0),(0.5,0.5),(1.0,1.0)],
          [(0.5,0.5),(0.5,0.0)]],
    'Z': [[(0.0,1.0),(1.0,1.0),(0.0,0.0),(1.0,0.0)]],
}

PEN_UP = None   # sentinel between strokes


def name_to_waypoints(name: str, canvas_x=CANVAS_X,
                      canvas_y=CANVAS_Y, canvas_z=CANVAS_Z):
    """
    Convert a string (A–Z) into a list of 3-D waypoints in the arm workspace.

    Returns a list of items where each item is either:
      np.ndarray shape (3,)  → move LED here (pen down)
      None                   → pen-up transition (arm lifts, then moves)

    Letter slots are laid out left-to-right across the canvas width.
    The pen-up height is 1 cm above the canvas surface (canvas_z[1] + 0.01).
    """
    name   = name.upper()
    n      = len(name)
    if n == 0:
        return []

    # Width of one letter slot (fraction of canvas width), with 10% spacing
    slot   = 1.0 / (n + (n - 1) * 0.1)
    gap    = slot * 0.1
    y_lo, y_hi = canvas_y
    z_lo, z_hi = canvas_z
    lift_z = z_hi + 0.01    # pen-up height

    waypoints = []
    for i, ch in enumerate(name):
        if ch not in GLYPHS:
            continue

        # X offset for this letter slot (normalized 0→1 maps to arm y-axis)
        x_offset = i * (slot + gap)

        for stroke_idx, stroke in enumerate(GLYPHS[ch]):
            if stroke_idx > 0:
                # Pen-up: lift to safe height, move to start of next stroke
                waypoints.append(PEN_UP)
                sx, sy = stroke[0]
                nx = x_offset + sx * slot
                wp_y = y_lo + nx * (y_hi - y_lo)
                wp_z = lift_z
                waypoints.append(np.array([canvas_x, wp_y, wp_z]))

            for (sx, sy) in stroke:
                # Map normalized letter coords into arm workspace
                nx    = x_offset + sx * slot       # x within full name width
                wp_y  = y_lo + nx * (y_hi - y_lo)
                wp_z  = z_lo + sy * (z_hi - z_lo)
                waypoints.append(np.array([canvas_x, wp_y, wp_z]))

        # Pen-up after each letter
        waypoints.append(PEN_UP)

    return waypoints
