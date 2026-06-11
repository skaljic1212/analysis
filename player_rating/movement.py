"""
Stage 3 — movement metrics per player.

Uses real pitch coordinates (meters, from the homography) when a track has
enough coverage, otherwise camera-adjusted pixels scaled by PX_TO_METERS.
Positions are smoothed before differentiating, and physically impossible
steps (tracker glitches) are dropped rather than counted as distance.
"""

import numpy as np
import pandas as pd

from . import config


def _position_series(player_tracks, pid):
    """(frames, xy array, units_are_meters) for one player."""
    fns, pts_t, pts_px = [], [], []
    for fn, frame in enumerate(player_tracks):
        info = frame.get(pid)
        if info is None:
            continue
        fns.append(fn)
        pts_t.append(info.get('position_transformed'))
        pa = info.get('position_adjusted') or info.get('position')
        if pa is None:
            b = info['bbox']
            pa = ((b[0] + b[2]) / 2, b[3])
        pts_px.append(pa)

    coverage = sum(1 for p in pts_t if p is not None) / max(1, len(pts_t))
    if coverage >= config.TRANSFORMED_MIN_COVERAGE:
        xy = np.array([p if p is not None else (np.nan, np.nan) for p in pts_t],
                      dtype=float)
        meters = True
    else:
        xy = np.array(pts_px, dtype=float) * config.PX_TO_METERS
        meters = True  # scaled pixels are an estimate of meters
    return np.array(fns), xy, meters


def movement_metrics(player_tracks, pid, fps=25):
    """{'distance_m', 'sprints', 'avg_x', 'avg_y', 'frames'} for one player."""
    fns, xy, _ = _position_series(player_tracks, pid)
    out = {'distance_m': 0.0, 'sprints': 0, 'avg_x': np.nan, 'avg_y': np.nan,
           'frames': len(fns)}
    if len(fns) < 5:
        return out

    df = pd.DataFrame(xy, columns=['x', 'y'])
    df = df.interpolate(limit_area='inside')
    df = df.rolling(config.SMOOTH_WINDOW_FRAMES, center=True, min_periods=1).mean()

    out['avg_x'] = float(np.nanmean(df['x']))
    out['avg_y'] = float(np.nanmean(df['y']))

    # Per-step speed; only count steps between consecutive frames
    speeds = np.zeros(len(fns))
    dist = 0.0
    for i in range(1, len(fns)):
        dt = (fns[i] - fns[i - 1]) / fps
        if dt <= 0 or np.isnan(df.iloc[i]['x']) or np.isnan(df.iloc[i - 1]['x']):
            continue
        step = float(np.hypot(df.iloc[i]['x'] - df.iloc[i - 1]['x'],
                              df.iloc[i]['y'] - df.iloc[i - 1]['y']))
        v = step / dt
        if v > config.MAX_PLAYER_SPEED_MS:
            continue  # tracker glitch, not running
        if fns[i] - fns[i - 1] == 1:
            dist += step
            speeds[i] = v
    out['distance_m'] = round(dist, 1)

    # Sprints: speed above threshold sustained for SPRINT_MIN_SECONDS
    thr = config.SPRINT_SPEED_KMH / 3.6
    need = max(2, int(config.SPRINT_MIN_SECONDS * fps))
    win = max(1, int(config.SPEED_WINDOW_SECONDS * fps))
    smooth_v = pd.Series(speeds).rolling(win, center=True, min_periods=1).mean()
    above = (smooth_v > thr).to_numpy()
    sprints, run = 0, 0
    for a in above:
        run = run + 1 if a else 0
        if run == need:
            sprints += 1
    out['sprints'] = sprints
    return out
