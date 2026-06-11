"""
Stage 1 — smoothed per-frame ball possession.

Raw nearest-player assignment flickers when two players are close to the
ball. Hysteresis fixes it: the current holder keeps possession until a
challenger is nearest for MIN_POSSESSION_FRAMES consecutive frames, and
possession only becomes vacant after the ball is away from everyone for
POSSESSION_LOSS_FRAMES.
"""

import numpy as np

from . import config


def _ball_center(ball_frame):
    b = ball_frame.get(1, {}).get('bbox')
    if not b:
        return None
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _nearest_player(player_frame, ball_pos):
    """Nearest field player (team 1/2) within MAX_BALL_DISTANCE_PX of the ball.

    Distance is measured from the closer bottom corner of the player's bbox
    (their feet), matching PlayerBallAssigner's convention.
    """
    best_pid, best_d = None, config.MAX_BALL_DISTANCE_PX
    for pid, info in player_frame.items():
        if info.get('team') not in (1, 2):
            continue  # referees never hold possession
        b = info['bbox']
        d = min(np.hypot(b[0] - ball_pos[0], b[3] - ball_pos[1]),
                np.hypot(b[2] - ball_pos[0], b[3] - ball_pos[1]))
        if d < best_d:
            best_pid, best_d = pid, d
    return best_pid


def compute_possession(player_tracks, ball_tracks):
    """Per-frame possession holder (track_id or None), smoothed."""
    n = len(player_tracks)
    raw = []
    for fn in range(n):
        ball_pos = _ball_center(ball_tracks[fn]) if fn < len(ball_tracks) else None
        raw.append(_nearest_player(player_tracks[fn], ball_pos)
                   if ball_pos is not None else None)

    possession = [None] * n
    holder = None
    cand, cand_streak = None, 0
    none_streak = 0

    for fn, pid in enumerate(raw):
        if pid is None:
            none_streak += 1
            cand, cand_streak = None, 0
            if none_streak >= config.POSSESSION_LOSS_FRAMES:
                holder = None
        else:
            none_streak = 0
            if pid == holder:
                cand, cand_streak = None, 0
            else:
                if pid == cand:
                    cand_streak += 1
                else:
                    cand, cand_streak = pid, 1
                needed = (config.FREE_BALL_TAKE_FRAMES if holder is None
                          else config.MIN_POSSESSION_FRAMES)
                if cand_streak >= needed:
                    holder = pid
                    cand, cand_streak = None, 0
        possession[fn] = holder

    return possession


def possession_segments(possession):
    """[(pid, start_frame, end_frame)] for every continuous holding spell."""
    segments = []
    start = None
    for fn, pid in enumerate(possession):
        if start is None or possession[start] != pid:
            if start is not None and possession[start] is not None:
                segments.append((possession[start], start, fn - 1))
            start = fn
    if start is not None and possession[start] is not None:
        segments.append((possession[start], start, len(possession) - 1))
    return segments
