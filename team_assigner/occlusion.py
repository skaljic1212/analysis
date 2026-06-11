"""
Collision/occlusion detection between player bounding boxes.

When two players overlap, the jersey crop of one contains pixels of the
other, so any team vote taken from such a frame is unreliable. Assigners
use this map to skip contaminated frames during classification.
"""

from collections import defaultdict


def compute_occluded_detections(player_tracks, overlap_thresh=0.15):
    """Map of pid -> sorted [frame_nums] where that player's bbox overlaps
    another player's bbox by more than overlap_thresh of its own area.
    """
    occluded = defaultdict(set)
    for fn, player_track in enumerate(player_tracks):
        items = list(player_track.items())
        for i in range(len(items)):
            pid_i, info_i = items[i]
            b1 = info_i["bbox"]
            a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
            for j in range(i + 1, len(items)):
                pid_j, info_j = items[j]
                b2 = info_j["bbox"]
                ix = min(b1[2], b2[2]) - max(b1[0], b2[0])
                iy = min(b1[3], b2[3]) - max(b1[1], b2[1])
                if ix <= 0 or iy <= 0:
                    continue
                inter = ix * iy
                a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                if a1 > 0 and inter / a1 > overlap_thresh:
                    occluded[pid_i].add(fn)
                if a2 > 0 and inter / a2 > overlap_thresh:
                    occluded[pid_j].add(fn)
    return {pid: sorted(fns) for pid, fns in occluded.items()}


def is_occluded(occluded, pid, frame_num):
    fns = occluded.get(pid)
    return fns is not None and frame_num in fns


def occlusions_in_window(occluded, pid, start_fn, end_fn):
    """Sorted occlusion frames for pid within [start_fn, end_fn]."""
    return [f for f in occluded.get(pid, []) if start_fn <= f <= end_fn]
