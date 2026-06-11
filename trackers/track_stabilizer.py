"""
Post-assignment track stabilization: make track IDs follow physical players.

ByteTrack IDs break in two ways that matter for scoring players by number:

1. Collision swaps — two players collide and trade IDs. The team assigner
   detects these as mirrored mid-track team switches; here the underlying
   track data is swapped back so each ID stays on one physical player.

2. Fragmentation — a player is briefly lost and re-detected under a fresh ID.
   A track that starts moments after another of the same team ended at nearly
   the same spot is the same player; the new ID is renamed to the old one.

Both run after team assignment (they need teams) and before positions/speed
are computed (those must see the corrected tracks).
"""

import numpy as np


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _iou(b1, b2):
    ix = min(b1[2], b2[2]) - max(b1[0], b2[0])
    iy = min(b1[3], b2[3]) - max(b1[1], b2[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(1e-6, a1 + a2 - inter)


def remove_duplicate_tracks(player_tracks, roster,
                            iou_thresh=0.35, overlap_frac=0.5,
                            center_dist_px=45, dist_overlap_frac=0.4):
    """Remove ghost tracks that ride on top of another (longer) track.

    YOLO sometimes double-detects one person; ByteTrack then gives the
    duplicate box its own short-lived ID, drawn as a second ellipse on the
    same player until it dies. Two criteria, either suffices:
      - IoU: ≥overlap_frac of the track's life overlapping (IoU ≥ iou_thresh)
      - center distance: boxes can be offset enough to fail IoU while the
        ellipses/labels still stack — ≥dist_overlap_frac of the track's life
        within center_dist_px of the longer track also marks a ghost.
    """
    frames_of = {}
    for fn, pt in enumerate(player_tracks):
        for pid in pt:
            frames_of.setdefault(pid, []).append(fn)

    by_length = sorted(frames_of, key=lambda p: len(frames_of[p]))
    removed = set()
    for i, p in enumerate(by_length):
        if p in removed:
            continue
        p_frames = frames_of[p]
        for q in by_length[i + 1:]:
            if q in removed:
                continue
            shared = [fn for fn in p_frames if q in player_tracks[fn]]
            if len(shared) < dist_overlap_frac * len(p_frames):
                continue
            iou_hits = dist_hits = 0
            for fn in shared:
                bp = player_tracks[fn][p]['bbox']
                bq = player_tracks[fn][q]['bbox']
                if _iou(bp, bq) >= iou_thresh:
                    iou_hits += 1
                cp, cq = _bbox_center(bp), _bbox_center(bq)
                if np.hypot(cp[0] - cq[0], cp[1] - cq[1]) <= center_dist_px:
                    dist_hits += 1
            is_dup = (len(shared) >= overlap_frac * len(p_frames)
                      and iou_hits >= overlap_frac * len(shared)) or \
                     (dist_hits >= 0.5 * len(shared)
                      and dist_hits >= dist_overlap_frac * len(p_frames))
            if is_dup:
                removed.add(p)
                break

    for pid in removed:
        for fn in frames_of[pid]:
            player_tracks[fn].pop(pid, None)
        roster.player_teams.pop(int(pid), None)
        roster.player_segments.pop(int(pid), None)
    if removed:
        print(f"  Removed {len(removed)} duplicate ghost track(s): "
              f"{sorted(removed)}")
    return removed


def renumber_players(player_tracks, roster, gk_pids=(),
                     team1_start=1, team2_start=13, referee_start=101):
    """Relabel track IDs to friendly numbers: team 1 -> 1.., team 2 -> 13..

    Runs after all stabilization AND goalkeeper correction, so teams are
    final. Goalkeepers get their team's first number (1 / 13) like real
    shirt numbers; outfielders follow by first appearance (then old ID),
    which is deterministic — reruns produce identical numbers. Referees
    get IDs from referee_start (they are drawn without numbers anyway).
    """
    first_fn = {}
    for fn, pt in enumerate(player_tracks):
        for pid in pt:
            first_fn.setdefault(pid, fn)

    counters = {1: team1_start, 2: team2_start, 0: referee_start}
    mapping = {}
    for pid in gk_pids:  # keepers first: they take 1 and 13
        team = roster.get_team(pid)
        if pid in first_fn and team in (1, 2):
            mapping[pid] = counters[team]
            counters[team] += 1
    for pid in sorted(first_fn, key=lambda p: (first_fn[p], int(p))):
        if pid in mapping:
            continue
        team = roster.get_team(pid)
        key = team if team in (1, 2) else 0
        mapping[pid] = counters[key]
        counters[key] += 1

    for pt in player_tracks:
        renamed = {mapping[pid]: info for pid, info in pt.items()}
        pt.clear()
        pt.update(renamed)
    roster.player_teams = {mapping.get(int(p), int(p)): t
                           for p, t in roster.player_teams.items()
                           if int(p) in mapping}
    roster.player_segments = {mapping[int(p)]: s
                              for p, s in roster.player_segments.items()
                              if int(p) in mapping}

    for team, label in ((1, "team 1"), (2, "team 2")):
        pairs = sorted((new, old) for old, new in mapping.items()
                       if roster.get_team(new) == team)
        print(f"  {label} numbers: " +
              ", ".join(f"{old}->{new}" for new, old in pairs))
    return mapping


def remove_short_tracks(player_tracks, roster, min_frames=70):
    """Drop tracks still shorter than min_frames after merging.

    Whatever survives dedup+merge below ~3 seconds is either a phantom
    detection (an ellipse on empty grass) or an unattachable scrap that
    would pollute the ratings table with a fake extra player.
    """
    count = {}
    for pt in player_tracks:
        for pid in pt:
            count[pid] = count.get(pid, 0) + 1
    short = {pid for pid, n in count.items() if n < min_frames}
    for fn, pt in enumerate(player_tracks):
        for pid in short:
            pt.pop(pid, None)
    for pid in short:
        roster.player_teams.pop(int(pid), None)
        roster.player_segments.pop(int(pid), None)
    if short:
        print(f"  Removed {len(short)} short leftover track(s): {sorted(short)}")
    return short


def fix_swap_pairs(player_tracks, roster, max_frame_diff=30, max_dist=300):
    """Swap track data back for ID-swap pairs found by the team assigner.

    A swap pair is two tracks whose teams flip in mirrored directions at
    nearly the same frame (the collision). From the common switch frame on,
    their per-frame entries are exchanged, so each track ID keeps one
    physical player and one team for the entire video.
    """
    candidates = {pid: segs for pid, segs in roster.player_segments.items()
                  if len(segs) == 2}
    pids = sorted(candidates)
    used = set()
    pairs = []

    for i, p in enumerate(pids):
        if p in used:
            continue
        (_, p_before), (p_fn, p_after) = candidates[p]
        for q in pids[i + 1:]:
            if q in used:
                continue
            (_, q_before), (q_fn, q_after) = candidates[q]
            if not (p_before == q_after and p_after == q_before):
                continue
            if abs(p_fn - q_fn) > max_frame_diff:
                continue
            # Both tracks must actually be near each other at the swap
            sf = min(p_fn, q_fn)
            dist = None
            for fn in range(sf, min(sf + 15, len(player_tracks))):
                pt = player_tracks[fn]
                if p in pt and q in pt:
                    pc = _bbox_center(pt[p]["bbox"])
                    qc = _bbox_center(pt[q]["bbox"])
                    dist = float(np.hypot(pc[0] - qc[0], pc[1] - qc[1]))
                    break
            if dist is None or dist > max_dist:
                continue
            pairs.append((p, q, sf, p_before, q_before))
            used.update((p, q))
            break

    for p, q, sf, p_team, q_team in pairs:
        for fn in range(sf, len(player_tracks)):
            pt = player_tracks[fn]
            p_entry = pt.pop(p, None)
            q_entry = pt.pop(q, None)
            if p_entry is not None:
                pt[q] = p_entry
            if q_entry is not None:
                pt[p] = q_entry
        # Each ID now wears its pre-switch team for the whole video
        roster.set_team(p, p_team)
        roster.set_team(q, q_team)
        print(f"  Swap fix: tracks {p}<->{q} exchanged from frame {sf} "
              f"(track {p} stays team {p_team}, track {q} stays team {q_team})")

    return [(p, q) for p, q, *_ in pairs]


def merge_fragmented_tracks(player_tracks, roster,
                            max_gap=45, base_dist=60, per_frame_dist=4,
                            dist_cap=160, overlap_tolerance=8):
    """Rename re-detected fragments to the original track's ID.

    A track B starting ≤max_gap frames after a same-team track A ended,
    within base_dist + per_frame_dist*gap pixels (capped) of where A ended,
    is treated as the same player. ByteTrack often spawns the replacement
    ID at the exact frame the old one dies (or a few frames before), so
    handoffs with gap 0 down to -overlap_tolerance also merge — provided
    the two boxes sit on the same spot during the overlap. Conservative on
    purpose: a wrong merge corrupts per-player scoring, a missed merge
    just leaves an extra ID.
    """
    first_fn, last_fn, first_pos, last_pos = {}, {}, {}, {}
    for fn, pt in enumerate(player_tracks):
        for pid, info in pt.items():
            c = _bbox_center(info["bbox"])
            if pid not in first_fn:
                first_fn[pid] = fn
                first_pos[pid] = c
            last_fn[pid] = fn
            last_pos[pid] = c

    # Tracks with an unresolved mid-video team switch are excluded: their
    # identity is already ambiguous and merging could chain the error.
    segmented = set(roster.player_segments)

    parent = {}

    def find(pid):
        while pid in parent:
            pid = parent[pid]
        return pid

    extended = set()  # pids already continued by a later fragment
    for b in sorted(first_fn, key=lambda p: first_fn[p]):
        if b in segmented or first_fn[b] == 0:
            continue
        b_team = roster.get_team(b)
        best_a, best_d = None, None
        for a in first_fn:
            ra = find(a)
            if a == b or ra == b or a in segmented:
                continue
            gap = first_fn[b] - last_fn[a]
            if not (-overlap_tolerance < gap <= max_gap):
                continue
            if a in extended:
                continue
            if roster.get_team(a) != b_team:
                continue
            allowed = min(dist_cap, base_dist + per_frame_dist * max(1, gap))
            d = float(np.hypot(last_pos[a][0] - first_pos[b][0],
                               last_pos[a][1] - first_pos[b][1]))
            if d > allowed:
                continue
            if gap <= 0:
                # Overlapping handoff: both IDs exist briefly — they must
                # sit on the same spot or it's two different players.
                shared = [fn for fn in range(first_fn[b], last_fn[a] + 1)
                          if a in player_tracks[fn] and b in player_tracks[fn]]
                if shared:
                    sep = np.mean([np.hypot(
                        *(np.array(_bbox_center(player_tracks[fn][a]['bbox']))
                          - np.array(_bbox_center(player_tracks[fn][b]['bbox']))))
                        for fn in shared])
                    if sep > 60:
                        continue
            if best_d is None or d < best_d:
                best_a, best_d = a, d
        if best_a is not None:
            parent[b] = find(best_a)
            extended.add(best_a)
            # The merged track's lifetime extends to B's end
            last_fn[parent[b]] = last_fn[b]
            last_pos[parent[b]] = last_pos[b]

    if not parent:
        return {}

    rename = {b: find(b) for b in parent}
    for fn, pt in enumerate(player_tracks):
        for b, canon in rename.items():
            if b in pt:
                entry = pt.pop(b)
                # During an overlapping handoff keep the canonical entry
                if canon not in pt:
                    pt[canon] = entry
    for b in rename:
        roster.player_teams.pop(int(b), None)
        roster.player_segments.pop(int(b), None)

    print(f"  Merged {len(rename)} fragmented track(s) into their "
          f"original IDs ({len(set(rename.values()))} players affected)")
    return rename
