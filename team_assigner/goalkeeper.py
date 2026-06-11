"""
Goalkeeper detection and team correction.

Goalkeepers wear a different jersey than their own team, so jersey-based
clustering routinely puts them on the wrong side. They are instead identified
by position (the extreme left/right long-lived tracks, far from the field
pack) and assigned to the team that *owns* their goal: per frame, the deepest
outfield player toward a goal is almost always a defender of the team
defending it (the offside rule keeps attackers in front of the last
defender), so the majority team of that deepest player marks the goal's owner.

The two keepers are jointly forced onto opposite teams, picking the
assignment with the highest combined vote confidence.
"""

import numpy as np


def detect_goalkeepers(player_tracks, roster, min_presence=0.2,
                       min_separation=350):
    """Return {'left': pid|None, 'right': pid|None}.

    A goalkeeper candidate is a long-lived non-referee track whose mean x
    position is the most extreme on its side AND clearly separated from the
    median of the field pack.
    """
    n_frames = len(player_tracks)
    xs_by_pid = {}
    for pt in player_tracks:
        for pid, info in pt.items():
            b = info["bbox"]
            xs_by_pid.setdefault(pid, []).append((b[0] + b[2]) / 2)

    means = {pid: float(np.mean(xs)) for pid, xs in xs_by_pid.items()
             if len(xs) >= min_presence * n_frames
             and roster.get_team(pid) != 0}
    if len(means) < 3:
        return {"left": None, "right": None}

    pack_median = float(np.median(list(means.values())))
    leftmost = min(means, key=means.get)
    rightmost = max(means, key=means.get)

    gks = {"left": None, "right": None}
    if pack_median - means[leftmost] >= min_separation:
        gks["left"] = leftmost
    if means[rightmost] - pack_median >= min_separation:
        gks["right"] = rightmost
    return gks


def _goal_owner_votes(player_tracks, roster, exclude_pids, sample_every=5):
    """Per goal side, the team of the deepest outfield player each frame.

    Returns {'left': {1: conf, 2: conf}, 'right': {...}} where conf is the
    fraction of sampled frames in which that team's player was deepest.
    """
    votes = {"left": [], "right": []}
    for fn in range(0, len(player_tracks), sample_every):
        field = []
        for pid, info in player_tracks[fn].items():
            if pid in exclude_pids:
                continue
            team = roster.get_team(pid, fn)
            if team not in (1, 2):
                continue
            b = info["bbox"]
            field.append(((b[0] + b[2]) / 2, team))
        if not field:
            continue
        votes["right"].append(max(field)[1])
        votes["left"].append(min(field)[1])

    conf = {}
    for side, v in votes.items():
        n = max(1, len(v))
        conf[side] = {1: v.count(1) / n, 2: v.count(2) / n}
    return conf


def assign_goalkeeper_teams(player_tracks, roster):
    """Detect both keepers and assign each the team that owns his goal.

    The two keepers are jointly constrained to opposite teams; the
    assignment maximizing combined deepest-defender confidence wins.
    Returns {'left': (pid, team), 'right': (pid, team)} for detected sides.
    """
    gks = detect_goalkeepers(player_tracks, roster)
    detected = {side: pid for side, pid in gks.items() if pid is not None}
    if not detected:
        print("  No goalkeeper tracks detected")
        return {}

    conf = _goal_owner_votes(player_tracks, roster,
                             exclude_pids=set(detected.values()))

    result = {}
    if len(detected) == 2:
        # Joint assignment: left gets team a, right gets the other team
        a = max((1, 2), key=lambda t: conf["left"][t] + conf["right"][3 - t])
        teams = {"left": a, "right": 3 - a}
    else:
        side = next(iter(detected))
        teams = {side: max((1, 2), key=lambda t: conf[side][t])}

    for side, pid in detected.items():
        team = teams[side]
        roster.set_team(pid, team)
        result[side] = (pid, team)
        print(f"  Goalkeeper ({side} goal): track {pid} -> team {team} "
              f"(deepest-defender confidence {conf[side][team]:.0%})")
    return result
