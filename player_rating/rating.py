"""
Stage 4 — accumulated grades (points), roles, and the 0-10 rating.

Points are the accumulated A-style grades from every event credit
(good actions add, bad subtract; one action can combine categories).
The 0-10 rating normalizes per-minute rates within role groups
(GK / defender / midfielder / attacker) so a keeper is never compared
to a striker, and a player tracked 200 frames isn't punished for
having fewer chances than one tracked 750.
"""

import numpy as np

from . import config
from .movement import movement_metrics

CATEGORY_COLUMNS = [
    'Short pass', 'Long pass', 'Through ball', 'Under pressure',
    'Failed pass', 'Intercept', 'Challenge won', 'Challenge lost',
    'Won in own third', 'Pressure', 'Close control', 'Link up', 'Dribble',
]


def infer_roles(player_tracks, stats, attack_dir=None, gk_pids=()):
    """Assign 'GK' / 'defender' / 'midfielder' / 'attacker' to each pid."""
    roles = {}
    by_team = {}
    for pid, s in stats.items():
        if pid in gk_pids:
            roles[pid] = 'GK'
        else:
            by_team.setdefault(s['team'], []).append(pid)

    # depth = how deep toward own goal a player's average x is
    for team, pids in by_team.items():
        depths = {}
        for pid in pids:
            xs = [(f[pid]['bbox'][0] + f[pid]['bbox'][2]) / 2
                  for f in player_tracks if pid in f]
            mean_x = float(np.mean(xs)) if xs else 0.0
            d = -mean_x * attack_dir[team] if attack_dir and team in attack_dir \
                else mean_x
            depths[pid] = d
        order = sorted(pids, key=lambda p: -depths[p])  # deepest first
        n = len(order)
        n_def = max(1, round(n * config.ROLE_DEFENDER_PCT))
        n_att = max(1, round(n * config.ROLE_ATTACKER_PCT))
        for i, pid in enumerate(order):
            if i < n_def:
                roles[pid] = 'defender'
            elif i >= n - n_att:
                roles[pid] = 'attacker'
            else:
                roles[pid] = 'midfielder'
    return roles


def collect_stats(player_tracks, events, fps=25, attack_dir=None, gk_pids=()):
    """{pid: stats dict} for every rated player (referees excluded)."""
    frames_by_pid, team_by_pid = {}, {}
    for frame in player_tracks:
        for pid, info in frame.items():
            if info.get('team') in (1, 2):
                frames_by_pid[pid] = frames_by_pid.get(pid, 0) + 1
                team_by_pid[pid] = info['team']

    stats = {}
    for pid, nframes in frames_by_pid.items():
        if nframes < config.MIN_TRACK_FRAMES:
            continue
        stats[pid] = {
            'team': team_by_pid[pid],
            'frames': nframes,
            'minutes': nframes / fps / 60.0,
            'points': 0.0,
            'categories': {c: 0 for c in CATEGORY_COLUMNS},
        }

    # accumulate grades + category counts from event credits
    for e in events:
        for pid, grade, category in e['credits']:
            if pid not in stats:
                continue
            stats[pid]['points'] += grade
            if category in stats[pid]['categories']:
                stats[pid]['categories'][category] += 1

    for pid, s in stats.items():
        s['points'] = round(s['points'], 2)
        s.update(movement_metrics(player_tracks, pid, fps=fps))
        c = s['categories']
        completed = c['Short pass'] + c['Long pass'] + c['Through ball']
        attempts = completed + c['Failed pass']
        s['pass_attempts'] = attempts
        s['pass_accuracy'] = 100.0 * completed / attempts if attempts else np.nan
        duels = c['Challenge won'] + c['Challenge lost']
        s['challenge_win_rate'] = (100.0 * c['Challenge won'] / duels
                                   if duels else np.nan)
        # sprints earn Pace points too
        s['points'] = round(s['points'] + s['sprints'] * config.GRADES['sprint'], 2)
        m = max(s['minutes'], 1e-6)
        s['points_per_10min'] = round(s['points'] / m * 10, 2)
        s['distance_per_min'] = round(s['distance_m'] / m, 1)
        s['sprints_per_10min'] = round(s['sprints'] / m * 10, 2)

    roles = infer_roles(player_tracks, stats, attack_dir=attack_dir,
                        gk_pids=gk_pids)
    for pid in stats:
        stats[pid]['role'] = roles.get(pid, 'midfielder')
    return stats


def _minmax(values):
    arr = np.array(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0 or finite.max() == finite.min():
        return np.full(len(arr), 0.5)
    norm = (arr - finite.min()) / (finite.max() - finite.min())
    norm[~np.isfinite(arr)] = 0.5  # no data -> neutral
    return np.clip(norm, 0, 1)


def add_ratings(stats):
    """0-10 rating from per-minute rates, normalized within role groups."""
    if not stats:
        return stats

    groups = {}
    for pid, s in stats.items():
        groups.setdefault(s['role'], []).append(pid)
    # small groups are rated against all outfielders instead
    outfield = [p for p, s in stats.items() if s['role'] != 'GK']
    rating_groups = []
    for role, pids in groups.items():
        rating_groups.append(pids if len(pids) >= 3 or role == 'GK'
                             else outfield)

    done = set()
    w = config.RATING_WEIGHTS
    total_w = sum(w.values())
    for pids in rating_groups:
        norm = {
            'points_rate': _minmax([stats[p]['points_per_10min'] for p in pids]),
            'pass_accuracy': _minmax([stats[p]['pass_accuracy'] for p in pids]),
            'challenge_win_rate': _minmax([stats[p]['challenge_win_rate']
                                           for p in pids]),
            'distance_rate': _minmax([stats[p]['distance_per_min'] for p in pids]),
            'sprint_rate': _minmax([stats[p]['sprints_per_10min'] for p in pids]),
        }
        for i, pid in enumerate(pids):
            if pid in done:
                continue
            score = sum(w[k] * norm[k][i] for k in w) / total_w
            stats[pid]['rating'] = round(10.0 * score, 2)
            done.add(pid)
    return stats


def highlights(s):
    """Human-readable 'what they did' summary in scouting vocabulary."""
    c = s['categories']
    good_order = ['Short pass', 'Long pass', 'Through ball', 'Under pressure',
                  'Intercept', 'Challenge won', 'Won in own third', 'Pressure',
                  'Close control', 'Link up', 'Dribble']
    good = [f"{c[k]}x {k}" for k in good_order if c[k]]
    if s['sprints']:
        good.append(f"{s['sprints']}x Pace (sprint)")
    bad = [f"{c[k]}x {k}" for k in ('Failed pass', 'Challenge lost') if c[k]]
    out = "; ".join(good) if good else "no recorded actions"
    if bad:
        out += " | bad: " + "; ".join(bad)
    return out
