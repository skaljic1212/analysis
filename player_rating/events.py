"""
Stage 2 — event detection, using the scouting categorization vocabulary.

Detected categories (from "Instrukcije za kategorizaciju"):
  Short pass / Long pass     completed pass to a teammate, by ball travel
  Through ball               long forward gain toward the opponent goal
  Failed pass                possession lost to an opponent via a traveled ball
  Intercept                  the opponent who cut that pass out (standalone)
  Challenge                  close-range duel for the ball; won/lost
  Pressure                   forcing the holder into an error without taking
                             the ball yourself
  Close control              clean reception + keeping the ball under pressure
  Link up                    reception under pressure followed by a quick
                             lay-off pass
  Dribble (att 1v1)          carrying the ball 8m+ within one possession spell

Every event carries `credits`: [(pid, grade, category)] so one action can
combine categories/bonuses, per the instructions. Duel scrambles (rapid
back-and-forth possession changes) merge into ONE challenge whose winner
is the last clean holder.
"""

import numpy as np

from . import config
from .possession import possession_segments


def _ball_center(ball_frame):
    b = ball_frame.get(1, {}).get('bbox')
    if not b:
        return None
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def _team_of(player_tracks, pid, fn):
    info = player_tracks[fn].get(pid)
    return info.get('team') if info else None


def _foot_dist(info, pos):
    b = info['bbox']
    return min(np.hypot(b[0] - pos[0], b[3] - pos[1]),
               np.hypot(b[2] - pos[0], b[3] - pos[1]))


def _nearest_opponent(player_tracks, fn, pid, team, ball_pos,
                      radius=None):
    """(opponent_pid, dist) nearest to the ball, or (None, inf)."""
    radius = radius or config.DUEL_RADIUS_PX
    best, best_d = None, radius
    for opid, info in player_tracks[fn].items():
        if opid == pid or info.get('team') not in (1, 2):
            continue
        if info.get('team') == team:
            continue
        d = _foot_dist(info, ball_pos)
        if d < best_d:
            best, best_d = opid, d
    return best, best_d


def _player_center(player_tracks, pid, fn):
    info = player_tracks[fn].get(pid)
    if info is None:
        return None
    b = info['bbox']
    return ((b[0] + b[2]) / 2, b[3])


def _in_defensive_third(x, team, attack_dir, frame_w):
    """True if pixel x lies in `team`'s own defensive third."""
    if attack_dir is None or team not in attack_dir:
        return False
    if attack_dir[team] > 0:        # attacks right -> defends left third
        return x < frame_w / 3
    return x > 2 * frame_w / 3


def detect_events(player_tracks, ball_tracks, possession,
                  attack_dir=None, frame_w=1920):
    """Return a chronological list of event dicts with per-player credits."""
    events = []
    segments = possession_segments(possession)

    def ball_at(fn):
        return _ball_center(ball_tracks[fn]) if fn < len(ball_tracks) else None

    under_pressure_at = {}  # (pid, fn) -> opponent pid, cached for reuse

    def pressure_on(pid, fn):
        key = (pid, fn)
        if key not in under_pressure_at:
            bp = ball_at(fn)
            team = _team_of(player_tracks, pid, fn)
            opp = None
            if bp is not None and team in (1, 2):
                opp, _ = _nearest_opponent(player_tracks, fn, pid, team, bp)
            under_pressure_at[key] = opp
        return under_pressure_at[key]

    # ------------------------------------------------------------------
    # Transition events: passes / failed passes / challenges
    # ------------------------------------------------------------------
    raw_duels = []
    received_via_pass = {}  # segment start frame -> True (for close control)

    for (pid_a, sa, ea), (pid_b, sb, eb) in zip(segments, segments[1:]):
        if pid_a == pid_b or sb - ea > config.MAX_EVENT_GAP_FRAMES:
            continue
        team_a = _team_of(player_tracks, pid_a, ea)
        team_b = _team_of(player_tracks, pid_b, sb)
        if team_a is None or team_b is None:
            continue
        a_pos, b_pos = ball_at(ea), ball_at(sb)
        travel = (float(np.hypot(b_pos[0] - a_pos[0], b_pos[1] - a_pos[1]))
                  if a_pos and b_pos else 0.0)

        if team_a == team_b:
            # ---- completed pass: classify short / long / through ball ----
            is_long = travel >= config.LONG_PASS_PX
            gain = 0.0
            if attack_dir and team_a in attack_dir and a_pos and b_pos:
                gain = (b_pos[0] - a_pos[0]) * attack_dir[team_a]
            categories = []
            if gain >= config.THROUGH_BALL_GAIN_PX:
                categories.append(('through_ball', 'Through ball'))
            elif is_long:
                categories.append(('long_pass', 'Long pass'))
            else:
                categories.append(('short_pass', 'Short pass'))
            credits = [(pid_a, config.GRADES[categories[0][0]], categories[0][1])]
            presser = pressure_on(pid_a, ea)
            if presser is not None:
                credits.append((pid_a, config.GRADES['pass_pressure_bonus'],
                                'Under pressure'))
            events.append({
                'type': 'pass', 'category': categories[0][1],
                'frame': ea, 'end_frame': sb,
                'from': pid_a, 'to': pid_b, 'team': team_a,
                'credits': credits,
                'detail': f'{categories[0][1]} {pid_a} -> {pid_b} '
                          f'({travel:.0f}px, gain {gain:+.0f}px'
                          + (', under pressure)' if presser is not None else ')'),
            })
            received_via_pass[sb] = True

        elif travel >= config.MIN_PASS_TRAVEL_PX:
            # ---- failed pass + standalone Intercept for the opponent ----
            is_long = travel >= config.LONG_PASS_PX
            key = 'failed_long_pass' if is_long else 'failed_pass'
            credits = [(pid_a, config.GRADES[key], 'Failed pass'),
                       (pid_b, config.GRADES['intercept'], 'Intercept')]
            presser = pressure_on(pid_a, ea)
            if presser is not None and presser != pid_b:
                credits.append((presser, config.GRADES['pressure'], 'Pressure'))
            events.append({
                'type': 'failed_pass', 'category': 'Failed pass',
                'frame': ea, 'end_frame': sb,
                'from': pid_a, 'to': pid_b, 'team': team_a,
                'credits': credits,
                'detail': f'failed {"long " if is_long else ""}pass by {pid_a}, '
                          f'Intercept by {pid_b} ({travel:.0f}px)',
            })

        else:
            # ---- close-range turnover: raw Challenge (merged below) ----
            raw_duels.append({'frame': ea, 'end_frame': sb,
                              'winner': pid_b, 'loser': pid_a,
                              'team': team_b})

    # ------------------------------------------------------------------
    # Merge duel scrambles: rapid chains involving the same players are
    # ONE challenge whose winner is the last clean holder.
    # ------------------------------------------------------------------
    merged = []
    for d in raw_duels:
        if merged and d['frame'] - merged[-1][-1]['end_frame'] <= \
                config.SCRAMBLE_WINDOW_FRAMES and \
                {d['winner'], d['loser']} & \
                {merged[-1][-1]['winner'], merged[-1][-1]['loser']}:
            merged[-1].append(d)
        else:
            merged.append([d])

    scramble_spans = []
    for group in merged:
        first, last = group[0], group[-1]
        winner = last['winner']
        # the main opponent: most frequent other participant
        others = [x for d in group for x in (d['winner'], d['loser'])
                  if x != winner]
        loser = max(set(others), key=others.count) if others else first['loser']
        win_team = _team_of(player_tracks, winner, last['end_frame'])
        credits = [(winner, config.GRADES['challenge_won'], 'Challenge won'),
                   (loser, config.GRADES['challenge_lost'], 'Challenge lost')]
        bp = ball_at(last['end_frame'])
        zone = ''
        if bp is not None and _in_defensive_third(bp[0], win_team,
                                                  attack_dir, frame_w):
            credits.append((winner, config.GRADES['defensive_zone_bonus'],
                            'Won in own third'))
            zone = ', in own third'
        n = len(group)
        events.append({
            'type': 'challenge', 'category': 'Challenge',
            'frame': first['frame'], 'end_frame': last['end_frame'],
            'winner': winner, 'loser': loser, 'team': win_team,
            'credits': credits,
            'detail': f'challenge: {winner} won ball off {loser}'
                      + (f' (scramble of {n} exchanges)' if n > 1 else '')
                      + zone,
        })
        scramble_spans.append((first['frame'], last['end_frame']))

    def in_scramble(fn):
        return any(s - config.DUEL_MIN_FRAMES <= fn <= e + config.DUEL_MIN_FRAMES
                   for s, e in scramble_spans)

    # ------------------------------------------------------------------
    # Per-segment events: Dribble, Close control, Link up,
    # and survived-pressure challenges
    # ------------------------------------------------------------------
    pass_from = {}  # pid -> sorted frames where pid completed a pass
    for e in events:
        if e['type'] == 'pass':
            pass_from.setdefault(e['from'], []).append(e['frame'])

    for pid, s, e_fn in segments:
        team = _team_of(player_tracks, pid, s)
        if team not in (1, 2):
            continue

        # ---- Dribble / att 1v1: carrying the ball far in one spell ----
        if e_fn - s >= config.DRIBBLE_MIN_FRAMES:
            p0 = _player_center(player_tracks, pid, s)
            p1 = _player_center(player_tracks, pid, e_fn)
            if p0 and p1:
                carry = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                if carry >= config.DRIBBLE_MIN_PX:
                    events.append({
                        'type': 'dribble', 'category': 'Dribble',
                        'frame': s, 'end_frame': e_fn,
                        'from': pid, 'team': team,
                        'credits': [(pid, config.GRADES['dribble'], 'Dribble')],
                        'detail': f'dribble/carry by {pid} '
                                  f'({carry * config.PX_TO_METERS:.0f}m)',
                    })

        # ---- Close control / Link up: reception under pressure ----
        if received_via_pass.get(s) and not in_scramble(s):
            bp = ball_at(s)
            if bp is None:
                continue
            opp, _ = _nearest_opponent(player_tracks, s, pid, team, bp)
            if opp is None:
                continue
            laid_off = any(s < f <= s + config.LINKUP_MAX_FRAMES
                           for f in pass_from.get(pid, []))
            kept = e_fn - s >= config.CLOSE_CONTROL_KEEP_FRAMES
            if laid_off:
                events.append({
                    'type': 'link_up', 'category': 'Link up',
                    'frame': s, 'end_frame': e_fn,
                    'from': pid, 'team': team,
                    'credits': [(pid, config.GRADES['link_up'], 'Link up')],
                    'detail': f'link up: {pid} received under pressure from '
                              f'{opp} and laid the ball off',
                })
            elif kept:
                events.append({
                    'type': 'close_control', 'category': 'Close control',
                    'frame': s, 'end_frame': e_fn,
                    'from': pid, 'team': team,
                    'credits': [(pid, config.GRADES['close_control'],
                                 'Close control')],
                    'detail': f'close control: {pid} kept the ball under '
                              f'pressure from {opp}',
                })

    events.sort(key=lambda ev: ev['frame'])
    return events


def print_events(events, fps=25):
    print(f"  {len(events)} events detected:")
    for e in events:
        t = e['frame'] / fps
        grades = " ".join(f"[{c}: {g:+.1f} {pid}]" for pid, g, c in e['credits'])
        print(f"    f{e['frame']:>4} ({t:5.1f}s)  {e['type']:<14} "
              f"{e['detail']}  {grades}")
