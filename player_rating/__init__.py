"""
Player rating system built on top of the tracking pipeline output.

Consumes tracks['players'] (with team labels) and tracks['ball'] —
it never modifies them, so the existing pipeline is unaffected.

Usage:
    from player_rating import rate_players
    rate_players(tracks, fps=fps, gk_teams=gk_result, debug=True)
"""

from .possession import compute_possession
from .events import detect_events, print_events
from .rating import collect_stats, add_ratings
from .export import export_all


def rate_players(tracks, fps=25, gk_teams=None, frame_w=1920, debug=False,
                 xlsx_path='output_videos/player_ratings.xlsx',
                 csv_path='output_videos/player_ratings.csv'):
    """Run all stages and export Excel + CSV. Returns the stats dict.

    gk_teams: the goalkeeper detection result {'left': (pid, team), ...};
    it provides both the GK identities and each team's attack direction
    (a team attacks away from its own keeper's goal).
    """
    print("Player rating:")

    # Attack direction per team: defends 'left' -> attacks right (+x)
    attack_dir, gk_pids = None, ()
    if gk_teams:
        attack_dir = {}
        gk_pids = tuple(pid for pid, _ in gk_teams.values())
        for side, (pid, team) in gk_teams.items():
            attack_dir[team] = +1 if side == 'left' else -1

    possession = compute_possession(tracks['players'], tracks['ball'])
    events = detect_events(tracks['players'], tracks['ball'], possession,
                           attack_dir=attack_dir, frame_w=frame_w)
    if debug:
        print_events(events, fps=fps)
    else:
        counts = {}
        for e in events:
            counts[e['category']] = counts.get(e['category'], 0) + 1
        print("  events: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    stats = collect_stats(tracks['players'], events, fps=fps,
                          attack_dir=attack_dir, gk_pids=gk_pids)
    stats = add_ratings(stats)
    export_all(stats, events, fps=fps, xlsx_path=xlsx_path, csv_path=csv_path)
    return stats
