"""
Persistent player→team roster.

Stores the final team of every player track (including referees as team 0)
so that:
  - team membership is queryable (`players_in_team`, `teams`)
  - the mapping survives between runs (`save` / `load` as JSON)
  - reruns can paint teams onto tracks directly (`apply_to_tracks`)
    without re-running SigLIP/KMeans classification
"""

import json
import os

REFEREE_TEAM = 0
REFEREE_COLOR = (0, 255, 255)  # cyan, matches referee ellipse color


class TeamRoster:
    def __init__(self, player_teams=None, player_segments=None):
        # player_teams: {track_id (int): team (int)}; team 0 = referee
        self.player_teams = {int(p): int(t) for p, t in (player_teams or {}).items()}
        # player_segments: {track_id: [(start_frame, team), ...]} for tracks whose
        # ByteTrack ID was handed to a different player mid-video (collision swap).
        # Sorted by start_frame; overrides player_teams when a frame is given.
        self.player_segments = {
            int(p): sorted((int(s), int(t)) for s, t in segs)
            for p, segs in (player_segments or {}).items()
        }

    # ---- building -----------------------------------------------------------

    @classmethod
    def from_assignment(cls, final_team, referee_pids=(), segments=None):
        """Build a roster from an assigner's final_team dict + referee set.

        segments: optional {pid: [(start_frame, team), ...]} for tracks that
        switch teams mid-video; their player_teams entry becomes the team of
        the last segment.
        """
        roster = cls(final_team, segments)
        for pid, segs in roster.player_segments.items():
            roster.player_teams[pid] = segs[-1][1]
        for pid in referee_pids:
            roster.player_teams[int(pid)] = REFEREE_TEAM
            roster.player_segments.pop(int(pid), None)
        return roster

    def set_team(self, player_id, team):
        self.player_teams[int(player_id)] = int(team)
        self.player_segments.pop(int(player_id), None)

    # ---- queries ------------------------------------------------------------

    def get_team(self, player_id, frame_num=None, default=1):
        """Team of a player; frame-aware for tracks with a mid-video switch."""
        pid = int(player_id)
        segs = self.player_segments.get(pid)
        if segs and frame_num is not None:
            team = segs[0][1]
            for start, t in segs:
                if frame_num >= start:
                    team = t
                else:
                    break
            return team
        return self.player_teams.get(pid, default)

    def players_in_team(self, team):
        """Sorted list of track_ids belonging to a team (0 = referees)."""
        return sorted(p for p, t in self.player_teams.items() if t == int(team))

    def teams(self):
        """{team: sorted [track_ids]} for every team present in the roster."""
        grouped = {}
        for pid, team in self.player_teams.items():
            grouped.setdefault(team, []).append(pid)
        return {t: sorted(pids) for t, pids in sorted(grouped.items())}

    def covers(self, player_ids):
        """True if every given track_id has a stored team."""
        return all(int(pid) in self.player_teams for pid in player_ids)

    def summary(self):
        grouped = self.teams()
        parts = []
        for team, pids in grouped.items():
            label = "referees" if team == REFEREE_TEAM else f"team {team}"
            parts.append(f"{label}: {len(pids)} players")
        return " | ".join(parts) if parts else "empty roster"

    # ---- persistence --------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # JSON keys are strings; grouped layout keeps the file human-readable
        payload = {
            "teams": {str(t): pids for t, pids in self.teams().items()},
        }
        if self.player_segments:
            payload["switches"] = {
                str(p): [[s, t] for s, t in segs]
                for p, segs in self.player_segments.items()
            }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path):
        """Load a roster from JSON; returns None if missing or unreadable."""
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
            player_teams = {}
            for team, pids in payload["teams"].items():
                for pid in pids:
                    player_teams[int(pid)] = int(team)
            return cls(player_teams, payload.get("switches"))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return None

    # ---- painting -----------------------------------------------------------

    def apply_to_tracks(self, player_tracks, team_colors):
        """Write team/team_color onto every frame each track appears in.

        Frame-aware: a track with a recorded mid-video switch is painted with
        its pre-switch team before the switch frame and post-switch team after.
        """
        for frame_num, player_track in enumerate(player_tracks):
            for pid, info in player_track.items():
                team = self.get_team(pid, frame_num)
                if team == REFEREE_TEAM:
                    info["team"] = REFEREE_TEAM
                    info["team_color"] = REFEREE_COLOR
                else:
                    info["team"] = team
                    info["team_color"] = team_colors[team]
