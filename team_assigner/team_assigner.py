import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict, deque

from .team_roster import TeamRoster
from .occlusion import compute_occluded_detections, is_occluded


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self._team_from_cluster = {0: 1, 1: 2}  # overridden after fit
        self.player_vote_history = defaultdict(lambda: deque(maxlen=30))
        self.roster = None  # built by assign_teams_to_tracks

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(image_2d)
        return kmeans

    def _is_yellow_jersey(self, color_bgr):
        """Return True if a BGR jersey color is clearly referee-yellow."""
        b, g, r = float(color_bgr[0]), float(color_bgr[1]), float(color_bgr[2])
        return r > 140 and g > 140 and b < 100 and min(r, g) > b + 60

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if image.size == 0:
            return np.array([128.0, 128.0, 128.0])
        h, w = image.shape[:2]

        # Tight jersey region: upper chest (15-45% height, central 25-75% width).
        region = image[int(0.15 * h):int(0.45 * h), int(0.25 * w):int(0.75 * w)]
        if region.size < 12:
            region = image

        # Run intra-crop KMeans to find the two dominant colours (jersey vs background).
        # The corner pixels are almost always pitch grass → background cluster.
        # The other cluster is the jersey colour, regardless of whether it's white or green.
        pixels = region.reshape(-1, 3).astype(np.float32)
        km = KMeans(n_clusters=2, init="k-means++", n_init=3, random_state=0)
        km.fit(pixels)

        rh, rw = region.shape[:2]  # noqa: used for corner indexing
        corners = np.array([
            region[0, 0], region[0, -1], region[-1, 0], region[-1, -1]
        ], dtype=np.float32)
        corner_labels = km.predict(corners)
        bg_cluster = int(np.bincount(corner_labels).argmax())

        return km.cluster_centers_[1 - bg_cluster].astype(float)

    def assign_team_color(self, frames, player_tracks_list, num_fit_frames=30):
        """Fit KMeans team color centroids by sampling frames evenly across the video.

        Sampling many frames instead of only frame 0 produces far more stable
        cluster centers under varying lighting and player compositions.
        Yellow-jersey players (referees) are excluded from fitting.
        """
        occluded = compute_occluded_detections(player_tracks_list)
        step = max(1, len(frames) // num_fit_frames)
        player_colors = []

        for fn in range(0, len(frames), step):
            if fn >= len(player_tracks_list):
                break
            player_detections = player_tracks_list[fn]
            if not player_detections:
                continue
            heights = [d["bbox"][3] - d["bbox"][1] for d in player_detections.values()]
            min_height = (max(heights) * 0.4) if heights else 0
            for pid, player_detection in player_detections.items():
                if is_occluded(occluded, pid, fn):
                    continue  # overlapping bbox → crop contains both jerseys
                bbox = player_detection["bbox"]
                if (bbox[3] - bbox[1]) >= min_height:
                    color = self.get_player_color(frames[fn], bbox)
                    if not self._is_yellow_jersey(color):
                        player_colors.append(color)
            if len(player_colors) >= num_fit_frames * 6:
                break

        if len(player_colors) < 2:
            player_colors = [
                self.get_player_color(frames[0], d["bbox"])
                for d in player_tracks_list[0].values()
            ]

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        kmeans.fit(player_colors)
        self.kmeans = kmeans

        # Anchor team labels so they never depend on KMeans internal ordering:
        # team 1 = the brighter (white-jersey) cluster,
        # team 2 = the darker (coloured-jersey) cluster.
        centers = kmeans.cluster_centers_
        brightness = [np.mean(c) for c in centers]
        lighter = int(brightness[1] > brightness[0])
        self._team_from_cluster = {lighter: 1, 1 - lighter: 2}
        self.team_colors[1] = centers[lighter]
        self.team_colors[2] = centers[1 - lighter]

    def get_player_team(self, frame, player_bbox, player_id):
        player_color = self.get_player_color(frame, player_bbox)
        cluster = int(self.kmeans.predict(player_color.reshape(1, -1))[0])
        team_id = self._team_from_cluster[cluster]

        self.player_vote_history[player_id].append(team_id)
        votes = list(self.player_vote_history[player_id])
        return max(set(votes), key=votes.count)

    def assign_teams_to_tracks(self, frames, player_tracks, team_colors, classify_every=1):
        """Assign ONE team per track for the whole video (no per-frame flicker).

        Every track_id is guaranteed at least one classification via `seen` tracking.
        Yellow-jersey players (referees in player dict) get cyan, not a team color.
        """
        votes_by_track = defaultdict(list)
        seen = set()
        referee_pids = set()
        occluded = compute_occluded_detections(player_tracks)

        for frame_num, player_track in enumerate(player_tracks):
            if not player_track:
                continue
            on_cadence = (frame_num % classify_every == 0)
            has_unseen = any(pid not in seen for pid in player_track)
            if not (on_cadence or has_unseen):
                continue
            for pid, info in player_track.items():
                if is_occluded(occluded, pid, frame_num):
                    continue  # contaminated crop; the safety pass covers
                              # tracks that never get a clean frame
                if on_cadence or pid not in seen:
                    color = self.get_player_color(frames[frame_num], info["bbox"])
                    if self._is_yellow_jersey(color):
                        referee_pids.add(pid)
                        seen.add(pid)
                        continue
                    cluster = int(self.kmeans.predict(color.reshape(1, -1))[0])
                    votes_by_track[pid].append(self._team_from_cluster[cluster])
                    seen.add(pid)

        final_team = {pid: max(set(v), key=v.count) for pid, v in votes_by_track.items()}

        # Safety: classify any pid still missing
        for frame_num, player_track in enumerate(player_tracks):
            for pid, info in player_track.items():
                if pid not in final_team and pid not in referee_pids:
                    color = self.get_player_color(frames[frame_num], info["bbox"])
                    if self._is_yellow_jersey(color):
                        referee_pids.add(pid)
                    else:
                        cluster = int(self.kmeans.predict(color.reshape(1, -1))[0])
                        final_team[pid] = self._team_from_cluster[cluster]

        self.roster = TeamRoster.from_assignment(final_team, referee_pids)
        self.roster.apply_to_tracks(player_tracks, team_colors)

        return final_team
