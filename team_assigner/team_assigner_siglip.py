"""
SigLIP embedding-based team assignment with color verification and boost pass.

Pipeline:
  1. Main cadence pass  — embed every Nth frame, collect SigLIP votes + fast color votes
  2. Boost pass         — for any track with <MIN_VOTES, embed ALL its frames so
                          short-lived tracks get as much data as long-lived ones
  3. Spatial continuity — when a new track_id starts where a recently-ended track_id
                          was, inherit the old team (fixes ByteTrack ID reassignment)
  4. Paint             — one fixed team per track, written to every frame it appears in
"""

import cv2
import numpy as np
from collections import defaultdict, deque

from .team_roster import TeamRoster
from .occlusion import compute_occluded_detections, is_occluded, occlusions_in_window


class TeamAssignerSiglip:
    def __init__(self,
                 model_name="google/siglip-base-patch16-224",
                 device=None,
                 batch_size=32,
                 vote_window=30,
                 use_umap=True):
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = SiglipVisionModel.from_pretrained(model_name).to(self.device).eval()

        self.batch_size = batch_size
        self.use_umap = use_umap
        self.kmeans = None
        self.reducer = None
        self._team_from_cluster = {0: 1, 1: 2}
        self.player_vote_history = defaultdict(lambda: deque(maxlen=vote_window))
        self.team_colors = {1: (255, 255, 255), 2: (255, 0, 0)}
        self.roster = None  # built by assign_teams_to_tracks

    # ---- crop helpers -------------------------------------------------------

    def _crop_rgb(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        h, w = crop.shape[:2]
        jersey = crop[int(0.15 * h):int(0.55 * h), int(0.20 * w):int(0.80 * w)]
        if jersey.size == 0:
            jersey = crop
        return cv2.cvtColor(jersey, cv2.COLOR_BGR2RGB)

    def _is_yellow_crop(self, crop_rgb):
        """Return True if the crop is clearly a yellow-jersey referee."""
        if crop_rgb is None or crop_rgb.size == 0:
            return False
        m = crop_rgb.mean(axis=(0, 1))
        r, g, b = float(m[0]), float(m[1]), float(m[2])
        return r > 140 and g > 140 and b < 100 and min(r, g) > b + 60

    def _is_dark_jersey(self, crop_rgb):
        """Return True if the crop is clearly a black-jersey referee (max channel < 75)."""
        if crop_rgb is None or crop_rgb.size == 0:
            return False
        m = crop_rgb.mean(axis=(0, 1))
        return float(np.max(m)) < 75

    def _is_referee_crop(self, crop_rgb):
        """Return True for any clear referee jersey (yellow or black)."""
        return self._is_yellow_crop(crop_rgb) or self._is_dark_jersey(crop_rgb)

    def _color_vote(self, crop_rgb):
        """Fast color-based vote for unambiguous jersey colors.

        Returns 1 (clearly white jersey), 2 (clearly green jersey), or None
        (ambiguous — only SigLIP vote will count for this crop).

        Uses the center of the jersey crop to avoid background contamination.
        Green threshold uses G > R+50 (not the old G > R+30 + R<120) so that
        players like pid=8 whose R≈157 still get a correct green vote while
        grass (G-R≈35-45) is safely below the 50-pt gap.
        """
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        h, w = crop_rgb.shape[:2]
        region = crop_rgb[int(0.25 * h):int(0.75 * h), int(0.25 * w):int(0.75 * w)]
        if region.size == 0:
            region = crop_rgb
        m = region.mean(axis=(0, 1))
        r, g, b = float(m[0]), float(m[1]), float(m[2])

        # Green jersey: G dominant by ≥50 pts over both R and B.
        # Grass: G-R≈35-45 — safely below threshold.
        # Mönchengladbach green: G-R≈60-80, G-B≈70-100 — fires reliably.
        if g > r + 50 and g > b + 50 and g > 80:
            return 2

        # White jersey (Frankfurt) — all channels high, very low saturation.
        if min(r, g, b) > 140 and (max(r, g, b) - min(r, g, b)) < 50:
            return 1

        return None  # ambiguous, only count SigLIP vote

    # ---- embedding ----------------------------------------------------------

    def _embed(self, crops_rgb):
        """(N, H, W, 3) uint8 RGB → (N, D) L2-normalized float32 embeddings."""
        from PIL import Image
        feats = []
        for i in range(0, len(crops_rgb), self.batch_size):
            batch = [Image.fromarray(c) for c in crops_rgb[i:i + self.batch_size]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                out = self.model(**inputs)
            feats.append(out.pooler_output.detach().cpu().numpy())
        emb = np.concatenate(feats, axis=0).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return emb / norms

    def _reduce_fit(self, emb):
        if self.use_umap:
            try:
                import umap
                self.reducer = umap.UMAP(n_components=3, random_state=0)
                return self.reducer.fit_transform(emb)
            except Exception:
                self.reducer = None
        return emb

    def _reduce_transform(self, emb):
        if self.reducer is not None:
            return self.reducer.transform(emb)
        return emb

    # ---- spatial continuity -------------------------------------------------

    def _spatial_continuity(self, player_tracks, final_team, votes_by_track):
        """Fix ByteTrack ID reassignment: if a new track_id starts very close
        (≤60 px) and very soon (≤6 frames) after a different-team track ended,
        inherit the ended track's team — but only if the predecessor has at
        least 5 votes AND at least 3 more votes than the new track.

        Conservative thresholds prevent corrupting correctly-classified tracks.
        Runs a single pass only (no chain propagation).
        """
        track_pos = {}
        track_first = {}
        track_last = {}
        for fn, player_track in enumerate(player_tracks):
            for pid, info in player_track.items():
                b = info['bbox']
                cx = (b[0] + b[2]) / 2
                cy = (b[1] + b[3]) / 2
                track_pos[(pid, fn)] = (cx, cy)
                if pid not in track_first:
                    track_first[pid] = fn
                track_last[pid] = fn

        max_gap = 6
        max_dist = 60

        changed = {}
        for pid, team in final_team.items():
            ff = track_first.get(pid)
            if ff is None or ff == 0:
                continue
            pos = track_pos.get((pid, ff))
            if pos is None:
                continue
            my_votes = len(votes_by_track.get(pid, []))

            best_pid = None
            best_dist = max_dist + 1

            for other_pid, other_team in final_team.items():
                if other_pid == pid or other_team == team:
                    continue
                ol = track_last.get(other_pid)
                if ol is None:
                    continue
                gap = ff - ol
                if not (0 < gap <= max_gap):
                    continue
                opos = track_pos.get((other_pid, ol))
                if opos is None:
                    continue
                dist = ((pos[0] - opos[0]) ** 2 + (pos[1] - opos[1]) ** 2) ** 0.5
                if dist >= best_dist:
                    continue
                pred_votes = len(votes_by_track.get(other_pid, []))
                # Only inherit from a clearly more-confident predecessor
                if pred_votes < 5 or pred_votes < my_votes + 3:
                    continue
                best_dist = dist
                best_pid = other_pid

            if best_pid is not None:
                changed[pid] = final_team[best_pid]

        for pid, team in changed.items():
            final_team[pid] = team

        n = len(changed)
        if n:
            print(f"  Spatial continuity corrected {n} track(s)")
        return final_team

    # ---- collision-swap detection -------------------------------------------

    def _refine_switch_frame(self, frames, player_tracks, pid,
                             start_fn, end_fn, team_after):
        """Pin the exact swap frame with cheap per-frame color votes.

        Votes are sampled every classify_every frames, so the flip is only
        located to within one cadence window. Scanning that window frame by
        frame finds the first frame from which every unambiguous color vote
        already agrees with the post-swap team.
        """
        votes = []
        for fn in range(start_fn, min(end_fn + 1, len(player_tracks))):
            info = player_tracks[fn].get(pid)
            if info is None:
                continue
            cv = self._color_vote(self._crop_rgb(frames[fn], info["bbox"]))
            if cv is not None:
                votes.append((fn, cv))
        for idx, (fn, cv) in enumerate(votes):
            if cv == team_after and all(v == team_after for _, v in votes[idx:]):
                return fn
        return None

    def _detect_track_switches(self, frames, player_tracks, votes_by_track,
                               occluded, final_team, referee_pids):
        """Find tracks whose votes flip teams cleanly mid-video near a collision.

        ByteTrack sometimes hands a track_id to the *other* player after two
        players collide. A single whole-track team is then wrong for half the
        video. Detection criteria (all must hold, so clean tracks are never split):
          - ≥ 6 votes, with a split point giving ≥ 3 votes on each side
          - each side ≥ 80% pure, and the two sides disagree
          - a recorded bbox collision exists between the last pre-flip vote
            and the first post-flip vote (±10 frames)

        Returns {pid: [(0, team_before), (switch_frame, team_after)]} and
        updates final_team to the post-switch team.
        """
        segments = {}
        for pid, pairs in votes_by_track.items():
            if pid in referee_pids:
                continue
            pairs = sorted(pairs)
            n = len(pairs)
            if n < 6:
                continue

            best = None
            for i in range(3, n - 2):
                left = [t for _, t in pairs[:i]]
                right = [t for _, t in pairs[i:]]
                lmaj = max(set(left), key=left.count)
                rmaj = max(set(right), key=right.count)
                if lmaj == rmaj:
                    continue
                lp = left.count(lmaj) / len(left)
                rp = right.count(rmaj) / len(right)
                if lp < 0.8 or rp < 0.8:
                    continue
                score = lp + rp
                if best is None or score > best[0]:
                    best = (score, i, lmaj, rmaj)

            if best is None:
                continue
            _, i, lmaj, rmaj = best
            last_left_fn = pairs[i - 1][0]
            first_right_fn = pairs[i][0]

            # Only split if a collision happened around the flip — pure
            # classification noise without a collision shouldn't split a track.
            occ = occlusions_in_window(occluded, pid,
                                       last_left_fn - 10, first_right_fn + 10)
            if not occ:
                continue

            # Prefer the per-frame color refinement; fall back to the first
            # frame after the collision ends.
            switch_fn = self._refine_switch_frame(
                frames, player_tracks, pid, last_left_fn, first_right_fn, rmaj)
            if switch_fn is None:
                switch_fn = occ[-1] + 1
            segments[pid] = [(0, lmaj), (switch_fn, rmaj)]
            final_team[pid] = rmaj
            print(f"  Track {pid}: collision ID swap — team {lmaj}→{rmaj} "
                  f"at frame {switch_fn}")
        return segments

    # ---- public API ---------------------------------------------------------

    def fit(self, frames, player_tracks, num_fit_frames=30):
        """Fit two team clusters on evenly-sampled frames (referees excluded).

        Crops from colliding players are skipped: an overlapped bbox contains
        both jerseys, which smears the cluster centers.
        """
        from sklearn.cluster import KMeans

        occluded = compute_occluded_detections(player_tracks)
        step = max(1, len(frames) // num_fit_frames)
        crops = []
        for fn in range(0, len(frames), step):
            if fn >= len(player_tracks):
                break
            for pid, info in player_tracks[fn].items():
                if is_occluded(occluded, pid, fn):
                    continue
                c = self._crop_rgb(frames[fn], info["bbox"])
                if c is not None and not self._is_referee_crop(c):
                    crops.append(c)
            if len(crops) >= num_fit_frames * 6:
                break

        if len(crops) < 2:
            raise RuntimeError("Not enough player crops to fit team clusters")

        emb = self._embed(crops)
        feats = self._reduce_fit(emb)
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, random_state=0)
        self.kmeans.fit(feats)

        # Anchor: brighter cluster → team 1 (white jerseys)
        labels = self.kmeans.predict(feats)
        bright_sum = {0: 0.0, 1: 0.0}
        bright_cnt = {0: 0, 1: 0}
        for crop, lbl in zip(crops, labels.tolist()):
            bright_sum[lbl] += float(np.mean(crop))
            bright_cnt[lbl] += 1
        avg_b = {c: bright_sum[c] / max(1, bright_cnt[c]) for c in (0, 1)}
        lighter = max(avg_b, key=avg_b.get)
        self._team_from_cluster = {lighter: 1, 1 - lighter: 2}

    def get_player_team(self, frame, player_bbox, player_id):
        c = self._crop_rgb(frame, player_bbox)
        if c is not None and not self._is_referee_crop(c) and self.kmeans is not None:
            emb = self._embed([c])
            feats = self._reduce_transform(emb)
            cluster = int(self.kmeans.predict(feats)[0])
            team_id = self._team_from_cluster.get(cluster, 1)
            self.player_vote_history[player_id].append(team_id)
        votes = list(self.player_vote_history[player_id])
        return max(set(votes), key=votes.count) if votes else 1

    def assign_teams_to_tracks(self, frames, player_tracks, team_colors,
                               classify_every=15):
        """Full pipeline: cadence pass → boost pass → spatial continuity → paint.

        Every track_id gets a team fixed for the whole video:
        - Main pass:  SigLIP + color vote every classify_every frames
        - Boost pass: for tracks still below MIN_VOTES, embed ALL their frames
                      so short-lived tracks get as many votes as long-lived ones
        - Spatial:    conservative ID-reassignment correction
        - Paint:      single team color written to every frame
        """
        from collections import defaultdict

        MIN_VOTES = 5       # tracks below this get the full boost pass

        # Votes are stored as (frame_num, team) so collision-swap detection
        # can find *where* in the video a track's votes flip.
        votes_by_track = defaultdict(list)
        seen = set()
        referee_pids = set()

        # Frames where a player's bbox overlaps another player's: the jersey
        # crop contains both players, so votes from these frames are skipped.
        occluded = compute_occluded_detections(player_tracks)

        # ---------------------------------------------------------------
        # Pass 1 — main cadence pass: SigLIP + color votes (clean frames only)
        # ---------------------------------------------------------------
        if self.kmeans is not None:
            for frame_num, player_track in enumerate(player_tracks):
                if not player_track:
                    continue
                on_cadence = (frame_num % classify_every == 0)
                has_unseen = any(pid not in seen for pid in player_track)
                if not (on_cadence or has_unseen):
                    continue

                ids, crops = [], []
                for pid, info in player_track.items():
                    if is_occluded(occluded, pid, frame_num):
                        continue  # contaminated crop — boost pass covers it
                    c = self._crop_rgb(frames[frame_num], info["bbox"])
                    if c is None:
                        continue
                    if self._is_referee_crop(c):
                        referee_pids.add(pid)
                        seen.add(pid)
                        continue
                    ids.append(pid)
                    crops.append(c)
                    seen.add(pid)

                if not crops:
                    continue

                feats = self._reduce_transform(self._embed(crops))
                preds = self.kmeans.predict(feats)
                for pid, pred, crop in zip(ids, preds, crops):
                    votes_by_track[pid].append(
                        (frame_num, self._team_from_cluster.get(int(pred), 1)))
                    cv = self._color_vote(crop)
                    if cv is not None:
                        votes_by_track[pid].append((frame_num, cv))

        # ---------------------------------------------------------------
        # Pass 2 — boost: embed every appearance of low-confidence tracks
        # ---------------------------------------------------------------
        pid_to_frames = defaultdict(list)
        for fn, player_track in enumerate(player_tracks):
            for pid in player_track:
                pid_to_frames[pid].append(fn)

        boost_pids = {
            pid for pid in pid_to_frames
            if len(votes_by_track.get(pid, [])) < MIN_VOTES
            and pid not in referee_pids
        }

        if boost_pids and self.kmeans is not None:
            print(f"  Boost pass: classifying all frames for "
                  f"{len(boost_pids)} low-confidence tracks …")
            all_crops, all_pids, all_fns = [], [], []
            for pid in boost_pids:
                frame_list = pid_to_frames[pid]
                # Prefer collision-free frames; if the track exists only
                # during a collision, fall back to whatever it has.
                clean = [fn for fn in frame_list
                         if not is_occluded(occluded, pid, fn)]
                if clean:
                    frame_list = clean
                # Sample evenly, up to 40 frames per track
                step = max(1, len(frame_list) // 40)
                for fn in frame_list[::step][:40]:
                    info = player_tracks[fn].get(pid)
                    if info is None:
                        continue
                    c = self._crop_rgb(frames[fn], info["bbox"])
                    if c is not None and not self._is_referee_crop(c):
                        all_crops.append(c)
                        all_pids.append(pid)
                        all_fns.append(fn)

            if all_crops:
                feats = self._reduce_transform(self._embed(all_crops))
                preds = self.kmeans.predict(feats)
                for pid, pred, crop, fn in zip(all_pids, preds, all_crops,
                                               all_fns):
                    votes_by_track[pid].append(
                        (fn, self._team_from_cluster.get(int(pred), 1)))
                    cv = self._color_vote(crop)
                    if cv is not None:
                        votes_by_track[pid].append((fn, cv))

        # ---------------------------------------------------------------
        # Compute final team per track (global majority vote)
        # ---------------------------------------------------------------
        final_team = {}
        for pid, votes in votes_by_track.items():
            teams = [t for _, t in votes]
            final_team[pid] = max(set(teams), key=teams.count) if teams else 1

        # Fallback for any pid that still has no votes (tiny single-frame track)
        for fn, player_track in enumerate(player_tracks):
            for pid, info in player_track.items():
                if pid in final_team or pid in referee_pids:
                    continue
                c = self._crop_rgb(frames[fn], info["bbox"])
                if c is not None and not self._is_referee_crop(c):
                    feat = self._reduce_transform(self._embed([c]))
                    pred = int(self.kmeans.predict(feat)[0])
                    cv = self._color_vote(c)
                    if cv is not None:
                        final_team[pid] = cv
                    else:
                        final_team[pid] = self._team_from_cluster.get(pred, 1)
                else:
                    final_team[pid] = 1
                votes_by_track[pid].append((fn, final_team[pid]))
                break

        self.final_team = final_team

        # ---------------------------------------------------------------
        # Spatial continuity — conservative ID-reassignment correction
        # ---------------------------------------------------------------
        final_team = self._spatial_continuity(player_tracks, final_team,
                                              votes_by_track)

        # ---------------------------------------------------------------
        # Collision-swap detection — split tracks whose ID changed players
        # ---------------------------------------------------------------
        segments = self._detect_track_switches(frames, player_tracks,
                                               votes_by_track, occluded,
                                               final_team, referee_pids)

        # ---------------------------------------------------------------
        # Store roster + paint one fixed team onto every frame
        # ---------------------------------------------------------------
        self.roster = TeamRoster.from_assignment(final_team, referee_pids,
                                                 segments=segments)
        self.roster.apply_to_tracks(player_tracks, team_colors)

        # Summary stats
        t1 = sum(1 for t in final_team.values() if t == 1)
        t2 = sum(1 for t in final_team.values() if t == 2)
        ref = len(referee_pids)
        print(f"  Final assignment: team1={t1}  team2={t2}  referees={ref}  "
              f"total_tracks={len(final_team)+ref}")

        return final_team
