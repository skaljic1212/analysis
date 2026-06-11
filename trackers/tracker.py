from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path, ball_model_path='models/ball.pt'):
        self.model = YOLO(model_path)
        # Optional dedicated ball-only detector (see ball_only_dataset/
        # README_TRAINING.md). When present it powers the ball-candidate
        # searches; players/teams/referees stay on the main model.
        self.ball_model = None
        if ball_model_path and os.path.exists(ball_model_path):
            self.ball_model = YOLO(ball_model_path)
            print(f"Dedicated ball detector loaded: {ball_model_path}")

    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
    
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            # imgsz=1280 improves recall for small/distant players; costs ~30% more inference time vs default 640
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, iou=0.5, imgsz=1280)
            detections += detections_batch
        return detections

    def get_ball_candidates(self, frames, read_from_stub=False, stub_path=None,
                            conf=0.05, max_per_frame=8):
        """ALL ball detections per frame (not just the best one).

        The main track stub keeps only the highest-confidence ball per frame,
        which discards the real ball whenever a white marking (penalty spot)
        scores higher. Trajectory selection needs the full candidate list.
        Returns [[{'bbox': [...], 'conf': float}, ...], ...] per frame.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        candidates = []
        batch_size = 20
        detector = self.ball_model or self.model
        for i in range(0, len(frames), batch_size):
            batch = detector.predict(frames[i:i+batch_size], conf=conf,
                                     iou=0.5, imgsz=1280)
            for det in batch:
                names_inv = {v: k for k, v in det.names.items()}
                ball_cls = names_inv['ball']
                frame_cands = []
                for box in det.boxes:
                    if int(box.cls) == ball_cls:
                        frame_cands.append({'bbox': box.xyxy[0].tolist(),
                                            'conf': float(box.conf)})
                frame_cands.sort(key=lambda c: -c['conf'])
                candidates.append(frame_cands[:max_per_frame])

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(candidates, f)
        return candidates

    def select_ball_trajectory(self, candidates, player_tracks,
                               camera_movement=None,
                               frame_w=1920, frame_h=1080):
        """Pick one physically consistent ball path through all candidates.

        1. Drop frame-edge candidates (corner flags, broadcast graphics).
        2. Blacklist recurring static *pitch* locations (painted markings):
           a 24px pitch-coordinate bin hit ≥20 times over a ≥150-frame span is
           a marking — unless its detections sit at a player's feet, which is
           a parked ball, not paint.
        3. Dynamic programming: choose the candidate chain maximizing
           confidence while only allowing plausible ball speed between picks.
        4. Interpolate the remaining gaps.
        """
        n = len(candidates)
        if camera_movement is not None:
            cum = np.cumsum(np.array(camera_movement, dtype=float), axis=0)
        else:
            cum = np.zeros((n, 2))

        # ---- collect nodes (frame, cx, cy, conf, bbox) ----------------------
        edge = 60
        nodes = []
        for fn, cands in enumerate(candidates[:n]):
            for c in cands:
                x1, y1, x2, y2 = c['bbox']
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if cx < edge or cx > frame_w - edge or cy < edge or cy > frame_h - edge:
                    continue
                nodes.append([fn, cx, cy, c['conf'], c['bbox']])

        # ---- blacklist recurring static pitch locations ---------------------
        def near_player_foot(fn, cx, cy, radius=60):
            if fn >= len(player_tracks):
                return False
            for info in player_tracks[fn].values():
                b = info['bbox']
                fx, fy = (b[0] + b[2]) / 2, b[3]
                if abs(fx - cx) <= radius and abs(fy - cy) <= radius:
                    return True
            return False

        bins = {}
        for idx, (fn, cx, cy, conf, bb) in enumerate(nodes):
            px, py = cx + cum[fn][0], cy + cum[fn][1]
            bins.setdefault((int(px // 24), int(py // 24)), []).append(idx)

        drop = set()
        for key, idxs in bins.items():
            fns = [nodes[i][0] for i in idxs]
            if len(idxs) >= 20 and (max(fns) - min(fns)) >= 150:
                near = sum(1 for i in idxs
                           if near_player_foot(nodes[i][0], nodes[i][1], nodes[i][2]))
                if near / len(idxs) < 0.5:
                    drop.update(idxs)
        if drop:
            print(f"  Ball: blacklisted {len(drop)} detections on "
                  f"{len({(int((nodes[i][1]+cum[nodes[i][0]][0])//24)) for i in drop})} static marking(s)")
        nodes = [nd for i, nd in enumerate(nodes) if i not in drop]
        nodes.sort(key=lambda nd: nd[0])

        # ---- DP chain selection ---------------------------------------------
        # score(j) = best score of a chain ending at node j.
        # Linking j to an earlier node i is free within the speed budget
        # (25 px/frame + 30 slack) and increasingly expensive beyond it;
        # impossible links (>2x budget + 200) are forbidden.
        CONF_BONUS = 40.0
        MAX_LINK_GAP = 60
        score = [nd[3] * CONF_BONUS for nd in nodes]
        prev = [-1] * len(nodes)
        for j in range(len(nodes)):
            fj, xj, yj = nodes[j][0], nodes[j][1], nodes[j][2]
            for i in range(j - 1, -1, -1):
                fi = nodes[i][0]
                gap = fj - fi
                if gap == 0:
                    continue
                if gap > MAX_LINK_GAP:
                    break
                d = float(np.hypot(xj - nodes[i][1], yj - nodes[i][2]))
                allowed = 25.0 * gap + 30.0
                if d > 2 * allowed + 200:
                    continue
                penalty = max(0.0, d - allowed) * 0.5
                cand = score[i] + nodes[j][3] * CONF_BONUS - penalty
                if cand > score[j]:
                    score[j] = cand
                    prev[j] = i

        chosen = {}
        if nodes:
            j = int(np.argmax(score))
            while j != -1:
                fn = nodes[j][0]
                if fn not in chosen:  # keep at most one candidate per frame
                    chosen[fn] = nodes[j][4]
                j = prev[j]
        print(f"  Ball: trajectory uses {len(chosen)}/{n} real detections")

        # ---- interpolate gaps + light smoothing -------------------------------
        rows = [chosen.get(fn, [np.nan] * 4) for fn in range(n)]
        df = pd.DataFrame(rows, columns=['x1', 'y1', 'x2', 'y2'])

        # Motion-aware gap filling: PCHIP continues each gap with the
        # velocity the ball had at the gap edges (a moving ball keeps its
        # momentum), instead of a straight line between the endpoints.
        # Box size doesn't need physics — plain linear is fine there.
        cx = (df['x1'] + df['x2']) / 2
        cy = (df['y1'] + df['y2']) / 2
        w = (df['x2'] - df['x1']).interpolate().bfill().ffill()
        h = (df['y2'] - df['y1']).interpolate().bfill().ffill()
        try:
            cx = cx.interpolate(method='pchip').bfill().ffill()
            cy = cy.interpolate(method='pchip').bfill().ffill()
        except (ValueError, ImportError):
            cx = cx.interpolate().bfill().ffill()
            cy = cy.interpolate().bfill().ffill()
        # A ball that disappears near players and reappears near players was
        # hidden between their legs the whole time — it doesn't glide across
        # open grass. For such gaps, pull the interpolated position toward
        # the nearest player's feet.
        def nearest_foot(fn, x, y, radius):
            best, best_d = None, radius
            for info in player_tracks[fn].values():
                b = info['bbox']
                fx, fy = (b[0] + b[2]) / 2, b[3]
                d = float(np.hypot(fx - x, fy - y))
                if d < best_d:
                    best, best_d = (fx, fy), d
            return best

        fn0 = 0
        while fn0 < n:
            if fn0 in chosen:
                fn0 += 1
                continue
            fn1 = fn0
            while fn1 < n and fn1 not in chosen:
                fn1 += 1
            # gap is [fn0, fn1); check both endpoints sit near a player
            ends_near = all(
                e is None or nearest_foot(e, cx[e], cy[e], 110) is not None
                for e in ((fn0 - 1 if fn0 > 0 else None),
                          (fn1 if fn1 < n else None)))
            if ends_near:
                for fn in range(fn0, fn1):
                    foot = nearest_foot(fn, cx[fn], cy[fn], 130)
                    if foot is not None:
                        cx[fn] = 0.3 * cx[fn] + 0.7 * foot[0]
                        cy[fn] = 0.3 * cy[fn] + 0.7 * (foot[1] - 8)
            fn0 = fn1

        df['x1'], df['x2'] = cx - w / 2, cx + w / 2
        df['y1'], df['y2'] = cy - h / 2, cy + h / 2

        # Smooth the center (window 5) and the box size (window 9) so the
        # marker doesn't jitter with per-frame detection noise.
        cx_s = ((df['x1'] + df['x2']) / 2).rolling(5, center=True, min_periods=1).mean()
        cy_s = ((df['y1'] + df['y2']) / 2).rolling(5, center=True, min_periods=1).mean()
        w_s = (df['x2'] - df['x1']).rolling(9, center=True, min_periods=1).mean()
        h_s = (df['y2'] - df['y1']).rolling(9, center=True, min_periods=1).mean()
        out = []
        for fn in range(n):
            bbox = [float(cx_s[fn] - w_s[fn] / 2), float(cy_s[fn] - h_s[fn] / 2),
                    float(cx_s[fn] + w_s[fn] / 2), float(cy_s[fn] + h_s[fn] / 2)]
            out.append({1: {"bbox": bbox, "detected": fn in chosen}})
        return out

    def detect_ball_in_gaps(self, frames, candidates, ball_positions,
                            stub_path=None, crop_radius=240, conf=0.02):
        """Recover the ball in interpolated gaps via zoomed re-detection.

        For every frame where the trajectory had no real detection, crop a
        window around the interpolated position and re-run the detector on
        the zoomed crop at a low threshold — far more sensitive to a small
        blurred ball than the full-frame pass. Found detections are added
        to the candidate list; the caller re-runs trajectory selection so
        every recovered detection still has to fit a physically consistent
        path. Results are cached in stub_path.
        """
        gap_fns = [fn for fn in range(len(ball_positions))
                   if not ball_positions[fn][1].get('detected', True)]
        if not gap_fns:
            return candidates

        # Search radius grows with distance from the nearest real detection:
        # the further into a gap, the less certain the motion prediction.
        detected_fns = sorted(fn for fn in range(len(ball_positions))
                              if ball_positions[fn][1].get('detected', True))

        def radius_for(fn):
            if not detected_fns:
                return crop_radius
            dist = min(abs(fn - d) for d in detected_fns)
            # Cap at 320 so the 640px crop is never downscaled by the
            # detector — zoom is the whole point of this pass.
            return int(min(320, crop_radius + 7 * dist))

        # Cache keyed by (frame, quantized crop origin): when a better motion
        # prediction moves the search window, the crop is re-run; otherwise
        # the cached result is reused (so iterating this step stays cheap).
        cache = {}
        if stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                cache = pickle.load(f)
            if not all(isinstance(k, tuple) for k in cache):
                cache = {}  # old cache format — discard

        h, w = frames[0].shape[:2]
        found = {}
        batch, metas = [], []

        def flush():
            if not batch:
                return
            detector = self.ball_model or self.model
            results = detector.predict(batch, conf=conf, iou=0.5,
                                       imgsz=640, verbose=False)
            for det, (fn, x0, y0, key) in zip(results, metas):
                names_inv = {v: k for k, v in det.names.items()}
                ball_cls = names_inv['ball']
                cands = []
                for box in det.boxes:
                    if int(box.cls) != ball_cls:
                        continue
                    bx = box.xyxy[0].tolist()
                    cands.append({'bbox': [bx[0] + x0, bx[1] + y0,
                                           bx[2] + x0, bx[3] + y0],
                                  'conf': float(box.conf)})
                cands.sort(key=lambda c: -c['conf'])
                cache[key] = cands[:3]
                found[fn] = cache[key]
            batch.clear()
            metas.clear()

        for fn in gap_fns:
            b = ball_positions[fn][1]['bbox']
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            r = radius_for(fn)
            x0 = int(max(0, min(w - 2 * r, cx - r)))
            y0 = int(max(0, min(h - 2 * r, cy - r)))
            key = (fn, x0 // 64, y0 // 64, r // 64)
            if key in cache:
                found[fn] = cache[key]
                continue
            batch.append(frames[fn][y0:y0 + 2 * r, x0:x0 + 2 * r].copy())
            metas.append((fn, x0, y0, key))
            if len(batch) >= 16:
                flush()
        flush()
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(cache, f)

        recovered = sum(1 for v in found.values() if v)
        print(f"  Ball: zoomed re-detection recovered candidates in "
              f"{recovered}/{len(gap_fns)} gap frames")
        augmented = [list(c) for c in candidates]
        for fn, cands in found.items():
            augmented[fn].extend(cands)
        return augmented

    def track_ball_through_gaps(self, frames, candidates, ball_positions,
                                search=110, min_score=0.55):
        """Follow the ball through detection gaps with template matching.

        The ball is a small distinctive white blob; normalized cross-
        correlation can often follow it through frames where the neural
        detector fails (motion blur, partial occlusion). Each gap is tracked
        forward from its last real detection and backward from its next one;
        a frame is only kept where both directions agree (within 30 px), so
        a track that drifts onto a sock or line mark gets discarded. Matches
        are added as low-confidence candidates — the trajectory selector
        still has the final word.
        """
        n = len(ball_positions)
        detected = [ball_positions[fn][1].get('detected', False) for fn in range(n)]

        def gray_crop(fn, x0, y0, x1, y1):
            x0, y0 = max(0, int(x0)), max(0, int(y0))
            x1, y1 = min(frames[fn].shape[1], int(x1)), min(frames[fn].shape[0], int(y1))
            if x1 - x0 < 4 or y1 - y0 < 4:
                return None
            return cv2.cvtColor(frames[fn][y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)

        def track(start_fn, gap_fns):
            """Track from the real detection at start_fn across gap_fns."""
            b = ball_positions[start_fn][1]['bbox']
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            bw = max(8.0, b[2] - b[0]); bh = max(8.0, b[3] - b[1])
            tpl = gray_crop(start_fn, cx - bw / 2 - 2, cy - bh / 2 - 2,
                            cx + bw / 2 + 2, cy + bh / 2 + 2)
            out = {}
            if tpl is None:
                return out
            th, tw = tpl.shape
            px, py = cx, cy
            for fn in gap_fns:
                win = gray_crop(fn, px - search, py - search,
                                px + search, py + search)
                if win is None or win.shape[0] < th or win.shape[1] < tw:
                    break
                res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
                _, maxv, _, maxloc = cv2.minMaxLoc(res)
                if maxv < min_score:
                    break
                wx0 = max(0, int(px - search))
                wy0 = max(0, int(py - search))
                px = wx0 + maxloc[0] + tw / 2
                py = wy0 + maxloc[1] + th / 2
                out[fn] = (px, py, float(maxv), bw, bh)
            return out

        added = 0
        augmented = [list(c) for c in candidates]
        fn0 = 0
        while fn0 < n:
            if detected[fn0]:
                fn0 += 1
                continue
            fn1 = fn0
            while fn1 < n and not detected[fn1]:
                fn1 += 1
            gap = list(range(fn0, fn1))
            fwd = track(fn0 - 1, gap) if fn0 > 0 else {}
            bwd = track(fn1, gap[::-1]) if fn1 < n else {}
            for fn in gap:
                f, bk = fwd.get(fn), bwd.get(fn)
                if f and bk and np.hypot(f[0] - bk[0], f[1] - bk[1]) <= 30:
                    x = (f[0] + bk[0]) / 2
                    y = (f[1] + bk[1]) / 2
                    bw, bh = f[3], f[4]
                    augmented[fn].append({
                        'bbox': [x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2],
                        'conf': 0.3 * min(f[2], bk[2])})
                    added += 1
            fn0 = fn1
        print(f"  Ball: template tracking filled {added} gap frames "
              f"(forward/backward agreement)")
        return augmented

    def detect_ball_tiled(self, frames, candidates, ball_positions,
                          stub_path=None, conf=0.03):
        """Native-resolution tiled detection for frames still missing the ball.

        Splits each missing frame into a 3x2 grid of overlapping tiles run at
        their natural scale — maximum recall, no dependence on a predicted
        search window. Cached per frame in stub_path.
        """
        gap_fns = [fn for fn in range(len(ball_positions))
                   if not ball_positions[fn][1].get('detected', True)]
        if not gap_fns:
            return candidates

        cache = {}
        if stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                cache = pickle.load(f)

        h, w = frames[0].shape[:2]
        cols, rows, overlap = 3, 2, 80
        tw = (w + (cols - 1) * overlap) // cols
        th = (h + (rows - 1) * overlap) // rows

        todo = [fn for fn in gap_fns if fn not in cache]
        for fn in todo:
            tiles, origins = [], []
            for r in range(rows):
                for c in range(cols):
                    x0 = min(c * (tw - overlap), w - tw)
                    y0 = min(r * (th - overlap), h - th)
                    tiles.append(frames[fn][y0:y0 + th, x0:x0 + tw].copy())
                    origins.append((x0, y0))
            detector = self.ball_model or self.model
            results = detector.predict(tiles, conf=conf, iou=0.5,
                                       imgsz=640, verbose=False)
            cands = []
            for det, (x0, y0) in zip(results, origins):
                names_inv = {v: k for k, v in det.names.items()}
                ball_cls = names_inv['ball']
                for box in det.boxes:
                    if int(box.cls) == ball_cls:
                        bx = box.xyxy[0].tolist()
                        cands.append({'bbox': [bx[0] + x0, bx[1] + y0,
                                               bx[2] + x0, bx[3] + y0],
                                      'conf': float(box.conf)})
            cands.sort(key=lambda c: -c['conf'])
            cache[fn] = cands[:4]
        if todo and stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(cache, f)

        recovered = sum(1 for fn in gap_fns if cache.get(fn))
        print(f"  Ball: tiled detection found candidates in "
              f"{recovered}/{len(gap_fns)} remaining gap frames")
        augmented = [list(c) for c in candidates]
        for fn in gap_fns:
            augmented[fn].extend(cache.get(fn, []))
        return augmented

    def interpolate_player_positions(self, player_tracks, max_gap=10):
        """Fill interior gaps of ≤max_gap consecutive missing frames per player track.
        Filled entries are marked interpolated=True so callers can distinguish them."""
        all_player_ids = set()
        for frame_data in player_tracks:
            all_player_ids.update(frame_data.keys())

        for player_id in all_player_ids:
            bboxes = []
            for frame_data in player_tracks:
                if player_id in frame_data:
                    bboxes.append(frame_data[player_id]['bbox'])
                else:
                    bboxes.append([np.nan, np.nan, np.nan, np.nan])

            df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])
            # limit_area='inside' only fills gaps that are bounded on both sides by real detections
            df = df.interpolate(limit=max_gap, limit_area='inside')

            for frame_num, row in df.iterrows():
                if player_id not in player_tracks[frame_num] and not np.isnan(row['x1']):
                    player_tracks[frame_num][player_id] = {'bbox': row.tolist(), 'interpolated': True}

        return player_tracks

    def add_smoothed_bboxes(self, object_tracks, window=9):
        """Store a temporally smoothed copy of each bbox as 'bbox_draw'.

        Drawing from raw per-frame detections makes the ellipse jitter with
        detection noise. A centered rolling mean over each contiguous run of
        a track removes the jitter without introducing lag. Analysis fields
        ('bbox', 'position', speed) are untouched — only drawing uses these.
        """
        all_ids = set()
        for frame_data in object_tracks:
            all_ids.update(frame_data.keys())

        for tid in all_ids:
            fns = [fn for fn, fd in enumerate(object_tracks) if tid in fd]
            # Split into contiguous runs so smoothing never bridges a gap
            runs, run = [], [fns[0]]
            for fn in fns[1:]:
                if fn == run[-1] + 1:
                    run.append(fn)
                else:
                    runs.append(run)
                    run = [fn]
            runs.append(run)

            for run in runs:
                boxes = pd.DataFrame(
                    [object_tracks[fn][tid]['bbox'] for fn in run],
                    columns=['x1', 'y1', 'x2', 'y2'])
                smooth = boxes.rolling(window, center=True, min_periods=1).mean()
                for row_idx, fn in enumerate(run):
                    object_tracks[fn][tid]['bbox_draw'] = smooth.iloc[row_idx].tolist()

        return object_tracks

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, fps=30):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        # Fresh tracker with persistence-tuned params every time we do real tracking
        tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,          # hold lost tracks for 60 frames (~2 s @ 30 fps)
            minimum_matching_threshold=0.8,
            frame_rate=fps,
        )

        detections = self.detect_frames(frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert GoalKeeper to player object
            for object_ind , class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # Track Objects
            detection_with_tracks = tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox":bbox}
            
            # Keep only the single highest-confidence ball detection per frame, so a
            # low-confidence false positive can't hijack the ball marker.
            best_ball_conf = -1.0
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                conf = frame_detection[2]
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball'] and conf is not None and conf > best_ball_conf:
                    best_ball_conf = conf
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)

        return tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control, dim_interpolated=False):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players (smoothed bbox when available, so the ellipse
            # doesn't jitter with per-frame detection noise)
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                if dim_interpolated and player.get("interpolated", False):
                    # Half-brightness so interpolated boxes are visually distinct during debugging
                    color = tuple(c // 2 for c in color)
                draw_bbox = player.get("bbox_draw", player["bbox"])
                # Referees (team 0) get an ellipse but never an ID number —
                # numbers are for players only (they're scored by number).
                label = None if player.get("team") == 0 else track_id
                frame = self.draw_ellipse(frame, draw_bbox, color, label)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, draw_bbox, (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee.get("bbox_draw", referee["bbox"]), (0,255,255))
            
            # Draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))

            # Team ball control overlay intentionally not drawn; the
            # team_ball_control array is still computed in main.py for export.

            output_video_frames.append(frame)

        return output_video_frames