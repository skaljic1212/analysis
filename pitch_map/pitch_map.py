

import cv2
import numpy as np

PITCH_LEN, PITCH_WID = 105.0, 68.0  # meters
GK_X_LEFT, GK_X_RIGHT = 7.0, 98.0   # typical keeper depth used as anchors


def draw_pitch(width=560, height=380, margin=28):
    """Clean dark-green FIFA-style pitch with white lines."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (38, 80, 38)  # dark green
    x0, y0 = margin, margin
    x1, y1 = width - margin, height - margin
    pw, ph = x1 - x0, y1 - y0

    # mowing pattern: alternating stripes
    for i in range(10):
        if i % 2 == 0:
            sx = x0 + i * pw // 10
            cv2.rectangle(img, (sx, y0), (x0 + (i + 1) * pw // 10, y1),
                          (44, 92, 44), -1)

    def m2px(mx, my):
        return (int(x0 + mx / PITCH_LEN * pw), int(y0 + my / PITCH_WID * ph))

    w_line, col = 1, (235, 235, 235)
    cv2.rectangle(img, (x0, y0), (x1, y1), col, w_line)
    cv2.line(img, m2px(52.5, 0), m2px(52.5, 68), col, w_line)
    cv2.circle(img, m2px(52.5, 34), int(9.15 / PITCH_LEN * pw), col, w_line)
    cv2.circle(img, m2px(52.5, 34), 2, col, -1)
    for left in (True, False):
        gx = 0 if left else PITCH_LEN
        s = 1 if left else -1
        # penalty box 16.5 x 40.32, goal area 5.5 x 18.32, spot at 11 m
        cv2.rectangle(img, m2px(gx, 34 - 20.16), m2px(gx + s * 16.5, 34 + 20.16), col, w_line)
        cv2.rectangle(img, m2px(gx, 34 - 9.16), m2px(gx + s * 5.5, 34 + 9.16), col, w_line)
        cv2.circle(img, m2px(gx + s * 11, 34), 2, col, -1)
        # penalty arc
        cx, cy = m2px(gx + s * 11, 34)
        r = int(9.15 / PITCH_LEN * pw)
        cv2.ellipse(img, (cx, cy), (r, r), 0,
                    -53 if left else 127, 53 if left else 233, col, w_line)
        # goal
        cv2.rectangle(img, m2px(gx - s * 1.5, 34 - 3.66), m2px(gx, 34 + 3.66), col, w_line)
    return img, m2px


class PitchMap:
    def __init__(self, width=560, height=380):
        self.pitch, self.m2px = draw_pitch(width, height)
        self.h, self.w = self.pitch.shape[:2]
        self.matrix = None
        self.x_a, self.x_b = 1.0, 0.0  # affine correction along pitch length
        self.y_a, self.y_b = 1.0, 0.0

    # ---- projection -------------------------------------------------------

    def _raw_project(self, pts):
        """Pixel points -> homography plane (extrapolated beyond calibration)."""
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(arr, self.matrix).reshape(-1, 2)

    def calibrate(self, player_tracks, view_transformer, gk_left=None,
                  gk_right=None):
        """Anchor the projection using the goalkeepers' mean positions.

        The homography covers only the central strip, so positions outside
        it are projective extrapolations on an unknown scale. Keepers stand
        at their goals — mapping their mean projected x to 7 m and 98 m
        pins the whole length axis. Width is scaled from observed extent.
        """
        self.matrix = view_transformer.persepctive_trasnformer

        feet = {}
        for fn, pt in enumerate(player_tracks):
            for pid, info in pt.items():
                b = info['bbox']
                feet.setdefault(pid, []).append(((b[0] + b[2]) / 2, b[3]))

        proj_mean = {pid: self._raw_project(p).mean(axis=0)
                     for pid, p in feet.items()}

        all_pts = np.concatenate([self._raw_project(p) for p in feet.values()])
        # length axis: anchor on keepers when available, else percentiles
        if gk_left in proj_mean and gk_right in proj_mean:
            lx, rx = proj_mean[gk_left][0], proj_mean[gk_right][0]
        else:
            lx, rx = np.percentile(all_pts[:, 0], [1, 99])
        self.x_a = (GK_X_RIGHT - GK_X_LEFT) / max(1e-6, rx - lx)
        self.x_b = GK_X_LEFT - self.x_a * lx
        # width axis: calibrated strip is trustworthy -> only clamp later
        y_lo, y_hi = np.percentile(all_pts[:, 1], [2, 98])
        self.y_a = 62.0 / max(1e-6, y_hi - y_lo)
        self.y_b = 3.0 - self.y_a * y_lo

    def project(self, points_px):
        """Frame pixel coords -> pitch meters (clamped to the field)."""
        raw = self._raw_project(points_px)
        mx = np.clip(self.x_a * raw[:, 0] + self.x_b, 0.5, PITCH_LEN - 0.5)
        my = np.clip(self.y_a * raw[:, 1] + self.y_b, 0.5, PITCH_WID - 0.5)
        return list(zip(mx, my))

    # ---- rendering ---------------------------------------------------------

    def render_frame_map(self, player_track, ball_bbox=None):
        """One radar image for one frame."""
        img = self.pitch.copy()
        pts, metas = [], []
        for pid, info in player_track.items():
            b = info.get('bbox_draw', info['bbox'])
            pts.append(((b[0] + b[2]) / 2, b[3]))
            metas.append((pid, info.get('team', 1),
                          info.get('team_color', (255, 255, 255))))
        if ball_bbox is not None:
            pts.append((((ball_bbox[0] + ball_bbox[2]) / 2,
                         (ball_bbox[1] + ball_bbox[3]) / 2)))
        if not pts:
            return img
        coords = self.project(pts)

        for (mx, my), (pid, team, color) in zip(coords, metas):
            px = self.m2px(mx, my)
            if team == 0:
                cv2.circle(img, px, 4, (0, 215, 255), -1)        # referee
                cv2.circle(img, px, 4, (20, 20, 20), 1)
            else:
                cv2.circle(img, px, 6, tuple(int(c) for c in color), -1)
                cv2.circle(img, px, 6, (20, 20, 20), 1)
                cv2.putText(img, str(pid), (px[0] - 6, px[1] + 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                            (0, 0, 0) if team == 1 else (255, 255, 255), 1,
                            cv2.LINE_AA)
        if ball_bbox is not None:
            bx = self.m2px(*coords[-1])
            cv2.circle(img, bx, 4, (60, 255, 60), -1)
            cv2.circle(img, bx, 4, (20, 20, 20), 1)
        return img

    def overlay_video(self, frames, tracks, scale=0.30, alpha=0.88):
        """Stamp the radar onto the bottom-center of every frame."""
        mh, mw = int(self.h * scale * 1.7), int(self.w * scale * 1.7)
        for fn, frame in enumerate(frames):
            ball = tracks['ball'][fn].get(1, {}).get('bbox') \
                if fn < len(tracks['ball']) else None
            radar = self.render_frame_map(tracks['players'][fn], ball)
            radar = cv2.resize(radar, (mw, mh), interpolation=cv2.INTER_AREA)
            H, W = frame.shape[:2]
            x0 = (W - mw) // 2
            y0 = H - mh - 18
            roi = frame[y0:y0 + mh, x0:x0 + mw]
            cv2.addWeighted(radar, alpha, roi, 1 - alpha, 0, dst=roi)
        return frames

    def average_position_map(self, player_tracks):
        """Static map of every player's mean position (for the README)."""
        img = self.pitch.copy()
        feet, team_info = {}, {}
        for pt in player_tracks:
            for pid, info in pt.items():
                b = info['bbox']
                feet.setdefault(pid, []).append(((b[0] + b[2]) / 2, b[3]))
                team_info[pid] = (info.get('team', 1),
                                  info.get('team_color', (255, 255, 255)))
        for pid, pts in feet.items():
            team, color = team_info[pid]
            if team == 0:
                continue
            mx, my = self.project([np.mean(pts, axis=0)])[0]
            px = self.m2px(mx, my)
            cv2.circle(img, px, 9, tuple(int(c) for c in color), -1)
            cv2.circle(img, px, 9, (20, 20, 20), 1)
            cv2.putText(img, str(pid), (px[0] - 8, px[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 0) if team == 1 else (255, 255, 255), 1,
                        cv2.LINE_AA)
        return img
