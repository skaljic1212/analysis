"""
Export a YOLO fine-tuning dataset from the pipeline's own trusted output.

The pipeline knows exactly which frames the detector finds the ball in
(trusted, auto-labeled) and which frames it fails on (the remaining gaps).
This script writes:

  ball_dataset/
    images/train/  frame_XXXX.jpg      every 2nd trusted frame
    labels/train/  frame_XXXX.txt      full auto-labels: ball + players +
                                       goalkeepers + referees (YOLO format)
    images/review/ frame_XXXX.jpg      ALL failure frames - the ball label
    labels/review/ frame_XXXX.txt      is MISSING; add it by hand
    review_hints.csv                   predicted ball position per review
                                       frame (to find it faster)
    data.yaml                          ready for `yolo train`
    README_TRAINING.md                 exact Colab fine-tuning steps

Manual work left: open the review images in any YOLO labeling tool
(Roboflow / CVAT / labelImg), add ONE ball box per image (the hints CSV
tells you roughly where to look), move them into train/, then fine-tune.

Run from the project root:  python tools/export_ball_dataset.py
"""

import os
import sys
import json
import pickle

import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import read_video  # noqa: E402
from trackers import Tracker  # noqa: E402
from trackers.track_stabilizer import (fix_swap_pairs, merge_fragmented_tracks,  # noqa: E402
                                       remove_duplicate_tracks, remove_short_tracks,
                                       renumber_players)
from team_assigner import TeamRoster  # noqa: E402
from team_assigner.goalkeeper import assign_goalkeeper_teams  # noqa: E402

VIDEO = 'input_videos/08fd33_4.mp4'
OUT = 'ball_dataset'
TRAIN_EVERY = 2  # export every 2nd trusted frame (consecutive frames are near-duplicates)


def yolo_line(cls_id, bbox, w, h):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
    bw, bh = (x2 - x1) / w, (y2 - y1) / h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main():
    print("Rebuilding pipeline state from stubs ...")
    video_frames = read_video(VIDEO)
    h, w = video_frames[0].shape[:2]
    tracker = Tracker('models/best.pt')
    names_inv = {v: k for k, v in tracker.model.names.items()}
    CLS_BALL = names_inv['ball']
    CLS_GK = names_inv.get('goalkeeper', names_inv['player'])
    CLS_PLAYER = names_inv['player']
    CLS_REF = names_inv['referee']

    tracks = tracker.get_object_tracks(video_frames, read_from_stub=True,
                                       stub_path='stubs/track_stubs_v3.pkl')
    tracks['players'] = tracker.interpolate_player_positions(tracks['players'])

    roster = TeamRoster.load('stubs/team_roster_v3.json')
    roster.apply_to_tracks(tracks['players'], {1: (255, 255, 255), 2: (255, 0, 0)})
    remove_duplicate_tracks(tracks['players'], roster)
    fix_swap_pairs(tracks['players'], roster)
    merge_fragmented_tracks(tracks['players'], roster)
    remove_short_tracks(tracks['players'], roster)
    tracks['players'] = tracker.interpolate_player_positions(tracks['players'],
                                                             max_gap=45)
    gk_result = assign_goalkeeper_teams(tracks['players'], roster)
    mapping = renumber_players(tracks['players'], roster,
                               gk_pids=[pid for pid, _ in gk_result.values()])
    gk_pids = {mapping.get(pid, pid) for pid, _ in gk_result.values()}
    roster.apply_to_tracks(tracks['players'], {1: (255, 255, 255), 2: (255, 0, 0)})

    # Ball trajectory exactly as in main.py (all cached)
    with open('stubs/camera_movement_stub.pkl', 'rb') as f:
        cam = pickle.load(f)
    cands = tracker.get_ball_candidates(video_frames, read_from_stub=True,
                                        stub_path='stubs/ball_candidates_v1.pkl')
    ball = tracker.select_ball_trajectory(cands, tracks['players'],
                                          camera_movement=cam, frame_w=w, frame_h=h)
    for _ in range(2):
        cands = tracker.detect_ball_in_gaps(video_frames, cands, ball,
                                            stub_path='stubs/ball_gap_candidates_v1.pkl')
        ball = tracker.select_ball_trajectory(cands, tracks['players'],
                                              camera_movement=cam, frame_w=w, frame_h=h)
    cands = tracker.track_ball_through_gaps(video_frames, cands, ball)
    cands = tracker.detect_ball_tiled(video_frames, cands, ball,
                                      stub_path='stubs/ball_tile_candidates_v1.pkl')
    ball = tracker.select_ball_trajectory(cands, tracks['players'],
                                          camera_movement=cam, frame_w=w, frame_h=h)

    # ---- write dataset ------------------------------------------------------
    for sub in ('images/train', 'labels/train', 'images/review', 'labels/review'):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)

    def player_labels(fn):
        lines = []
        for pid, info in tracks['players'][fn].items():
            if info.get('interpolated'):
                continue  # don't teach the detector interpolated boxes
            team = info.get('team', 1)
            cls = CLS_REF if team == 0 else (CLS_GK if pid in gk_pids else CLS_PLAYER)
            lines.append(yolo_line(cls, info['bbox'], w, h))
        for _, info in tracks['referees'][fn].items():
            lines.append(yolo_line(CLS_REF, info['bbox'], w, h))
        return lines

    n_train = n_review = 0
    hints = ["frame,predicted_x,predicted_y"]
    for fn in range(len(video_frames)):
        detected = ball[fn][1].get('detected', False)
        if detected and fn % TRAIN_EVERY != 0:
            continue
        split = 'train' if detected else 'review'
        name = f"frame_{fn:04d}"
        cv2.imwrite(os.path.join(OUT, f'images/{split}/{name}.jpg'),
                    video_frames[fn], [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = player_labels(fn)
        if detected:
            lines.append(yolo_line(CLS_BALL, ball[fn][1]['bbox'], w, h))
            n_train += 1
        else:
            b = ball[fn][1]['bbox']
            hints.append(f"{name},{(b[0]+b[2])/2:.0f},{(b[1]+b[3])/2:.0f}")
            n_review += 1
        with open(os.path.join(OUT, f'labels/{split}/{name}.txt'), 'w') as f:
            f.write("\n".join(lines) + "\n")

    with open(os.path.join(OUT, 'review_hints.csv'), 'w') as f:
        f.write("\n".join(hints) + "\n")

    names_yaml = "\n".join(f"  {k}: {v}" for k, v in sorted(tracker.model.names.items()))
    with open(os.path.join(OUT, 'data.yaml'), 'w') as f:
        f.write(f"path: .\ntrain: images/train\nval: images/train\n"
                f"names:\n{names_yaml}\n")

    with open(os.path.join(OUT, 'README_TRAINING.md'), 'w') as f:
        f.write(f"""# Fine-tuning best.pt on its own failures

{n_train} auto-labeled frames are in `images/train`, {n_review} failure
frames in `images/review` (ball label missing — that's your manual work).

## 1. Label the review frames (~30-60 min)
Open `images/review` in Roboflow / CVAT / labelImg. Each image needs ONE
`ball` box added (other classes are pre-labeled). `review_hints.csv` gives
the pipeline's predicted ball position per frame — look near it first.
The ball is often blurred or between feet — that's exactly why these
frames matter. If the ball is genuinely invisible, delete that image.
Then move the finished images+labels into the `train` folders.

## 2. Fine-tune in Google Colab (free GPU, ~20 min)
Upload `ball_dataset/` and `models/best.pt` to Drive, then:

```python
!pip install ultralytics
from ultralytics import YOLO
model = YOLO('/content/drive/MyDrive/best.pt')
model.train(data='/content/drive/MyDrive/ball_dataset/data.yaml',
            epochs=15, imgsz=1280, lr0=1e-4, freeze=10,
            batch=8, patience=5)
```

`freeze=10` keeps the backbone intact so the model doesn't forget
players/referees; the low learning rate fine-tunes gently.

## 3. Use the result
Replace `models/best.pt` with the trained `best.pt` from
`runs/detect/train/weights/`, delete `stubs/ball_*.pkl` and
`stubs/track_stubs_v3.pkl`, and rerun `python main.py`.
""")

    print(f"\nDone: {n_train} trusted frames -> images/train (auto-labeled)")
    print(f"      {n_review} failure frames -> images/review (add the ball box by hand)")
    print(f"      instructions in {OUT}/README_TRAINING.md")


if __name__ == '__main__':
    main()
