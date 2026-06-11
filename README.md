# Football Analysis & Player Rating

Turn a raw football match clip into:

- an **annotated video** — every player tracked with a stable shirt-style number
  (team 1 → `1–12`, team 2 → `13–24`, keepers are `1` and `13`), team-colored
  ellipses, referees marked without numbers, and a marker that stays on the ball
- a **player rating Excel sheet** — points for every good action, deductions for
  every bad one, using professional scouting categories (Short/Long pass,
  Through ball, Challenge, Pressure, Intercept, Close control, Link up,
  Dribble, Pace), plus a normalized 0–10 rating per role
- **exported match data** — team ball possession per frame and the definitive
  player → team roster, as JSON

Built on YOLO detection + ByteTrack tracking, with heavy post-processing that
fixes the things raw trackers get wrong: identity swaps after collisions,
fragmented tracks, ghost double-detections, goalkeeper team assignment, and
ball/penalty-spot confusion.

---

## Quick start

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Download the large files (too big for GitHub) — see links below
#    models/best.pt            <- trained YOLO model (players/referees/ball)
#    input_videos/<clip>.mp4   <- the match clip (set the path in main.py)

# 3. Run
python main.py
```

**Everything heavy is cached in `stubs/`** (and the caches for the sample clip
ship with the repo), so runs take about a minute. Only a *new* video is slow
the first time (~an hour on CPU: YOLO runs over every frame twice and SigLIP
classifies the teams) — after that it's cached too. Delete a stub file to
force that stage to recompute.

### Download large files (not in the repo — GitHub size limits)

| File | Size | Link |
|---|---|---|
| `models/best.pt` — trained YOLO model | 186 MB | [Google Drive](https://drive.google.com/file/d/1BahaQ-U5Wxkmz9xcK5NRn4koFpV-tbya/view?usp=sharing) |
| `input_videos/08fd33_4.mp4` — sample clip | 19 MB | [Google Drive](https://drive.google.com/file/d/1S80r2fIoa7ZjSjPaWZqK3G5RcZ4tek8H/view?usp=sharing) |
| `output_videos/output_video.avi` — example result | 66 MB | [Google Drive](https://drive.google.com/file/d/1l0kfQ4lQmeCzMUytLLbtZkYfzBRH_Y14/view?usp=sharing) |

Place each file at the path shown above after downloading.

### Outputs

| File | Content |
|---|---|
| `output_videos/output_video.avi` | annotated match video |
| `output_videos/player_ratings.xlsx` | **Ratings** sheet (points, 0–10 rating, category counts, "What They Did") + **Events** sheet (every detected event with frame, time, players, grades) |
| `output_videos/player_ratings.csv` | ratings table as CSV |
| `output_videos/team_ball_control.json` | possession % + per-frame team in possession |
| `stubs/team_roster_final.json` | definitive player number → team mapping |

---

## How the pipeline works

```
input video
   ├─ 1. Detect + track (YOLO @1280px + ByteTrack)          [cached]
   ├─ 2. Team assignment (SigLIP embeddings + KMeans,
   │      collision-aware voting, ID-swap detection)         [cached]
   ├─ 3. Track stabilization
   │      • remove ghost duplicate tracks (double ellipses)
   │      • swap back collision ID exchanges
   │      • merge fragmented tracks (same player, new ID)
   │      • drop phantom scraps
   ├─ 4. Goalkeepers: found by position, assigned to the team
   │      that owns their goal (deepest-defender vote)
   ├─ 5. Renumber: team 1 → 1-12, team 2 → 13-24 (GK = 1 / 13)
   ├─ 6. Manual overrides (team_overrides.json)
   ├─ 7. Positions, camera movement, pitch homography, speed
   ├─ 8. Ball trajectory
   │      • ALL ball candidates per frame                    [cached]
   │      • static-marking blacklist (penalty spot etc.)
   │      • dynamic-programming path selection
   │      • zoomed re-detection inside gaps                  [cached]
   ├─ 9. Possession, events, player ratings  → Excel/CSV
   └─ 10. Draw (smoothed ellipses) → output video
```

### Why the unusual parts exist

- **Ball candidates instead of best-per-frame**: the painted penalty spot often
  out-scores the real ball in YOLO confidence. Keeping only the best detection
  permanently loses the ball. We keep all candidates and pick the physically
  consistent path through them.
- **ID-swap fixing**: when two players collide, ByteTrack often hands each one
  the other's ID. The team assigner detects mirrored mid-track team flips near
  a collision and the stabilizer exchanges the underlying track data back.
- **Goalkeeper voting**: keepers wear different colors than their team, so
  jersey clustering always misassigns them. Instead, the majority team of the
  deepest outfield player at each goal (offside logic ⇒ that's a defender)
  decides which team owns the goal.

---

## Player rating system (`player_rating/`)

Possession → events → movement → grades, using the scouting categorization
vocabulary. Every action earns an A-style grade in 0.5 steps (good adds, bad
subtracts), one action can combine categories:

| Category | Grade | Detected as |
|---|---|---|
| Short pass | +0.5 | completed pass < ~13 m |
| Long pass | +1.0 | completed pass ≥ ~13 m |
| Through ball | +1.5 | large forward gain toward opponent goal |
| Under pressure | +0.5 bonus | opponent at the ball during the pass |
| Failed pass | −0.5 / −1.0 | possession lost via a traveled ball |
| Intercept | +1.0 | cutting out that pass (always standalone) |
| Challenge won / lost | +1.0 / −0.5 | close-range duel (scrambles merge into one) |
| Won in own third | +0.5 bonus | defensive-zone ball win |
| Pressure | +0.5 | forcing the holder into an error |
| Close control | +0.5 | clean reception kept under pressure |
| Link up | +1.0 | reception under pressure + quick lay-off |
| Dribble / att 1v1 | +1.0 | carrying the ball ≥ ~8 m in one spell |
| Pace (sprint) | +0.25 | > 20 km/h sustained |

The 0–10 rating normalizes per-minute rates (points, pass accuracy, challenge
win rate, distance, sprints) **within role groups** (GK / defender /
midfielder / attacker, inferred from average depth) so keepers aren't compared
to strikers and short appearances aren't punished.

**Everything is tunable** in `player_rating/config.py`: grade values, rating
weights, and all detection thresholds. Set `debug=True` on `rate_players()` in
`main.py` (default) to print every event with its frame number for manual
verification against the video.

> ⚠️ Ratings from a 30-second clip are statistically noisy. They become
> meaningful with 10+ minutes of footage.

---

## Fixing mistakes by hand

`team_overrides.json` (created automatically in the project root) is applied
last and wins over everything:

```json
{
  "player_teams": { "17": 1 },     // force player 17 (video number) to team 1
  "swap_goalkeepers": true          // flip both keepers' teams at once
}
```

Edit it and rerun — reruns are fast because classification is cached.

---

## Project structure

```
main.py                        pipeline orchestration
trackers/
  tracker.py                   YOLO+ByteTrack, ball candidates & trajectory,
                               drawing, bbox smoothing
  track_stabilizer.py          dedup, swap-fix, fragment merge, renumbering
team_assigner/
  team_assigner_siglip.py      SigLIP embedding team classification
  team_assigner.py             color-clustering fallback
  team_roster.py               persistent player→team roster (JSON)
  goalkeeper.py                GK detection + goal-ownership team vote
  occlusion.py                 bbox-overlap detection (collision frames)
player_rating/
  config.py                    ALL grades, weights, thresholds
  possession.py                smoothed per-frame possession
  events.py                    pass/duel/dribble/link-up/... detection
  movement.py                  distance, sprints (pitch meters)
  rating.py                    points, roles, 0-10 rating
  export.py                    Excel/CSV export
camera_movement_estimator/     optical-flow camera compensation
view_transformer/              pixel → pitch-meter homography
speed_and_distance_estimator/  per-player speed & distance
player_ball_assigner/          per-frame ball-to-player assignment
utils/                         video & bbox helpers
training/                      notebook used to train models/best.pt
stubs/                         cached computation (safe to delete)
```

## Notes & limitations

- The pitch homography (`view_transformer`) is calibrated for this camera
  angle; movement in meters is approximate outside the calibrated zone.
- One green-team player appears under two numbers (his track broke too
  severely mid-clip to reconnect safely) — visible as an extra roster entry.
- Categories needing ball height or human judgment (Aerial, Save, Cross,
  Shoot, set pieces, On/Off-ball intelligence) are not auto-detected.
- Model: `models/best.pt` is a YOLO model fine-tuned on the DFL Bundesliga
  Data Shootout dataset (see `training/`). Project originally based on the
  [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis)
  tutorial, since heavily extended.
