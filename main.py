from utils import read_video, save_video
from trackers import Tracker
from trackers.track_stabilizer import (fix_swap_pairs, merge_fragmented_tracks,
                                       remove_duplicate_tracks, remove_short_tracks,
                                       renumber_players)
import cv2
import json
import os
import numpy as np
from team_assigner import TeamAssigner, TeamAssignerSiglip, TeamRoster
from team_assigner.goalkeeper import assign_goalkeeper_teams
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
from player_rating import rate_players


def main():
    video_path = 'input_videos/08fd33_4.mp4'

    # Read Video
    video_frames = read_video(video_path)

    # Detect real FPS so ByteTrack's lost_track_buffer is frame-count-accurate
    _cap = cv2.VideoCapture(video_path)
    fps = int(round(_cap.get(cv2.CAP_PROP_FPS) or 30))
    _cap.release()

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    # Stub versioned: v3 reflects imgsz=1280 detection (different tracks from v2's default res)
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs_v3.pkl',
                                       fps=fps)

    # Fill short player gaps (≤10 frames) before team assignment
    tracks["players"] = tracker.interpolate_player_positions(tracks["players"])

    # ------------------------------------------------------------------
    # Assign Player Teams (runs on raw track IDs; cached in the roster stub)
    # ------------------------------------------------------------------
    # Fixed annotation colors (BGR): team 1 = white, team 2 = blue
    team_annotation_colors = {1: (255, 255, 255), 2: (255, 0, 0)}

    # Persistent roster: stores every player's team on disk so reruns skip
    # classification entirely. Versioned alongside the track stub (v3) because
    # track_ids are only meaningful for the tracks they were computed from.
    roster_path = 'stubs/team_roster_v3.json'
    all_player_ids = {pid for frame in tracks['players'] for pid in frame}

    roster = TeamRoster.load(roster_path)
    if roster is not None and roster.covers(all_player_ids):
        roster.apply_to_tracks(tracks['players'], team_annotation_colors)
        print(f"Team assignment: loaded stored roster ({roster.summary()})")
    else:
        if roster is not None:
            print("Stored roster doesn't cover current tracks; reclassifying")

        # SigLIP in raw 768-D (use_umap=False): most robust separator.
        # UMAP to 3-D can distort the space enough to misplace ambiguous jerseys
        # (e.g. green-and-white stripes), so UMAP is disabled.
        USE_SIGLIP = True
        team_assigner = None
        if USE_SIGLIP:
            try:
                team_assigner = TeamAssignerSiglip(use_umap=False)
                team_assigner.fit(video_frames, tracks['players'], num_fit_frames=50)
                print("Team assignment: using SigLIP embedding clustering")
            except Exception as e:
                print(f"SigLIP team assigner unavailable ({e}); falling back to color method")
                team_assigner = None
        if team_assigner is None:
            team_assigner = TeamAssigner()
            team_assigner.assign_team_color(video_frames, tracks['players'])
            print("Team assignment: using color clustering")

        # Color extraction is cheap — classify every frame for maximum vote accuracy.
        # SigLIP embedding is expensive — sample every 15th frame (seen-tracking
        # still guarantees every track_id gets at least one classification).
        cadence = 15 if isinstance(team_assigner, TeamAssignerSiglip) else 1
        team_assigner.assign_teams_to_tracks(video_frames, tracks['players'],
                                             team_annotation_colors, classify_every=cadence)

        roster = team_assigner.roster
        roster.save(roster_path)
        print(f"Team roster saved to {roster_path} ({roster.summary()})")

    # ------------------------------------------------------------------
    # Track stabilization: make IDs follow physical players
    # ------------------------------------------------------------------
    print("Track stabilization:")
    # 1. Ghost tracks: YOLO double-detections get their own short ID and
    #    draw a second ellipse on top of a player — remove them first.
    remove_duplicate_tracks(tracks['players'], roster)
    # 2. Collision ID swaps: exchange the underlying track data so each ID
    #    stays on one physical player (and one team) for the whole video.
    fix_swap_pairs(tracks['players'], roster)
    # 3. Fragmented tracks: a player re-detected under a fresh ID after a
    #    short gap (or a brief overlap) gets his original ID back.
    merge_fragmented_tracks(tracks['players'], roster)
    # 4. Whatever is still shorter than ~3s is a phantom or unattachable
    #    scrap — drop it so it neither draws nor pollutes the ratings.
    remove_short_tracks(tracks['players'], roster)
    # 5. Bridge the merge seams so the ellipse doesn't blink out during the gap.
    tracks['players'] = tracker.interpolate_player_positions(tracks['players'],
                                                             max_gap=45)

    # ------------------------------------------------------------------
    # Goalkeepers: position-detected, assigned to the team of their nearest
    # outfield players (jersey clustering gets keepers wrong by design —
    # they wear different colors than their own team).
    # ------------------------------------------------------------------
    print("Goalkeepers:")
    gk_result = assign_goalkeeper_teams(tracks['players'], roster)

    # Friendly numbers AFTER goalkeeper correction so each team's range is
    # final: team 1 -> 1-12, team 2 -> 13-24, keepers take 1 and 13.
    # These are the numbers shown on the video, in the Excel, and used in
    # team_overrides.json. Deterministic across reruns.
    mapping = renumber_players(tracks['players'], roster,
                               gk_pids=[pid for pid, _ in gk_result.values()])
    gk_result = {side: (mapping.get(pid, pid), team)
                 for side, (pid, team) in gk_result.items()}

    # ------------------------------------------------------------------
    # Manual overrides — the "button" for fixing any wrong team by hand.
    # Edit team_overrides.json and rerun; it wins over everything above.
    # ------------------------------------------------------------------
    overrides_path = 'team_overrides.json'
    if not os.path.exists(overrides_path):
        with open(overrides_path, 'w') as f:
            json.dump({
                "_help": ("Manual team fixes, applied last (they win over all "
                          "automatic assignment). player_teams maps a player's "
                          "ID number (shown in the video) to a team: 1 = white, "
                          "2 = blue. swap_goalkeepers: true flips both keepers. "
                          "Example: \"player_teams\": {\"483\": 2}"),
                "player_teams": {},
                "swap_goalkeepers": False,
            }, f, indent=2)
        print(f"Created {overrides_path} — edit it to fix any player's team by hand")
    else:
        with open(overrides_path) as f:
            overrides = json.load(f)
        if overrides.get('swap_goalkeepers') and gk_result:
            for side, (pid, team) in gk_result.items():
                roster.set_team(pid, 3 - team)
                print(f"  Override: swapped {side} goalkeeper (track {pid}) to team {3 - team}")
        for pid, team in overrides.get('player_teams', {}).items():
            roster.set_team(int(pid), int(team))
            print(f"  Override: player {pid} -> team {team}")

    # Repaint every frame with the final teams and save the authoritative
    # roster (these are the IDs/teams to score players by).
    roster.apply_to_tracks(tracks['players'], team_annotation_colors)
    roster.save('stubs/team_roster_final.json')
    print(f"Final roster ({roster.summary()}) saved to stubs/team_roster_final.json")

    # ------------------------------------------------------------------
    # Positions, camera movement, view transform, speed — computed AFTER
    # stabilization so they describe the corrected tracks.
    # ------------------------------------------------------------------
    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Ball trajectory: use ALL per-frame ball candidates (not just the
    # highest-confidence one — that discards the real ball whenever the
    # penalty spot scores higher) and pick the physically consistent chain.
    frame_h, frame_w = video_frames[0].shape[:2]
    print("Ball trajectory:")
    ball_candidates = tracker.get_ball_candidates(video_frames,
                                                  read_from_stub=True,
                                                  stub_path='stubs/ball_candidates_v1.pkl')
    tracks["ball"] = tracker.select_ball_trajectory(ball_candidates,
                                                    tracks['players'],
                                                    camera_movement=camera_movement_per_frame,
                                                    frame_w=frame_w, frame_h=frame_h)
    # Predict-and-search loop: gaps are filled with motion-predicted positions
    # (the ball keeps its momentum), a zoomed re-detection searches around each
    # prediction, and the trajectory is re-selected so recovered detections
    # must still fit a consistent path. Second iteration searches again with
    # the improved predictions (cached, so repeats are cheap).
    for _ in range(2):
        ball_candidates = tracker.detect_ball_in_gaps(video_frames, ball_candidates,
                                                      tracks["ball"],
                                                      stub_path='stubs/ball_gap_candidates_v1.pkl')
        tracks["ball"] = tracker.select_ball_trajectory(ball_candidates,
                                                        tracks['players'],
                                                        camera_movement=camera_movement_per_frame,
                                                        frame_w=frame_w, frame_h=frame_h)

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            player_team = tracks['players'][frame_num][assigned_player].get('team', 1)
            if player_team not in (0,):  # skip referees (team=0)
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(player_team)
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
        else:
            # Default to team 1 if no prior frame exists yet
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 1)
    team_ball_control= np.array(team_ball_control)

    # Export ball control (not drawn on the video anymore): per-frame team
    # in possession plus overall percentages, for later analysis.
    t1_frames = int((team_ball_control == 1).sum())
    t2_frames = int((team_ball_control == 2).sum())
    with open('output_videos/team_ball_control.json', 'w') as f:
        json.dump({
            "team_1_percent": round(100 * t1_frames / len(team_ball_control), 2),
            "team_2_percent": round(100 * t2_frames / len(team_ball_control), 2),
            "per_frame": team_ball_control.tolist(),
        }, f)
    print(f"Ball control exported to output_videos/team_ball_control.json "
          f"(team 1: {100*t1_frames/len(team_ball_control):.1f}%, "
          f"team 2: {100*t2_frames/len(team_ball_control):.1f}%)")

    # Player ratings: possession -> events -> movement -> points & 0-10 rating.
    # Exports output_videos/player_ratings.xlsx (+ .csv). debug=True prints
    # every detected event with its frame number for manual verification.
    # gk_result supplies keeper identities and each team's attack direction.
    rate_players(tracks, fps=fps, gk_teams=gk_result, frame_w=frame_w, debug=True)

    # Smooth bboxes for drawing only (kills ellipse jitter; analysis untouched)
    tracker.add_smoothed_bboxes(tracks['players'])
    tracker.add_smoothed_bboxes(tracks['referees'])

    # Draw output (camera-movement and speed/distance overlays intentionally
    # not drawn — they cluttered the video; the underlying data is still
    # computed and available in tracks)
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')

if __name__ == '__main__':
    main()
