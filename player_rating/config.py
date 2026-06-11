"""
All tunable constants for the player rating system.

Grades follow the scouting instructions ("Word sa uputama za ocene"):
actions are graded in 0.5 steps (A+0.5, A+1, A+1.5 ... / A-0.5, A-1),
good actions add, bad actions subtract, and one action can carry several
categories. Categories use the vocabulary from "Instrukcije za
kategorizaciju" (Short pass, Long pass, Through ball, Challenge,
Pressure, Intercept, Close control, Link up, Dribble, Pace...).

Distances are in pixels unless stated otherwise; tune to your footage.
"""

# ---- Stage 1: possession ----------------------------------------------------
MAX_BALL_DISTANCE_PX = 70      # ball must be this close to a player's feet
MIN_POSSESSION_FRAMES = 4      # frames a challenger must be nearest to take over
FREE_BALL_TAKE_FRAMES = 2      # frames to claim a ball nobody holds
POSSESSION_LOSS_FRAMES = 12    # ball away from everyone this long -> nobody holds

# ---- Stage 2: events ---------------------------------------------------------
MIN_PASS_TRAVEL_PX = 60        # opponent gain counts as failed pass/intercept
                               # only if the ball traveled at least this far
MAX_EVENT_GAP_FRAMES = 50      # possession changes further apart than this are
                               # not connected into an event (ball lost/out)
DUEL_RADIUS_PX = 55            # opponent within this radius of the ball = contact
DUEL_MIN_FRAMES = 5            # contested window must last this long

LONG_PASS_PX = 450             # pass traveling at least this far = Long pass (~13m)
THROUGH_BALL_GAIN_PX = 400     # forward gain toward opponent goal for Through ball
SCRAMBLE_WINDOW_FRAMES = 40    # duels closer together than this merge into one
DRIBBLE_MIN_PX = 250           # holder must carry the ball this far (~8m)
DRIBBLE_MIN_FRAMES = 12        # ... over at least this many frames
CLOSE_CONTROL_KEEP_FRAMES = 10 # keep the ball this long under pressure = Close control
LINKUP_MAX_FRAMES = 40         # reception under pressure + pass within this = Link up

# ---- Stage 3: movement --------------------------------------------------------
SMOOTH_WINDOW_FRAMES = 7       # centered rolling mean on positions
PX_TO_METERS = 0.03            # fallback scale when homography coords missing
TRANSFORMED_MIN_COVERAGE = 0.6 # use pitch meters only if ≥60% frames have them
MAX_PLAYER_SPEED_MS = 12.0     # steps implying faster than this are tracker noise
SPRINT_SPEED_KMH = 20.0        # sprint threshold (Pace category)
SPRINT_MIN_SECONDS = 0.6       # sustained at least this long
SPEED_WINDOW_SECONDS = 0.4     # speed measured over this window

# ---- Stage 4: grading ----------------------------------------------------------
MIN_TRACK_FRAMES = 50          # players tracked less than this are not rated

# A-style grades in 0.5 steps: every good action adds, every bad one subtracts.
GRADES = {
    "short_pass": +0.5,
    "long_pass": +1.0,
    "through_ball": +1.5,          # deep low pass in behind — merits a good grade
    "pass_pressure_bonus": +0.5,   # completed while an opponent breathes down the neck
    "failed_pass": -0.5,
    "failed_long_pass": -1.0,      # losing a long/deep attempt hurts more
    "intercept": +1.0,             # always a standalone category
    "challenge_won": +1.0,
    "challenge_lost": -0.5,
    "defensive_zone_bonus": +0.5,  # winning the ball in your own third is worth more
    "pressure": +0.5,              # forcing the holder into an error
    "close_control": +0.5,         # clean reception/control under pressure
    "link_up": +1.0,               # receive under pressure, lay off, continue
    "dribble": +1.0,               # att 1v1 / carrying the ball 8m+
    "sprint": +0.25,               # Pace
}

# Role split inside each team by average depth (GK comes from detection)
ROLE_DEFENDER_PCT = 0.35       # deepest 35% of a team's outfielders = defenders
ROLE_ATTACKER_PCT = 0.30       # highest 30% = attackers; the rest = midfielders

# 0-10 rating weights — metrics are normalized per minute on pitch and within
# role groups (GK / defender / midfielder / attacker) when the group has ≥3
# players, otherwise across all outfield players.
RATING_WEIGHTS = {
    "points_rate": 0.35,        # graded points per 10 minutes
    "pass_accuracy": 0.20,
    "challenge_win_rate": 0.15,
    "distance_rate": 0.20,      # meters per minute (Effort)
    "sprint_rate": 0.10,        # sprints per 10 minutes (Pace)
}
