# Man usuk — Temporal evidence fusion for team possession

## Objective

Estimate the possession state of every frame in the Phase 1 tactical recording:
H (home), A (away), or D (dead ball). This deterministic offline method combines
team-level proximity, automatic tactical phase evidence and temporal continuity.
It builds on the official Metrica Sports starter and retains its CSV validator.

## Inputs and alignment

We use the tactical ATD metadata, tactical raw tracking and tactical Smart Tagging.
Metadata assigns tracking slots to teams; no persistent player identities are assumed.
Raw ATD frame IDs are 1-based indices on the tactical video timeline, so output frame
equals raw frame minus one. We do not apply the offsets used for corrected tracking.
The parser requires every expected frame exactly once and checks track-slot counts.
Coordinates are multiplied by the pitch dimensions published in `games.py`.
Tag intervals are converted from seconds to frames by the official loader.

Videos, broadcast tracking and homographies are not used. Their omission keeps the
pipeline lightweight and avoids needing to align two different camera timelines.

## Method

For each frame, compute the closest finite player distance for each team. Proximity
is `exp(-distance / 3 metres)`. The home emission is 1.8 times the difference between
home and away proximity; the away emission is its negative. Missing ball coordinates
provide no proximity evidence. This bounded score prevents distant players from
producing strong possession claims during loose or aerial balls.

Automatic attacking phases support the tagged team; defensive phases support its
opponent. We use build-up, progression, chance creation, transitions, final-third/box
phases and long balls. Overlapping same-team tags do not accumulate strength, and
opposing tags cancel. The resulting team evidence has weight 0.65. Set pieces and
goals are deliberately excluded: their intervals do not identify exact restart or
whistle times. Unknown team tags are ignored.

The dead-ball emission starts at -0.7 when the ball is visible. Coordinates more than
0.5 metres beyond the pitch add 3.0. A nearly stationary ball adds 1.0 when over 90%
of adjacent displacements in a centred two-second window are below 0.08 metres.
Missingness gives weak dead-ball evidence only far from any observation: zero within
four seconds of a visible ball, increasing linearly to 0.6 over the next eight seconds.
These are heuristics, not claims that stationary or occluded balls are always dead.

An offline Viterbi decoder maximises the sum of frame emissions minus transition
costs: 12 for H/A changes and 8 for entering or leaving D. Self-transitions cost zero.
It requires accumulated evidence before changing state, while preserving frame-level
transition placement through backtracking. All constants are fixed in `CONFIG` and
were chosen from physical/time-scale assumptions before public evaluation. The
program always emits H/A/D; it does not exploit X abstention handling.

## Possession interpretation

A brief proximity advantage for the opponent should not establish control by itself.
Short occlusions or ambiguous loose balls are bridged through sequence continuity.
Sustained opposing evidence can establish a takeover. Stoppages are inferred from
geometry, motion and prolonged missingness, not from manually identified frames.
This approximates controlled possession; it cannot directly establish referee intent.

## Training and compliance

No model training, external labelled dataset, hand annotation, manual correction,
private ground truth or match-specific human event data is used. The prediction
pipeline never reads public ground truth. Synthetic tests use invented arrays and
invented tracking records. Public evaluation is only a post-freeze pipeline check;
its score must not be used to select parameters or versions. No match data is sent
to hosted LLMs by the pipeline. AI assistance was used to develop code and documentation
from public specifications; competition asset contents were processed locally.

## Limitations

Long occlusion can be confused with dead ball. Stationary live possession, especially
by a goalkeeper, can be mistaken for a stoppage. Tracking errors can place the ball
outside the field. Automatic tactical tags can be inaccurate or extend across actual
possession boundaries. Fixed transition penalties may suppress genuine brief spells.
No whistle, referee gesture, explicit pass model or ball-height information is used.
This is a first heuristic implementation; superiority to the official baseline on the
hidden full match has not been established. Public five-minute scores do not establish it.

## Reproduction

Use Python 3.11, install `requirements-local.txt`, and place the three input files in
`data/tactical/`. Run `python -m solution.predict --team "Man usuk"`, then
`python evaluation/validate_submission.py submission/Man_usuk.csv`.
`RUN_PROJECT.md` gives the full setup and test commands. The local manifest records
input and output SHA-256 hashes, configuration, Python/NumPy versions and source hash.
Re-running the same source on identical inputs must produce the same CSV bytes.

## Verification record

Eight synthetic unit tests passed, including an exhaustive optimal-path comparison
for the decoder, a short deflection, sustained takeover, stoppage, short occlusion,
outside-field ball, conflicting tactical tags, and frame-index parsing checks.
The official validator accepted the generated 199,319-row CSV.
Two independent prediction runs produced byte-identical CSV files (SHA-256 checked).
After the method was fixed, one public five-minute pipeline evaluation returned
macroF1 0.522753 and spellF1 0.227027. This is below the published public-window
baseline_smooth50 spellF1 of 0.3256. The result does not establish full-match quality,
and no parameter changes or model selection were made in response to this score.
The weak dead-ball result is a known limitation, not a solved problem.

## Attribution

Official starter, definitions, loaders and evaluation tools:
https://github.com/metrica-sports/more-than-a-hack-2027 (MIT-licensed code).
Competition data is separately licensed and is not included in the source release.
