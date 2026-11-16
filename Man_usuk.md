# Man usuk — Control-confirmation v2

## Task and status

Produce one H/A/D/X label for every frame of the Phase 1 tactical recording.
This is a deterministic, lightweight heuristic implementation. No full-match
performance claim or qualification guarantee is made. V2 is the default method;
v1 remains available for reproducibility, not for public-score-based selection.

## Inputs and alignment

The three inputs are tactical ATD metadata, tactical raw tracking, and automatic
Smart Tagging. Team membership comes from metadata track slots, not persistent
player identities. Raw frame IDs are one-based video indices: subtract one, with
no corrected-tracking offsets. Every expected frame must appear exactly once.
Normalised coordinates are converted to metres using the official pitch dimensions.
Tags are converted from seconds to frames by the official loader.

Player activity is the median finite non-negative speed of at least ten tracked
players, excluding values above 15 m/s as implausible. This team-level summary
avoids deriving velocities across identity switches. Insufficient coverage yields
unknown activity rather than an assumption of low movement.

Videos, audio, broadcast tracking and homography are not used. No external training
data or pretrained model is used, and no competition assets are uploaded by the program.

## Method

### Reliable motion

Adjacent ball positions implying speeds above 45 m/s invalidate both endpoints.
Ball speed uses a backward difference over 0.2 seconds only when every intervening
frame is reliable. No velocity is inferred across an occlusion; no trajectory is
manually corrected or interpolated. Thresholds use seconds/metres and scale by FPS.

### Stoppages, independently of team ownership

A dead-ball segment requires either:

- reliable ball positions more than 0.5 metres beyond a pitch boundary, sustained
  for 0.4 seconds; or
- a ball speed below 0.5 m/s and median player speed below 1.2 m/s, inside an automatic
  SET PIECES interval, sustained for 1.5 seconds.

Entire qualifying runs are marked offline. No fixed restart or whistle frame is
assumed from the tags. A stationary goalkeeper without restart evidence is not
marked dead merely for standing still. Missing ball data alone never establishes D.
Once established, D persists until confirmed control restarts play or evidence
expires after eight seconds. This persistence is a heuristic, not referee detection.

### Confirmed control

A team is a control candidate if its closest player is within 2.5 metres of the
reliably observed ball, at least 0.75 metres closer than the opposing nearest player,
and ball speed is at most 12 m/s. The candidate must remain consistent for 0.4 seconds.
Once confirmed, the takeover is assigned back to the first frame of that continuous
candidate run. Ties, fast passes and momentary touches do not establish a new owner.
The most recent controller is retained during ambiguous loose-ball intervals.

### Automatic phase evidence and uncertainty

Attacking tags support their team; defensive tags support its opponent. Same-team
intervals are capped and opposing evidence cancels. Tags never force a takeover
against established control and cannot restart a dead-ball state. Consistent tags
can initialise unknown live possession after one second without a geometric
candidate, and same-owner tags refresh the current evidence timestamp.

After eight seconds without supporting evidence, the state becomes X. This is an
explicit uncertainty policy, not a claim that the ball is dead. The official scorer
charges X as a miss under macroF1 and fills it temporally for spellF1, so abstention
is not free and can carry the wrong owner through a real transition.

## Parameters and development discipline

All constants are explicit in `solution/control.py::CONFIG_V2`. They are engineering
assumptions, not learned estimates or parameters tuned against this match's answer
key. No match-specific frame exceptions exist. V2 addresses synthetic requirements:
no forced owner on entirely missing input, no unsupported stoppage, no fast-pass
interception, no phase-tag override, correct sustained takeover, and team symmetry.

V1 had a public five-minute spellF1 of 0.227027 and macroF1 of 0.522753. These are
historical v1 results, not v2 scores. No v2 public-score search, comparison or
selection was performed. The five-minute labels are not training/tuning data.

## Verification

The synthetic suite covers control, deflection, rapid passes, goalkeeper holding,
unknown input, occlusions, teleports, stoppages, conflicting tags, team symmetry,
FPS scaling, parser alignment, canonical CSV bytes and the legacy decoder optimum.
No synthetic test contains a competition tracking excerpt, frame label or video.
The official validator checks output format, not prediction accuracy.
Independent generation and clean-checkout reproduction are checked by SHA-256.

## Limitations

One-touch sequences or brief genuine control may be missed. Projection noise and
missing ball height affect geometric and speed tests. The programme does not detect
whistles or referee gestures. Out-of-field coordinates can be tracking errors; set
piece intervals can be inaccurate. Stationary-restart timing may precede/follow
actual play. D persistence may delay recognising a restart. Tags can preserve a
wrong owner. Long unknown spans may be filled incorrectly by the official scorer.
Synthetic correctness is not evidence of competitive match-level performance.

## Reproduction

Use Python 3.11 and `requirements-local.txt`. With authorised assets in
`data/tactical/`, run:

```text
python -m solution.predict --team "Man usuk" --method v2
python evaluation/validate_submission.py submission/Man_usuk.csv
```

The output is a 199,319-row UTF-8 CSV with canonical LF endings. A local manifest
records input/source/output hashes, parameters and environment versions. No random
sampling, network access, training, ground-truth reading or hand correction occurs
in prediction. `README.md` provides full setup. The public code package excludes
match assets, local manifests and predictions.

## Attribution and AI assistance

Official starter/loaders/validator: https://github.com/metrica-sports/more-than-a-hack-2027
(MIT-licensed code; competition assets have separate restricted terms).
AI assistance was used for code, synthetic tests and documentation from public
specifications. Restricted asset contents were processed locally, not sent to a
hosted LLM as input. The code never reads the private ground truth.
