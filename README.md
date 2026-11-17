# Man usuk — More Than a Hack 2027

A reproducible Phase 1 possession-estimation project using the official tactical
tracking and automatic Smart Tagging. Team: **Man usuk**.

The default method, **control-confirmation v2**, separates stoppage evidence from
team control. It confirms sustained control, avoids treating fast passes as
opponent takeovers, and abstains when evidence expires. This is an experimental
heuristic method, not a proven top-20 or finalist solution.

## Quick start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m solution.predict --team "Man usuk" --data-dir "C:\path\to\dataset"
.\.venv\Scripts\python.exe evaluation/validate_submission.py submission/Man_usuk.csv
```

The dataset directory must contain a `tactical` subfolder with:

- `BAR-ATH_tactical_atd.xml`
- `BAR-ATH_tactical_atd.txt`
- `BAR-ATH_tactical_smart_tagging.xml`

Alternatively place these in `data/tactical/` and omit `--data-dir`.
Python 3.11 is tested. On Linux/macOS use `.venv/bin/python` instead of the Windows path.
Videos, broadcast tracking and homographies are not used by this implementation.

## Outputs

- `submission/Man_usuk.csv`: 199,319 frame/label rows, UTF-8 with canonical LF endings.
- `submission/Man_usuk.manifest.json`: local input/source/output hashes and configuration.

Labels are H (home), A (away), D (dead ball) or X (abstain). The command validates
its output with the official validator. It does not upload anything or submit the form.

## How it works

1. Parse every tactical tracking frame once; convert positions to metres.
2. Reject implausible ball jumps and estimate motion only over fully observed intervals.
3. Detect sustained out-of-field observations or a stationary restart supported by low player activity.
4. Confirm close, unambiguous, sufficiently slow-ball control for 0.4 seconds before a takeover.
5. Keep the last controller through short loose-ball/occlusion intervals; abstain after evidence expires.
6. Use automatic tactical tags only to acquire unknown live possession or support the same owner.

All constants are engineering assumptions. No model is trained, no public-answer
parameter search is performed, and prediction never reads ground truth.

## Verification and limitations

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover invented football scenarios, parser alignment, sequence decoding and
cross-platform CSV bytes. CI runs synthetic tests only, with no match assets.
Passing tests proves those cases work; it does not establish match accuracy.

V2 has **not been ranked using the public five-minute labels**. V1 scored public
spellF1 0.227027 (below the published public smooth50 baseline). V1 is retained via
`--method v1`, with its original report in `docs/technical-v1.md`. V2 development
addresses independently specified synthetic cases; no v2 public-score sweep or
selection against that answer window was performed. Full-match performance remains unknown.

Current limitations include one-touch football, whistle-driven stoppages, long
occlusions, automatic tag errors and ambiguous restart boundaries. Details and all
parameter values are in [the technical document](Man_usuk.pdf).

## Submission

Provide the organisers with:

1. `submission/Man_usuk.csv`
2. This public repository URL
3. `Man_usuk.pdf` (the required short technical document; Markdown is the editable source)

The raw dataset, videos, local manifests, logs and generated predictions are excluded
from this repository. Download match assets only through the organisers' authorised
channel. Do not send restricted match material to hosted LLMs.

## Attribution

Based on the [Metrica Sports official starter](https://github.com/metrica-sports/more-than-a-hack-2027).
Its licence, official evaluation code and reference documentation are retained.
See `UPSTREAM_README.md` for the upstream introduction. Code and competition data
have separate licences; this repository includes no licensed match assets.
