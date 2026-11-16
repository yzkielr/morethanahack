"""Offline three-state decoding. Never loads ground truth or uses the network."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re

import numpy as np

from games import GAME, GAMES, require, source_frames
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import load as load_tags


# Chosen from physical/time-scale assumptions, before any public evaluation.
CONFIG = {
    'control_radius_m': 3.0,
    'switch_penalty': 12.0,
    'stoppage_penalty': 8.0,
    'tag_weight': 0.65,
    'outside_margin_m': 0.5,
    'missing_grace_s': 4.0,
    'missing_ramp_s': 8.0,
    'stationary_window_s': 2.0,
    'stationary_step_m': 0.08,
}
ATTACK = {'BUILD UP', 'PROGRESSION', 'CREATING CHANCES',
          'ATTACKING TRANSITION', 'BALL IN FINAL THIRD', 'BALL IN THE BOX',
          'PLAYERS IN THE BOX', 'LONG BALL'}
DEFENCE = {'DEFENDING IN ATTACKING THIRD', 'DEFENDING IN MIDDLE THIRD',
           'DEFENDING IN DEFENSIVE THIRD', 'DEFENSIVE TRANSITION'}


def read_tracking(meta, raw, n_frames, pitch, with_activity=False):
    """Read 1-based video-frame IDs; never apply corrected-tracking offsets."""
    teams = team_of_each_track(meta)
    ball = np.full((n_frames, 2), np.nan)
    distances = np.full((n_frames, 2), np.inf)
    seen = np.zeros(n_frames, dtype=bool)
    activity = np.full(n_frames, np.nan)
    scale = np.asarray(pitch)
    with open(raw, encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            parts = line.strip().split(':')
            if len(parts) != 3:
                raise ValueError(f'Malformed tracking record at line {line_no}')
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames or seen[i]:
                raise ValueError(f'Invalid or duplicate frame at line {line_no}')
            seen[i] = True
            groups = parts[1].rstrip(';').split(';')
            if len(groups) != len(teams):
                raise ValueError(f'Track slot mismatch at line {line_no}')
            if with_activity:
                speeds = np.asarray([g.split(',')[2] for g in groups], dtype=float)
                speeds = speeds[np.isfinite(speeds) & (speeds >= 0) & (speeds <= 15)]
                if len(speeds) >= 10:
                    activity[i] = np.median(speeds)
            b = np.asarray(parts[2].rstrip(';').split(',')[:2], dtype=float)
            if b.shape != (2,):
                raise ValueError(f'Invalid ball coordinates at line {line_no}')
            if not np.isfinite(b).all():
                continue
            ball[i] = b * scale
            players = np.asarray([g.split(',')[:2] for g in groups], dtype=float) * scale
            d = np.linalg.norm(players - ball[i], axis=1)
            d[~np.isfinite(d)] = np.inf
            for team in (0, 1):
                distances[i, team] = np.min(d[teams == team], initial=np.inf)
    if not seen.all():
        raise ValueError('Tracking does not contain every expected video frame')
    return (ball, distances, activity) if with_activity else (ball, distances)


def tag_prior(events, n_frames):
    """Capped soft evidence; no copying tags into final labels, no restart timing guesses."""
    prior = np.zeros((n_frames, 2))
    # Documented names in official DATA.md, not labels inferred from restricted data.
    team_map = {'barcelona': 0, 'fc barcelona': 0,
                'atletic club': 1, 'athletic club': 1}
    used = 0
    for event in events:
        code = (event['code'] or '').strip().upper()
        team = team_map.get((event['tags'].get('Team') or '').strip().lower())
        if team is None or code not in ATTACK | DEFENCE:
            continue
        if code in DEFENCE:
            team = 1 - team
        start = max(0, min(n_frames, event['start_frame']))
        end = max(0, min(n_frames, event['end_frame']))
        if end <= start:
            continue
        prior[start:end, team] = 1.0
        used += 1
    # Conflicting tags cancel rather than arbitrarily giving one team priority.
    difference = prior[:, 0] - prior[:, 1]
    return np.column_stack((difference, -difference)), used


def rolling_mean(values, radius):
    prefix = np.r_[0., np.cumsum(values)]
    index = np.arange(len(values))
    left = np.maximum(0, index - radius)
    right = np.minimum(len(values), index + radius + 1)
    return (prefix[right] - prefix[left]) / (right - left)


def missing_depth(visible):
    """Distance to nearest observed ball; avoids calling brief occlusions dead balls."""
    n = len(visible)
    ix = np.arange(n)
    left = np.maximum.accumulate(np.where(visible, ix, -n))
    right = np.minimum.accumulate(np.where(visible, ix, 2*n)[::-1])[::-1]
    return np.minimum(ix-left, right-ix)


def emission_scores(ball, distances, prior, pitch, fps, config=CONFIG):
    n = len(ball)
    visible = np.isfinite(ball).all(axis=1)
    # Bounded proximity: aerial/loose balls cannot create arbitrarily strong evidence.
    proximity = np.exp(-np.minimum(distances, 100) / config['control_radius_m'])
    advantage = proximity[:, 0] - proximity[:, 1]
    scores = np.zeros((n, 3))
    scores[:, 0] = 1.8 * advantage
    scores[:, 1] = -1.8 * advantage
    scores[:, 2] = -0.7 * visible
    scores[:, :2] += config['tag_weight'] * prior
    # Missingness is weak stoppage evidence only deep inside a long gap.
    depth_s = missing_depth(visible) / fps
    gap = np.clip((depth_s - config['missing_grace_s']) / config['missing_ramp_s'], 0, 1)
    scores[:, 2] += 0.6 * gap
    # Coordinates beyond the field are stronger stoppage evidence, but still denoised.
    margin = config['outside_margin_m']
    outside = visible & ((ball[:, 0] < -margin) | (ball[:, 0] > pitch[0]+margin)
                         | (ball[:, 1] < -margin) | (ball[:, 1] > pitch[1]+margin))
    scores[outside, 2] += 3.0
    # A nearly stationary observed ball is only weak evidence: a player can hold it live.
    step = np.full(n, np.inf)
    if n > 1:
        delta = np.linalg.norm(np.diff(ball, axis=0), axis=1)
        step[1:] = np.where(np.isfinite(delta), delta, np.inf)
    radius = max(1, round(config['stationary_window_s'] * fps / 2))
    stationary = rolling_mean((step < config['stationary_step_m']).astype(float), radius)
    scores[:, 2] += 1.0 * (stationary > 0.9)
    return scores


def decode(scores, switch_penalty=12.0, stoppage_penalty=8.0):
    """Viterbi maximum-score sequence over H/A/D, fixed deterministic tie breaks."""
    if scores.ndim != 2 or scores.shape[1] != 3 or not len(scores):
        raise ValueError('Expected a nonempty N x 3 score array')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite emission score')
    penalty = np.array([[0., switch_penalty, stoppage_penalty],
                        [switch_penalty, 0., stoppage_penalty],
                        [stoppage_penalty, stoppage_penalty, 0.]])
    back = np.zeros(scores.shape, dtype=np.uint8)
    best = scores[0].copy()
    for i in range(1, len(scores)):
        candidates = best[:, None] - penalty
        back[i] = np.argmax(candidates, axis=0)
        best = candidates[back[i], np.arange(3)] + scores[i]
        best -= best.max()
    states = np.empty(len(scores), dtype=np.uint8)
    states[-1] = np.argmax(best)
    for i in range(len(scores)-1, 0, -1):
        states[i-1] = back[i, states[i]]
    return np.array(['H', 'A', 'D'])[states]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, labels):
    """Canonical UTF-8 / LF bytes on Windows and Linux."""
    with Path(path).open('w', encoding='utf-8', newline='') as handle:
        handle.write('frame,label\n')
        handle.writelines(f'{i},{label}\n' for i, label in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team', default='Man usuk')
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('submission'))
    parser.add_argument('--method', choices=['v1', 'v2'], default='v2')
    args = parser.parse_args()
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', args.team.strip()).strip('_')
    if not slug:
        parser.error('Team name must contain letters or digits')
    meta, raw, tags = require(['atd_meta', 'atd_raw', 'smart_tagging'], data_dir=args.data_dir)
    n = source_frames('tactical')
    pitch, fps = GAMES[GAME]['pitch'], GAMES[GAME]['fps']
    ball, distances, activity = read_tracking(meta, raw, n, pitch, with_activity=True)
    events = load_tags(data_dir=args.data_dir)
    prior, used = tag_prior(events, n)
    if not used:
        raise ValueError('No supported tactical tags matched; check input schema')
    if args.method == 'v1':
        config = CONFIG
        scores = emission_scores(ball, distances, prior, pitch, fps)
        labels = decode(scores, CONFIG['switch_penalty'], CONFIG['stoppage_penalty'])
    else:
        from solution.control import CONFIG_V2, predict_control
        config = CONFIG_V2
        labels = predict_control(ball, distances, activity, prior, events, pitch, fps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'{slug}.csv'
    write_csv(output, labels)
    # Validation only; this module never opens a ground-truth file.
    from evaluation.possession_metrics import read_label_csv, validate_submission
    reason = validate_submission(read_label_csv(str(output)), n)
    if reason:
        raise ValueError(reason)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root/'solution').glob('*.py')) + [root/'games.py',
               root/'starter/baseline.py', root/'starter/load_smart_tagging.py']
    manifest = {'team': args.team, 'method': args.method, 'config': config,
                'python': platform.python_version(), 'numpy': np.__version__,
                'rows': n, 'csv_sha256': sha256(output),
                'inputs_sha256': {Path(p).name: sha256(p) for p in (meta, raw, tags)},
                'source_sha256': {str(p.relative_to(root)): sha256(p) for p in sources},
                'ground_truth_used_for_prediction': False}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(f'Created {output}; official format validation passed ({n} rows).')


if __name__ == '__main__':
    main()
