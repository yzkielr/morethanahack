"""Control-confirmation v2: separate stoppages, control and uncertain observations.

All thresholds are engineering assumptions, tested on synthetic scenarios, not
selected using competition labels. No match-specific frame constants are used.
"""
import numpy as np


CONFIG_V2 = {
    'control_distance_m': 2.5,
    'opponent_margin_m': 0.75,
    'max_control_ball_speed_mps': 12.0,
    'max_plausible_ball_speed_mps': 45.0,
    'velocity_interval_s': 0.2,
    'confirm_control_s': 0.4,
    'confirm_outside_s': 0.4,
    'outside_margin_m': 0.5,
    'stationary_speed_mps': 0.5,
    'quiet_players_mps': 1.2,
    'confirm_stationary_s': 1.5,
    'max_unobserved_s': 8.0,
    'tag_acquire_s': 1.0,
}


def frame_count(seconds, fps):
    return max(1, int(round(seconds * fps)))


def sustained(mask, minimum):
    """Mark entire qualifying runs, with no filling across missing observations."""
    out = np.zeros(len(mask), dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, end in zip(edges[::2], edges[1::2]):
        if end - start >= minimum:
            out[start:end] = True
    return out


def motion(ball, fps, config):
    """Backward finite differences only within fully observed intervals."""
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    lag = frame_count(config['velocity_interval_s'], fps)
    speed = np.full(n, np.nan)
    # Reject both ends of impossible adjacent jumps; this is a reliability mask,
    # not a manually corrected trajectory or interpolation over missing frames.
    jump = np.zeros(n, dtype=bool)
    adjacent = np.linalg.norm(np.diff(ball, axis=0), axis=1) * fps
    bad = np.isfinite(adjacent) & (adjacent > config['max_plausible_ball_speed_mps'])
    jump[:-1] |= bad
    jump[1:] |= bad
    reliable = observed & ~jump
    if n > lag:
        invalid = np.r_[0, np.cumsum(~reliable)]
        valid_window = (invalid[lag+1:] - invalid[:n-lag]) == 0
        delta = np.linalg.norm(ball[lag:] - ball[:-lag], axis=1) * fps / lag
        speed[lag:] = np.where(valid_window, delta, np.nan)
    return reliable, speed


def restart_mask(events, n):
    result = np.zeros(n, dtype=bool)
    for event in events:
        if (event['code'] or '').strip().upper() == 'SET PIECES':
            lo = max(0, min(n, event['start_frame']))
            hi = max(0, min(n, event['end_frame']))
            if hi > lo:
                result[lo:hi] = True
    return result


def predict_control(ball, distances, activity, prior, events, pitch, fps, config=CONFIG_V2):
    n = len(ball)
    if n == 0 or fps <= 0:
        raise ValueError('Nonempty tracking and positive fps required')
    if distances.shape != (n, 2) or prior.shape != (n, 2) or activity.shape != (n,):
        raise ValueError('Feature shapes do not match video timeline')
    reliable, speed = motion(ball, fps, config)
    margin = config['outside_margin_m']
    outside = reliable & ((ball[:, 0] < -margin) | (ball[:, 0] > pitch[0]+margin)
                          | (ball[:, 1] < -margin) | (ball[:, 1] > pitch[1]+margin))
    stationary_restart = (reliable & (speed < config['stationary_speed_mps'])
                          & (activity < config['quiet_players_mps'])
                          & restart_mask(events, n))
    dead = (sustained(outside, frame_count(config['confirm_outside_s'], fps))
            | sustained(stationary_restart, frame_count(config['confirm_stationary_s'], fps)))
    # Geometry may establish control only with a usable motion estimate. A fast
    # pass near an opponent is not, by itself, evidence that they gained control.
    candidate = np.full(n, -1, dtype=int)
    moving = reliable & np.isfinite(speed) & (speed <= config['max_control_ball_speed_mps']) & ~outside
    for team in (0, 1):
        close = distances[:, team] <= config['control_distance_m']
        # Avoid inf-inf warnings and prevent ties establishing either side.
        separated = distances[:, 1-team] > distances[:, team] + config['opponent_margin_m']
        candidate[moving & close & separated] = team
    labels = np.full(n, 'X', dtype='<U1')
    owner = -1
    pending = -1
    start = 0
    count = 0
    tag_pending = -1
    tag_start = 0
    tag_count = 0
    last_evidence = -n
    is_dead = False
    confirm = frame_count(config['confirm_control_s'], fps)
    tag_confirm = frame_count(config['tag_acquire_s'], fps)
    timeout = frame_count(config['max_unobserved_s'], fps)
    for i in range(n):
        if dead[i]:
            labels[i] = 'D'
            owner, pending, count = -1, -1, 0
            tag_pending, tag_count = -1, 0
            is_dead = True
            last_evidence = i
            continue
        c = candidate[i]
        if c >= 0:
            if pending == c:
                count += 1
            else:
                pending, start, count = c, i, 1
            if count >= confirm:
                if owner != c or is_dead:
                    labels[start:i+1] = 'HA'[c]
                owner = c
                is_dead = False
                last_evidence = i
        else:
            pending, count = -1, 0
        # Tags may initialise an unknown live segment, never force a takeover
        # against an established controller or force a restart from D.
        t = int(np.argmax(prior[i])) if np.max(prior[i]) > 0 else -1
        if owner < 0 and not is_dead and t >= 0 and c < 0 and not outside[i]:
            if tag_pending == t:
                tag_count += 1
            else:
                tag_pending, tag_start, tag_count = t, i, 1
            if tag_count >= tag_confirm:
                owner = t
                last_evidence = i
                labels[tag_start:i+1] = 'HA'[t]
        else:
            tag_pending, tag_count = -1, 0
        if owner >= 0 and t == owner:
            last_evidence = i
        if i - last_evidence > timeout:
            owner = -1
            is_dead = False
        if is_dead:
            labels[i] = 'D'
        elif owner >= 0:
            labels[i] = 'HA'[owner]
    return labels
