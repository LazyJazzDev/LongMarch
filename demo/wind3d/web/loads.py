"""Select and summarize a complete, per-solver-step load window on the relay."""
import json
import math
import struct

ROW = struct.Struct('<Qdf6f')
WINDOWS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)


def window_packet(data, seconds, skip_steps=0):
    length = struct.unpack_from('<I', data)[0]
    meta = json.loads(data[4:4 + length])
    body = memoryview(data)[4 + length:]
    if meta['version'] != 1 or meta['stride'] != ROW.size or len(body) != meta['count'] * ROW.size:
        raise ValueError('Invalid load-history packet')
    rows = list(ROW.iter_unpack(body))
    if not rows:
        raise ValueError('Empty load-history packet')
    cutoff = rows[-1][1] - seconds
    first = next((i for i, row in enumerate(rows) if row[1] >= cutoff and row[0] > skip_steps), len(rows))
    meta.update(skip_steps=skip_steps, skipped_in_window=sum(row[0] <= skip_steps and row[1] >= cutoff for row in rows))
    if first == len(rows):
        meta.update(count=0, warming_up=True, last_step=rows[-1][0], requested_seconds=seconds)
        header = json.dumps(meta, separators=(',', ':'), allow_nan=False).encode()
        return struct.pack('<I', len(header)) + header
    selected = rows[first:]
    # The first event describes the retained baseline, not a new input change.
    events = meta.get('controls_events', [])
    changes = [event['step'] for event in events[1:]]
    invalid = set()
    for step in changes:
        invalid.update(row[0] for row in selected if step <= row[0] < step + 20)
    valid = [row for row in selected if row[0] not in invalid]
    meta.update(invalid_indices=[i for i, row in enumerate(selected) if row[0] in invalid],
                valid_count=len(valid), change_skip_steps=20)
    # Weighted online variance remains accurate even for tiny fluctuations on a large mean.
    names = ('norm', 'x', 'y', 'z')
    stats = {kind: {name: dict(mean=0.0, min=math.inf, max=-math.inf, std=0.0)
                    for name in names} for kind in ('force', 'moment')}
    weight = 0.0
    for row in valid:
        dt = max(0.0, min(row[2], row[1] - cutoff))
        new_weight = weight + dt
        for kind, vector in (('force', row[3:6]), ('moment', row[6:9])):
            for name, value in zip(names, (math.hypot(*vector), *vector)):
                stat = stats[kind][name]
                stat['min'] = min(stat['min'], value)
                stat['max'] = max(stat['max'], value)
                if new_weight > 0:
                    delta = value - stat['mean']
                    stat['mean'] += delta * dt / new_weight
                    stat['std'] += dt * delta * (value - stat['mean'])
        weight = new_weight
    for group in stats.values():
        for stat in group.values():
            stat['std'] = math.sqrt(max(0.0, stat['std'] / weight)) if weight else 0.0
            stat['peak_to_peak'] = stat['max'] - stat['min']
            if not valid:
                stat.update(mean=None, min=None, max=None, std=None, peak_to_peak=None)
    events = meta.get('controls_events', [])
    active = [event for event in events if event['step'] <= selected[0][0]]
    meta['controls_events'] = active[-1:] + [event for event in events if event['step'] > selected[0][0]]
    meta.update(count=len(selected), requested_seconds=seconds, start=selected[0][1], end=selected[-1][1],
                first_step=selected[0][0], last_step=selected[-1][0],
                dt_min=min(row[2] for row in selected), dt_max=max(row[2] for row in selected),
                contiguous=all(b[0] == a[0] + 1 for a, b in zip(selected, selected[1:])),
                statistics=stats, source='every_solver_step')
    header = json.dumps(meta, separators=(',', ':'), allow_nan=False).encode()
    return struct.pack('<I', len(header)) + header + body[first * ROW.size:].tobytes()
