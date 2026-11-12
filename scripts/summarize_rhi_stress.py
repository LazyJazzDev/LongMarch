"""Summarize complete paired stress runs without discarding outliers."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics


def estimate(pairs, metric, rng):
    logs = [math.log(p['B'][metric] / p['A'][metric]) for p in pairs]
    blocks = sorted({p['A']['block'] for p in pairs})
    strata = [[logs[i] for i, p in enumerate(pairs) if p['A']['block'] == b] for b in blocks]
    boots = sorted(math.exp(statistics.mean([v for group in strata for v in rng.choices(group, k=len(group))]))
                   for _ in range(20000))
    return {'pairs': len(pairs), 'ratio': math.exp(statistics.mean(logs)),
            'ci95': [boots[499], boots[19499]],
            'ci98_75': [boots[124], boots[19874]],
            'min_ratio': math.exp(min(logs)), 'max_ratio': math.exp(max(logs)),
            'block_ratios': [math.exp(statistics.mean(group)) for group in strata],
            'mean_ms_per_frame': {label: statistics.mean(p[label][metric] * 1000 / p[label]['frames'] for p in pairs)
                                  for label in ['A', 'B']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    plan = json.loads((directory / 'plan.json').read_text())
    completion = json.loads((directory / 'complete.json').read_text())
    assert plan['hashes'] == completion['hashes']
    rows = [json.loads(line) for line in (directory / 'samples.jsonl').read_text().splitlines()]
    assert len(rows) == 160, len(rows)
    results = {}
    rng = random.Random(1142026)
    table = ['| Backend / workload | Main ms/frame | PR ms/frame | Time change (95% CI) | Main / PR recording ms |',
             '| --- | ---: | ---: | ---: | ---: |']
    for case in plan['cases']:
        api, profile = case.split('/')
        result = {}
        for mode, count in [('comparison', 8), ('control', 2)]:
            selected = [r for r in rows if r['api'] == api and r['profile'] == profile and r['mode'] == mode]
            pairs = []
            for block in range(2):
                for pair in range(count):
                    pair_rows = [r for r in selected if r['block'] == block and r['pair'] == pair]
                    assert len(pair_rows) == 2 and {r['label'] for r in pair_rows} == {'A', 'B'}
                    assert len({r['frames'] for r in pair_rows}) == 1
                    assert all(r['actual_binary'] == ('A' if mode == 'control' else r['label']) for r in pair_rows)
                    pairs.append({r['label']: r for r in pair_rows})
            result[mode] = {metric: estimate(pairs, metric, rng) for metric in ['seconds', 'record_seconds']}
        results[case] = result
        d = result['comparison']['seconds']; r = result['comparison']['record_seconds']['mean_ms_per_frame']
        avg = d['mean_ms_per_frame']; lo, hi = [100 * (x - 1) for x in d['ci95']]
        table.append(f"| {case} | {avg['A']:.3f} | {avg['B']:.3f} | {100*(d['ratio']-1):+.2f}% [{lo:+.2f}, {hi:+.2f}] | {r['A']:.3f} / {r['B']:.3f} |")
        print(case, json.dumps(result['comparison']['seconds']), flush=True)
    # Keep the exact timings and run order; commands are reconstructed from the plan.
    compact = [{k: v for k, v in row.items() if k not in ['command', 'backend', 'grid', 'width', 'height',
                'iterations', 'in_flight', 'warmup', 'draws_per_frame', 'submissions_per_frame']} for row in rows]
    report = {'plan': plan, 'summary': results, 'samples': compact,
              'interval_method': '20,000 paired bootstrap resamples in log-ratio space, stratified by the two blocks; no samples removed. ci98_75 is a Bonferroni-adjusted interval for four wall-time cases. Intervals do not cover systematic machine/session effects.'}
    (directory / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (directory / 'table.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    print('\n'.join(table))


if __name__ == '__main__':
    main()
