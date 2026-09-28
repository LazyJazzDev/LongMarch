import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

root = Path(__file__).resolve().parents[1]
groups = defaultdict(list)
with (root / 'docs/reports/nsight-scene-update-frames.csv').open() as stream:
    for row in csv.DictReader(stream):
        groups[row['scene'], row['backend'], row['phase']].append(float(row['frame_ms']))
pixels = {'classroom': 1920 * 1080, 'junkshop': 2000 * 1000, 'monster': 1024 * 1024}
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
for ax, scene in zip(axes, pixels):
    for shift, phase, color in [(-0.18, 'before', '#64748b'), (0.18, 'after', '#098579')]:
        values = []
        for backend in ['d3d12', 'vulkan']:
            times = groups[scene, backend, phase]
            values.append(pixels[scene] * 8 / (sum(times) / len(times)) / 1000)
        bars = ax.bar([shift, 1 + shift], values, width=0.34, label=phase.title(), color=color)
        ax.bar_label(bars, labels=[f'{value:.1f}' for value in values], padding=3)
    ax.set_title(scene.title())
    ax.set_xticks([0, 1], ['D3D12', 'Vulkan'])
    ax.set_ylabel('Primary camera M Rays/s')
    ax.set_ylim(0, ax.get_ylim()[1] * 1.14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False)
fig.suptitle('Scene update optimization: measured throughput', fontsize=16)
fig.text(0.5, 0.035, 'RTX 3090 Ti | Native ray query | 8 samples/dispatch | 60 frames, first 10 discarded\n'
         'Original scene resolutions and bounce limits | One run per condition | 2026-09-27',
         ha='center', fontsize=9)
fig.tight_layout(rect=(0, 0.12, 1, 0.94))
output = root / 'assets/reports/nsight-scene-updates/throughput.png'
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=160)
print(output)
