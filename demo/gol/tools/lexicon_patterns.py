#!/usr/bin/env python3
"""Builds the built-in Game of Life pattern library from the Life Lexicon.

Input: the plaintext Life Lexicon (life-lexicon-nowrap-plaintext.txt from
https://github.com/dvgrn/life-lexicon), CC BY-SA 3.0, Stephen A. Silver et al.
Output: a JSON catalog of named patterns that fit the 256 x 256 grid, each
classified by simulating it: still lifes and oscillators return to their
cells, spaceships return shifted. Patterns that simulation cannot classify
(guns, puffers, methuselahs) fall back to keywords in their definitions.

Usage: lexicon_patterns.py <lexicon.txt> <output.json>
"""

import json
import re
import sys
from fractions import Fraction

import numpy as np

MAX_SIZE = 256
MAX_GENERATIONS = 1024
MAX_EXTENT = 1024
MAX_POPULATION = 50000

CATEGORIES = ['still', 'oscillator', 'spaceship', 'gun', 'puffer', 'methuselah', 'other']


def parse(text):
    """Yields (name, definition, rows) for every pattern block in the lexicon."""
    body = text.split('\nLIFE LEXICON', 1)[1] if '\nLIFE LEXICON' in text else text
    name, definition, rows, blocks = None, '', [], {}
    for line in body.replace('\r', '').split('\n'):
        entry = re.match(r'^:([^:]+):(.*)$', line)
        if entry:
            if name and rows:
                yield name, definition, rows
            name, definition, rows = entry.group(1).strip(), entry.group(2).strip(), []
            continue
        if name is None:
            continue
        if re.match(r'^\t[.*]+\s*$', line) or re.match(r'^ {8}[.*]+\s*$', line):
            rows.append(line.strip())
        else:
            if rows:
                yield name, definition, rows
                rows = []
            if line.startswith('   '):
                definition += ' ' + line.strip()
    if name and rows:
        yield name, definition, rows


def to_array(rows):
    width = max(len(row) for row in rows)
    cells = np.zeros((len(rows), width), dtype=bool)
    for y, row in enumerate(rows):
        for x, c in enumerate(row):
            cells[y, x] = c == '*'
    ys, xs = np.nonzero(cells)
    if len(xs) == 0:
        return None
    return cells[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def step(cells):
    """One generation on an unbounded plane; returns cells and the origin shift."""
    padded = np.pad(cells, 1)
    neighbors = sum(np.roll(np.roll(padded, dy, 0), dx, 1)
                    for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
    alive = (neighbors == 3) | (padded & (neighbors == 2))
    ys, xs = np.nonzero(alive)
    if len(xs) == 0:
        return alive[:0, :0], (0, 0)
    y0, x0 = ys.min(), xs.min()
    return alive[y0:ys.max() + 1, x0:xs.max() + 1], (int(x0) - 1, int(y0) - 1)


# Small patterns that settle into a cycle or vanish only after this many
# generations are methuselahs; larger transients such as fuses and reactions
# are left to their definitions.
METHUSELAH_GENERATIONS = 50
METHUSELAH_POPULATION = 20


def classify(cells):
    """Returns (category, period, (dx, dy)) from simulation, or (None, None, None)."""
    start = cells
    x, y = 0, 0
    current = cells
    seen = {(start.shape, start.tobytes()): 0}
    small = start.sum() <= METHUSELAH_POPULATION
    for generation in range(1, MAX_GENERATIONS + 1):
        current, (dx, dy) = step(current)
        x, y = x + dx, y + dy
        if current.size == 0:
            return ('methuselah', None, None) if small and generation >= METHUSELAH_GENERATIONS else (None, None, None)
        if max(current.shape) > MAX_EXTENT or current.sum() > MAX_POPULATION:
            return None, None, None
        if current.shape == start.shape and np.array_equal(current, start):
            if x == 0 and y == 0:
                return ('still' if generation == 1 else 'oscillator'), generation, (0, 0)
            return 'spaceship', generation, (x, y)
        key = (current.shape, current.tobytes())
        if key in seen:
            # Settled into a still life, oscillator or spaceship after a transient.
            if small and seen[key] >= METHUSELAH_GENERATIONS:
                return 'methuselah', None, None
            return None, None, None
        seen[key] = generation
    return None, None, None


def speed(period, displacement):
    dx, dy = (abs(v) for v in displacement)
    shift = max(dx, dy)
    ratio = Fraction(shift, period)
    text = 'c' if ratio == 1 else f'{ratio.numerator}c/{ratio.denominator}' if ratio.numerator > 1 else f'c/{ratio.denominator}'
    if dx and dy:
        return text + (' diagonal' if dx == dy else ' oblique')
    return text + ' orthogonal'


def keyword_category(name, definition):
    text = (name + ' ' + definition[:300]).lower()
    if re.search(r'\bgun\b|\{gun\}', text):
        return 'gun'
    if re.search(r'puffer|rake|breeder|spacefiller|\{wick\}', text):
        return 'puffer'
    if re.search(r'methuselah|stabiliz', text):
        return 'methuselah'
    return 'other'


def rle(cells):
    lines = []
    for row in cells:
        runs, previous, count = [], None, 0
        for c in row:
            symbol = 'o' if c else 'b'
            if symbol == previous:
                count += 1
            else:
                if previous:
                    runs.append((count, previous))
                previous, count = symbol, 1
        if previous == 'o':
            runs.append((count, previous))
        lines.append(''.join(f'{n if n > 1 else ""}{s}' for n, s in runs))
    # Merge empty rows into row-break counts.
    out, breaks = [], 0
    for line in lines:
        if line:
            if out:
                out.append(f'{breaks if breaks > 1 else ""}$')
            out.append(line)
            breaks = 1
        else:
            breaks += 1
    return ''.join(out) + '!'


def main():
    source, output = sys.argv[1], sys.argv[2]
    text = open(source, encoding='latin-1').read()
    seen, catalog, counts = set(), [], {}
    names = {}
    for name, definition, rows in parse(text):
        cells = to_array(rows)
        if cells is None or cells.shape[0] > MAX_SIZE or cells.shape[1] > MAX_SIZE:
            continue
        key = rle(cells)
        if key in seen:
            continue
        seen.add(key)
        category, period, displacement = classify(cells)
        entry = {'name': name, 'width': int(cells.shape[1]), 'height': int(cells.shape[0])}
        if category is None:
            category = keyword_category(name, definition)
        else:
            entry['period'] = period
            if category == 'spaceship':
                entry['speed'] = speed(period, displacement)
        entry['category'] = category
        entry['rle'] = key
        # Later blocks of one entry show variants or stages of the same object.
        names[name] = names.get(name, 0) + 1
        if names[name] > 1:
            entry['name'] = f'{name} ({names[name]})'
        catalog.append(entry)
        counts[category] = counts.get(category, 0) + 1
    catalog.sort(key=lambda e: (CATEGORIES.index(e['category']), e['name'].lower()))
    document = {
        'source': 'Life Lexicon, https://github.com/dvgrn/life-lexicon (plaintext release of 2019-10-29)',
        'license': 'CC BY-SA 3.0; the Life Lexicon is copyright Stephen A. Silver, 1997-2018, '
                   'updated by Dave Greene and David Bell',
        'patterns': catalog,
    }
    with open(output, 'w', encoding='utf-8') as file:
        json.dump(document, file, ensure_ascii=False, separators=(',', ':'))
        file.write('\n')
    print(len(catalog), 'patterns', counts)


if __name__ == '__main__':
    main()
