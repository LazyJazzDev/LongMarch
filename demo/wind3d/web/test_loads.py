import json
import math
import struct
import unittest

from loads import ROW, window_packet


def packet(rows, generation=0, events=None):
    meta = dict(version=1, run='test', generation=generation, count=len(rows), stride=44,
                retention_seconds=2, capacity_limited=False, controls_events=events or [])
    header = json.dumps(meta).encode()
    return struct.pack('<I', len(header)) + header + b''.join(ROW.pack(*row) for row in rows)


def unpack(data):
    length = struct.unpack_from('<I', data)[0]
    return json.loads(data[4:4 + length]), list(ROW.iter_unpack(data[4 + length:]))


class HistoryWindowTest(unittest.TestCase):
    def test_full_resolution_and_weighted_statistics(self):
        # Unequal time steps: a packet arrival average would give the wrong answer.
        rows = [(1, .01, .01, 1, 0, 0, 0, 2, 0), (2, .03, .02, 3, 0, 0, 0, 4, 0),
                (3, .04, .01, 9, 0, 0, 0, 8, 0)]
        meta, samples = unpack(window_packet(packet(rows), .1))
        self.assertEqual([row[0] for row in samples], [1, 2, 3])
        self.assertAlmostEqual(meta['statistics']['force']['x']['mean'], 4, places=6)
        self.assertAlmostEqual(meta['statistics']['force']['x']['std'], 3, places=6)
        self.assertAlmostEqual(meta['statistics']['force']['norm']['peak_to_peak'], 8)
        self.assertAlmostEqual(meta['statistics']['moment']['norm']['mean'], 4.5)
        self.assertTrue(meta['contiguous'])

    def test_window_keeps_a_single_step_spike(self):
        rows = [(i, i*.0001, .0001, 100 if i == 950 else 0, 0, 0, 0, 0, 0) for i in range(1, 1001)]
        meta, selected = unpack(window_packet(packet(rows), .05))
        self.assertGreaterEqual(len(selected), 500)
        self.assertEqual(meta['last_step'], 1000)
        self.assertEqual(meta['statistics']['force']['norm']['max'], 100)
        self.assertEqual(sum(row[3] == 100 for row in selected), 1)
        self.assertTrue(meta['contiguous'])

    def test_restart_and_gaps_are_explicit(self):
        rows = [(1, .001, .001, 0, 0, 0, 0, 0, 0), (3, .003, .001, 0, 0, 0, 0, 0, 0)]
        meta, _ = unpack(window_packet(packet(rows, generation=7), .25))
        self.assertEqual(meta['generation'], 7)
        self.assertFalse(meta['contiguous'])
        self.assertTrue(math.isfinite(meta['statistics']['force']['norm']['std']))

    def test_startup_exclusion_preserves_later_spikes_and_raw_data(self):
        rows = [(i, i*.001, .001, 100000 if i <= 20 else (99 if i == 40 else 1), 0, 0, 0, 0, 0) for i in range(1, 51)]
        data = packet(rows, generation=3)
        meta, selected = unpack(window_packet(data, .1, 20))
        self.assertEqual(len(selected), 30)
        self.assertEqual(meta['first_step'], 21)
        self.assertEqual(meta['skipped_in_window'], 20)
        self.assertEqual(meta['statistics']['force']['x']['max'], 99)
        self.assertAlmostEqual(meta['statistics']['force']['x']['mean'], (29+99)/30, places=5)
        raw, _ = unpack(window_packet(data, .1, 0))
        self.assertEqual(raw['count'], 50)
        self.assertEqual(raw['statistics']['force']['x']['max'], 100000)
        warm, selected = unpack(window_packet(packet(rows[:5], generation=4), .1, 20))
        self.assertEqual(selected, [])
        self.assertTrue(warm['warming_up'])
        self.assertEqual(warm['generation'], 4)

    def test_invalid_packet_rejected(self):
        rows = [(1, .001, .001, 0, 0, 0, 0, 0, 0)]
        with self.assertRaises(ValueError):
            window_packet(packet(rows)[:-1], .25)


if __name__ == '__main__':
    unittest.main()
