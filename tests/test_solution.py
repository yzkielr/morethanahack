"""Synthetic tests only: no competition labels, video or tracking excerpts."""
import itertools
from pathlib import Path
import tempfile
import unittest

import numpy as np

from solution.predict import decode, emission_scores, missing_depth, read_tracking, tag_prior, write_csv


class SolutionTests(unittest.TestCase):
    def test_csv_is_canonical_across_operating_systems(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'synthetic.csv'
            write_csv(path, ['H', 'D', 'A', 'X'])
            self.assertEqual(path.read_bytes(), b'frame,label\n0,H\n1,D\n2,A\n3,X\n')

    def test_decoder_matches_exhaustive_optimum(self):
        rng = np.random.default_rng(8)
        for _ in range(8):
            scores = rng.normal(size=(5, 3))
            penalties = np.array([[0, 1.2, .8], [1.2, 0, .8], [.8, .8, 0]])
            def objective(states):
                return sum(scores[i, s] for i, s in enumerate(states)) - sum(
                    penalties[a, b] for a, b in zip(states, states[1:]))
            labels = decode(scores, 1.2, .8)
            actual = ['HAD'.index(x) for x in labels]
            expected = max(objective(s) for s in itertools.product(range(3), repeat=5))
            self.assertAlmostEqual(objective(actual), expected)

    def test_brief_touch_does_not_split_possession(self):
        scores = np.tile([1., -1., -1.], (200, 1))
        scores[95:100] = [-1., 1., -1.]
        self.assertTrue(np.all(decode(scores) == 'H'))

    def test_sustained_takeover_and_stoppage(self):
        scores = np.repeat([[2., -2., -2.], [-2., 2., -2.], [-2., -2., 2.]], 80, axis=0)
        np.testing.assert_array_equal(decode(scores), np.repeat(['H', 'A', 'D'], 80))

    def test_short_occlusion_preserves_possession(self):
        ball = np.column_stack((np.arange(200)*.2, np.full(200, 30.)))
        distances = np.tile([.5, 8.], (200, 1))
        ball[80:105] = np.nan
        distances[80:105] = np.inf
        scores = emission_scores(ball, distances, np.zeros((200, 2)), (105., 68.), 25)
        self.assertTrue(np.all(decode(scores) == 'H'))

    def test_outside_ball_is_dead(self):
        ball = np.tile([-3., 30.], (100, 1))
        distances = np.tile([1., 8.], (100, 1))
        scores = emission_scores(ball, distances, np.zeros((100, 2)), (105., 68.), 25)
        self.assertTrue(np.all(decode(scores) == 'D'))

    def test_tag_defence_inverts_and_conflict_cancels(self):
        events = [dict(code='BUILD UP', start_frame=-5, end_frame=5, tags={'Team':'Barcelona'}),
                  dict(code='DEFENSIVE TRANSITION', start_frame=2, end_frame=8,
                       tags={'Team':'Barcelona'}),
                  dict(code='SET PIECES', start_frame=0, end_frame=9, tags={'Team':'N/A'})]
        prior, count = tag_prior(events, 10)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(prior[0], [1, -1])
        np.testing.assert_array_equal(prior[3], [0, 0])
        np.testing.assert_array_equal(prior[6], [-1, 1])

    def test_missing_depth_and_invalid_scores(self):
        np.testing.assert_array_equal(missing_depth(np.array([True, False, False, True])), [0, 1, 1, 0])
        with self.assertRaises(ValueError):
            decode(np.array([[np.nan, 0, 0]]))

    def test_parser_frame_index_and_team_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            meta, raw = Path(folder)/'meta.xml', Path(folder)/'raw.txt'
            meta.write_text('<root><Team id="home"/><Team id="away"/>'
                            '<Player teamId="home"/><Player teamId="away"/></root>')
            raw.write_text('1:0.1,0.5,0;0.9,0.5,0:0.11,0.5,0\n'
                           '2:0.1,0.5,0;0.9,0.5,0:NaN,NaN,NaN\n')
            ball, distances = read_tracking(meta, raw, 2, (100, 60))
            np.testing.assert_allclose(ball[0], [11, 30])
            np.testing.assert_allclose(distances[0], [1, 79])
            self.assertTrue(np.isinf(distances[1]).all())
            raw.write_text('1:0.1,0.5,0;0.9,0.5,0:0.11,0.5,0\n'*2)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_tracking(meta, raw, 2, (100, 60))


if __name__ == '__main__':
    unittest.main()
