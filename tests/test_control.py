"""Football scenarios invented independently of the competition match and labels."""
import unittest
import numpy as np
from solution.control import predict_control, motion, CONFIG_V2


def features(n=250, fps=25):
    ball = np.column_stack((10 + np.arange(n)*3/fps, np.full(n, 34.)))
    distances = np.tile([.5, 8.], (n, 1))
    activity = np.full(n, 2.)
    prior = np.zeros((n, 2))
    return ball, distances, activity, prior


def predict(f, events=(), fps=25):
    return predict_control(*f, events, (105., 68.), fps)


class ControlTests(unittest.TestCase):
    def test_stationary_goalkeeper_is_not_automatically_dead(self):
        f = features()
        f[0][:] = [5., 34.]
        f[2][:] = .2
        labels = predict(f)
        self.assertTrue(np.all(labels[20:] == 'H'))
        self.assertNotIn('D', labels)

    def test_total_missingness_does_not_invent_possession_or_stoppage(self):
        f = features(500)
        f[0][:] = np.nan
        f[1][:] = np.inf
        self.assertTrue(np.all(predict(f) == 'X'))

    def test_short_touch_keeps_owner_but_control_changes_owner(self):
        f = features()
        f[1][70:75] = [8., .5]
        f[1][150:] = [8., .5]
        labels = predict(f)
        self.assertTrue(np.all(labels[20:150] == 'H'))
        self.assertTrue(np.all(labels[150:] == 'A'))

    def test_fast_pass_near_opponent_does_not_change_control(self):
        f = features()
        speed = np.full(250, 3.)
        speed[80:110] = 20.
        f[0][:, 0] = 10 + np.cumsum(speed)/25
        f[1][85:105] = [8., .5]
        self.assertTrue(np.all(predict(f)[20:] == 'H'))

    def test_conflicting_phase_tag_cannot_override_confirmed_control(self):
        f = features()
        f[3][60:] = [-1, 1]
        self.assertTrue(np.all(predict(f)[20:] == 'H'))

    def test_short_occlusion_keeps_owner_long_occlusion_abstains(self):
        f = features(500)
        f[0][50:75] = np.nan
        f[1][50:75] = np.inf
        f[0][150:450] = np.nan
        f[1][150:450] = np.inf
        labels = predict(f)
        self.assertTrue(np.all(labels[30:100] == 'H'))
        self.assertTrue(np.all(labels[360:450] == 'X'))
        self.assertNotIn('D', labels)
        self.assertEqual(labels[-1], 'H')

    def test_outside_is_dead(self):
        f = features()
        f[0][:] = [-2., 30.]
        self.assertTrue(np.all(predict(f) == 'D'))

    def test_stationary_restart_requires_independent_activity_evidence(self):
        f = features()
        f[0][:] = [20., 34.]
        events = [dict(code='SET PIECES', start_frame=0, end_frame=250)]
        active = predict(f, events)
        self.assertNotIn('D', active)
        f[2][:] = .2
        quiet = predict(f, events)
        self.assertTrue(np.all(quiet[20:] == 'D'))

    def test_motion_does_not_bridge_occlusion_or_accept_teleport(self):
        f = features()
        f[0][30:35] = np.nan
        f[0][80] = [100, 60]
        reliable, speed = motion(f[0], 25, CONFIG_V2)
        self.assertFalse(reliable[80])
        self.assertTrue(np.isnan(speed[30:40]).all())

    def test_no_bias_when_swapping_teams(self):
        f = features()
        f[1][100:] = [8., .5]
        original = predict(f)
        swapped = predict((f[0], f[1][:, ::-1], f[2], f[3][:, ::-1]))
        self.assertEqual(''.join(original).translate(str.maketrans('HA', 'AH')), ''.join(swapped))

    def test_time_rules_scale_with_fps(self):
        at25 = predict(features(250, 25), fps=25)
        at50 = predict(features(500, 50), fps=50)
        np.testing.assert_array_equal(at25[20:], at50[40::2])


if __name__ == '__main__':
    unittest.main()
