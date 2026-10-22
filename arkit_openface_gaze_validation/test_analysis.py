import unittest

import numpy as np
import pandas as pd

from analyze_gaze import audio_offset, derive_arkit_gaze, relative_timecodes
from compare_gaze_corrections import head_to_world_rotation, remove_head_rotation


class AnalysisTests(unittest.TestCase):
    def test_timecode_sampling(self):
        times_60 = relative_timecodes(
            pd.Series(["01:10:05:29.500", "01:10:05:30.500", "01:10:05:31.500"])
        )
        times_30 = relative_timecodes(
            pd.Series(["01:10:05:29.500", "01:10:05:31.500", "01:10:05:33.500"])
        )
        np.testing.assert_allclose(np.diff(times_60), 1 / 60)
        np.testing.assert_allclose(np.diff(times_30), 1 / 30)

    def test_gaze_driver_directions(self):
        frame = pd.DataFrame(
            {
                "eyeLookInLeft": [0.8],
                "eyeLookOutLeft": [0.2],
                "eyeLookOutRight": [0.7],
                "eyeLookInRight": [0.1],
                "eyeLookUpLeft": [0.9],
                "eyeLookDownLeft": [0.3],
                "eyeLookUpRight": [0.6],
                "eyeLookDownRight": [0.2],
            }
        )
        horizontal, vertical = derive_arkit_gaze(frame)
        self.assertAlmostEqual(horizontal[0], 0.6)
        self.assertAlmostEqual(vertical[0], 0.5)

    def test_audio_offset_recovers_embedded_signal(self):
        rng = np.random.default_rng(7)
        reference = rng.normal(size=3000)
        target = -reference[800:1800]
        result = audio_offset(reference, target, sample_rate=1000)
        self.assertAlmostEqual(result["offset_s"], 0.8, places=3)
        self.assertLess(result["correlation"], -0.99)

    def test_inverse_rotation_recovers_head_relative_vector(self):
        rotation = head_to_world_rotation(0.2, -0.3, 0.1)
        head_relative = np.array([0.1, -0.2, -0.97])
        world = rotation @ head_relative
        corrected = remove_head_rotation(
            world[None, :], np.array([[0.2, -0.3, 0.1]])
        )[0]
        np.testing.assert_allclose(corrected, head_relative, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
