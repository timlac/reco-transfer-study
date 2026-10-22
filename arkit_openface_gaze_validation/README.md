# ARKit–OpenFace gaze validation

This analysis synchronizes the 64 selected ARKit recordings with the corresponding Human, MetaHuman, and Furhat videos and tests whether OpenFace gaze follows the original ARKit `EyeLook` trajectories.

## Method

1. Pair files by stimulus identifier.
2. Infer the ARKit sampling clock from the LiveLinkFace `Timecode` column.
3. Locate each video inside the paired ARKit audio using normalized waveform correlation.
4. Estimate one small residual visual lag per embodiment from head-pose correspondence.
5. Resample ARKit onto the successful OpenFace frame timestamps.
6. Construct binocular ARKit horizontal and vertical gaze drivers from the eight `EyeLook...` coefficients.
7. Compare these with head-relative OpenFace gaze trajectories.

The ARKit drivers are defined as:

- horizontal (rightward positive): `((InLeft − OutLeft) + (OutRight − InRight)) / 2`
- vertical (upward positive): `((UpLeft − DownLeft) + (UpRight − DownRight)) / 2`

OpenFace documents `gaze_angle_x` and `gaze_angle_y` as gaze in camera/world coordinates. ARKit `EyeLook` values are eye-in-head controls. The primary comparison therefore removes OpenFace head yaw/pitch using a first-order angular correction. The raw world-coordinate comparison is retained as a sensitivity analysis.

## Run

```bash
../.venv/bin/python analyze_gaze.py
../.venv/bin/python -m unittest test_analysis.py
```

The script requires `ffmpeg` plus the Python packages listed in `requirements.txt`.

## Outputs

- `results/manifest.csv`: pairing, sampling, synchronization, and OpenFace quality information.
- `results/visual_alignment_lags.csv`: residual visual timing correction estimated from head pose.
- `results/clip_level_metrics.csv`: per-clip trajectory and dynamic-range results.
- `results/summary_metrics.csv`: actor-cluster bootstrap summaries.
- `results/data_quality_summary.csv`: OpenFace detection and audio-alignment summary.
- `results/gaze_validation.svg` and `.png`: diagnostic figure.
- `results/SUMMARY.md`: concise numerical interpretation.

## Head-pose validation

Run `../.venv/bin/python analyze_head_pose.py` to compare synchronized ARKit
head rotations with OpenFace pitch, yaw, and roll. Results are written separately
to `head_pose_results/`; the gaze results are not changed.

Run `../.venv/bin/python compare_gaze_corrections.py` for the audio-synchronized
comparison of uncorrected gaze, full OpenFace rotation-matrix correction, and
ARKit rotation-matrix correction. Results are written to
`gaze_correction_results/`.

OpenFace output definitions: <https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format>
