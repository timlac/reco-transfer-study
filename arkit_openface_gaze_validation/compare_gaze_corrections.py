#!/usr/bin/env python3
"""Compare three simple OpenFace gaze corrections using audio synchronization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_gaze as shared


METHODS = ("uncorrected", "openface_rotation", "arkit_rotation")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=script_dir.parent / "data")
    parser.add_argument(
        "--manifest", type=Path, default=script_dir / "results" / "manifest.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "gaze_correction_results"
    )
    parser.add_argument(
        "--synchronization",
        choices=("audio_only", "actor_held_out_residual_lag"),
        default="audio_only",
    )
    parser.add_argument(
        "--cv-lags",
        type=Path,
        default=script_dir / "head_pose_results" / "cross_validated_lags.csv",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def left_handed_axis_rotations(
    pitch: float, yaw: float, roll: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OpenFace left-handed Rx, Ry, Rz matrices."""
    cx, sx = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cz, sz = np.cos(roll), np.sin(roll)
    rx = np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    ry = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    rz = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    return rx, ry, rz


def head_to_world_rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Construct the head-to-world matrix for OpenFace's pose convention."""
    rx, ry, rz = left_handed_axis_rotations(pitch, yaw, roll)
    # OpenFace reports left-handed world-referenced pose. Rx @ Ry @ Rz maps
    # world directions into the head frame, so its transpose is head-to-world.
    return (rx @ ry @ rz).T


def remove_head_rotation(
    gaze_vectors: np.ndarray, rotations: np.ndarray
) -> np.ndarray:
    """Apply the inverse head rotation to each world-coordinate gaze vector."""
    corrected = []
    for gaze, (pitch, yaw, roll) in zip(gaze_vectors, rotations):
        head_to_world = head_to_world_rotation(pitch, yaw, roll)
        corrected.append(head_to_world.T @ gaze)
    return np.asarray(corrected)


def vectors_to_arkit_aligned_angles(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert camera vectors to ARKit-aligned rightward/upward angles."""
    horizontal = -np.arctan2(vectors[:, 0], -vectors[:, 2])
    vertical = -np.arctan2(vectors[:, 1], -vectors[:, 2])
    return horizontal, vertical


def arkit_rotations_at_openface_frames(
    clip: shared.ClipData, target_time: np.ndarray
) -> np.ndarray:
    pitch = np.interp(target_time, clip.arkit_time_s, clip.arkit_head_pitch)
    yaw = np.interp(target_time, clip.arkit_time_s, clip.arkit_head_yaw)
    roll = np.interp(target_time, clip.arkit_time_s, clip.arkit_head_roll)
    rotations = np.column_stack([pitch, yaw, roll])

    if clip.embodiment == "furhat":
        means = np.array(
            [
                np.mean(clip.arkit_head_pitch),
                np.mean(clip.arkit_head_yaw),
                np.mean(clip.arkit_head_roll),
            ]
        )
        rotations = 0.5 * (rotations - means)

    # Axis directions established by the separate ARKit–OpenFace head-pose
    # validation. Furhat's NECK_TILT uses the opposite pitch direction.
    signs = np.array([1.0, -1.0, -1.0]) if clip.embodiment == "furhat" else -np.ones(3)
    return rotations * signs


def valid_openface_rows(frame: pd.DataFrame) -> pd.Series:
    required = [
        "timestamp",
        "gaze_0_x",
        "gaze_0_y",
        "gaze_0_z",
        "gaze_1_x",
        "gaze_1_y",
        "gaze_1_z",
        "pose_Rx",
        "pose_Ry",
        "pose_Rz",
    ]
    return (shared.column(frame, "success") == 1) & np.all(
        np.isfinite(frame[required]), axis=1
    )


def calculate_clip_metrics(
    data_root: Path,
    clips: list[shared.ClipData],
    synchronization: str,
    lag_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    rows: list[dict] = []
    for clip in clips:
        path = (
            data_root
            / "openface_files"
            / clip.embodiment
            / f"{clip.stimulus_id}.csv"
        )
        frame = shared.clean_columns(pd.read_csv(path))
        frame = frame.loc[valid_openface_rows(frame)].copy()
        residual_lag = (
            0.0
            if synchronization == "audio_only"
            else lag_lookup[(clip.embodiment, clip.actor)]
        )
        target_time = (
            frame["timestamp"].to_numpy(float)
            + clip.audio_offset_s
            + residual_lag
        )
        keep = (target_time >= clip.arkit_time_s[0]) & (
            target_time <= clip.arkit_time_s[-1]
        )
        frame = frame.iloc[np.flatnonzero(keep)]
        target_time = target_time[keep]

        gaze = np.column_stack(
            [
                (frame["gaze_0_x"] + frame["gaze_1_x"]) / 2,
                (frame["gaze_0_y"] + frame["gaze_1_y"]) / 2,
                (frame["gaze_0_z"] + frame["gaze_1_z"]) / 2,
            ]
        )
        openface_rotations = frame[["pose_Rx", "pose_Ry", "pose_Rz"]].to_numpy()
        arkit_rotations = arkit_rotations_at_openface_frames(clip, target_time)
        gaze_by_method = {
            "uncorrected": gaze,
            "openface_rotation": remove_head_rotation(gaze, openface_rotations),
            "arkit_rotation": remove_head_rotation(gaze, arkit_rotations),
        }
        source = {
            "x": np.interp(target_time, clip.arkit_time_s, clip.arkit_gaze_x),
            "y": np.interp(target_time, clip.arkit_time_s, clip.arkit_gaze_y),
        }

        for method in METHODS:
            horizontal, vertical = vectors_to_arkit_aligned_angles(
                gaze_by_method[method]
            )
            for axis, observed in (("x", horizontal), ("y", vertical)):
                rows.append(
                    {
                        "stimulus_id": clip.stimulus_id,
                        "actor": clip.actor,
                        "embodiment": clip.embodiment,
                        "method": method,
                        "axis": axis,
                        "synchronization": synchronization,
                        "residual_lag_s": residual_lag,
                        "n_overlapping_frames": len(observed),
                        "pearson_r": shared.safe_correlation(
                            source[axis], observed, "pearson"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize(metrics: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for embodiment in shared.EMBODIMENTS:
        for method in METHODS:
            for axis in ("x", "y"):
                subset = metrics.loc[
                    (metrics["embodiment"] == embodiment)
                    & (metrics["method"] == method)
                    & (metrics["axis"] == axis)
                ].dropna(subset=["pearson_r"])
                estimate = float(subset["pearson_r"].median())
                low, high = shared.cluster_bootstrap(
                    subset,
                    lambda sample: float(sample["pearson_r"].median()),
                    repetitions,
                    rng,
                )
                rows.append(
                    {
                        "embodiment": embodiment,
                        "method": method,
                        "axis": axis,
                        "median_pearson_r": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n_clips": len(subset),
                    }
                )
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path, summary: pd.DataFrame, synchronization: str
) -> None:
    labels = {
        "uncorrected": "Uncorrected world gaze",
        "openface_rotation": "OpenFace rotation matrix",
        "arkit_rotation": "ARKit rotation matrix",
    }
    lines = [
        "| Embodiment | Correction | Horizontal r | Vertical r |",
        "|---|---|---:|---:|",
    ]
    for embodiment in shared.EMBODIMENTS:
        for method in METHODS:
            values = {}
            for axis in ("x", "y"):
                row = summary.loc[
                    (summary["embodiment"] == embodiment)
                    & (summary["method"] == method)
                    & (summary["axis"] == axis)
                ].iloc[0]
                values[axis] = (
                    f"{row.median_pearson_r:.3f} "
                    f"[{row.ci_low:.3f}, {row.ci_high:.3f}]"
                )
            lines.append(
                f"| {shared.DISPLAY_NAMES[embodiment]} | {labels[method]} | "
                f"{values['x']} | {values['y']} |"
            )

    synchronization_text = (
        "audio-fade synchronization only"
        if synchronization == "audio_only"
        else "audio-fade synchronization plus actor-held-out residual visual lag"
    )
    final_note = (
        "No residual lag optimization is used."
        if synchronization == "audio_only"
        else "For each actor, the residual lag was estimated using only the other actors."
    )
    content = f"""# Gaze correction comparison

All results use {synchronization_text}. Values are medians of 64
within-clip Pearson correlations with actor-cluster bootstrap 95% intervals.

{chr(10).join(lines)}

The OpenFace method averages the two 3D world-coordinate gaze vectors, applies
the inverse of the full `Rx @ Ry @ Rz` head rotation, and converts the resulting
head-relative vector back to horizontal and vertical angles. The ARKit method
uses synchronized ARKit pitch/yaw/roll instead; Furhat rotations are mean-centred
and multiplied by 0.5 as documented. Coordinate directions are harmonized using
the separate head-pose validation.

{final_note}
"""
    (output_dir / "SUMMARY.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest.resolve())
    clips = shared.load_clip_data(data_root, manifest)
    lag_lookup: dict[tuple[str, str], float] = {}
    if args.synchronization == "actor_held_out_residual_lag":
        lag_frame = pd.read_csv(args.cv_lags.resolve())
        lag_lookup = {
            (row.embodiment, row.actor): float(row.selected_residual_lag_s)
            for row in lag_frame.itertuples(index=False)
        }
    metrics = calculate_clip_metrics(
        data_root, clips, args.synchronization, lag_lookup
    )
    summary = summarize(metrics, args.bootstrap, args.seed)
    metrics.to_csv(output_dir / "clip_level_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    write_summary(output_dir, summary, args.synchronization)
    print(f"Wrote gaze-correction comparison to {output_dir}")


if __name__ == "__main__":
    main()
