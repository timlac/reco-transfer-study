#!/usr/bin/env python3
"""Validate OpenFace head rotations against synchronized ARKit rotations.

This is intentionally separate from the gaze analysis. It compares matching
rotation axes within each clip and reports audio-only synchronization plus a
leave-one-actor-out residual-lag sensitivity analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_gaze as shared


AXES = {
    "pitch": ("arkit_head_pitch", "openface_pose_pitch"),
    "yaw": ("arkit_head_yaw", "openface_pose_yaw"),
    "roll": ("arkit_head_roll", "openface_pose_roll"),
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=script_dir.parent / "data")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=script_dir / "results" / "manifest.csv",
        help="Manifest produced by analyze_gaze.py; contains audio offsets.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=script_dir / "head_pose_results"
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def expected_arkit_rotation(clip: shared.ClipData, axis: str) -> np.ndarray:
    """Return the rotation sent to the embodiment, per the supplement."""
    source_name, _ = AXES[axis]
    values = np.asarray(getattr(clip, source_name), dtype=float)
    if clip.embodiment == "furhat":
        return 0.5 * (values - np.mean(values))
    return values


def score_grid(clips: list[shared.ClipData], grid: np.ndarray) -> pd.DataFrame:
    """Precompute per-clip alignment scores for cross-validated lag selection."""
    rows: list[dict] = []
    for clip in clips:
        for lag_s in grid:
            rows.append(
                {
                    "stimulus_id": clip.stimulus_id,
                    "actor": clip.actor,
                    "embodiment": clip.embodiment,
                    "lag_s": float(lag_s),
                    "alignment_score": shared.head_alignment_score(clip, float(lag_s)),
                }
            )
    return pd.DataFrame(rows)


def cross_validated_lags(
    clips: list[shared.ClipData], grid: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose each actor's lag using all other actors in the same embodiment."""
    scores = score_grid(clips, grid)
    lag_rows: list[dict] = []
    for embodiment in shared.EMBODIMENTS:
        subset = scores.loc[scores["embodiment"] == embodiment]
        for actor in sorted(subset["actor"].unique()):
            training = subset.loc[subset["actor"] != actor]
            lag_scores = training.groupby("lag_s")["alignment_score"].median()
            best_lag = float(lag_scores.idxmax())
            lag_rows.append(
                {
                    "embodiment": embodiment,
                    "actor": actor,
                    "selected_residual_lag_s": best_lag,
                    "training_median_alignment_abs_r": float(lag_scores.loc[best_lag]),
                }
            )
    return pd.DataFrame(lag_rows), scores


def calculate_clip_metrics(
    clips: list[shared.ClipData], cv_lags: pd.DataFrame
) -> pd.DataFrame:
    lag_lookup = {
        (row.embodiment, row.actor): float(row.selected_residual_lag_s)
        for row in cv_lags.itertuples(index=False)
    }
    rows: list[dict] = []
    for synchronization in ("audio_only", "actor_held_out_residual_lag"):
        for clip in clips:
            residual_lag = (
                0.0
                if synchronization == "audio_only"
                else lag_lookup[(clip.embodiment, clip.actor)]
            )
            target_time, keep = shared.valid_overlap(clip, residual_lag)
            for axis, (_, observed_name) in AXES.items():
                source = np.interp(
                    target_time,
                    clip.arkit_time_s,
                    expected_arkit_rotation(clip, axis),
                )
                observed = np.asarray(getattr(clip, observed_name), dtype=float)[keep]
                pearson_r = shared.safe_correlation(source, observed, "pearson")
                rows.append(
                    {
                        "stimulus_id": clip.stimulus_id,
                        "actor": clip.actor,
                        "embodiment": clip.embodiment,
                        "axis": axis,
                        "synchronization": synchronization,
                        "audio_offset_s": clip.audio_offset_s,
                        "residual_lag_s": residual_lag,
                        "n_overlapping_frames": len(source),
                        "pearson_r": pearson_r,
                        "abs_pearson_r": abs(pearson_r),
                        "spearman_rho": shared.safe_correlation(
                            source, observed, "spearman"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize(
    metrics: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for synchronization in metrics["synchronization"].unique():
        for embodiment in shared.EMBODIMENTS:
            for axis in AXES:
                subset = metrics.loc[
                    (metrics["synchronization"] == synchronization)
                    & (metrics["embodiment"] == embodiment)
                    & (metrics["axis"] == axis)
                ].dropna(subset=["pearson_r"])
                for metric in ("pearson_r", "abs_pearson_r"):
                    estimate = float(subset[metric].median())
                    low, high = shared.cluster_bootstrap(
                        subset,
                        lambda sample, item=metric: float(sample[item].median()),
                        repetitions,
                        rng,
                    )
                    rows.append(
                        {
                            "synchronization": synchronization,
                            "embodiment": embodiment,
                            "axis": axis,
                            "metric": f"median_{metric}",
                            "estimate": estimate,
                            "ci_low": low,
                            "ci_high": high,
                            "n_clips": len(subset),
                        }
                    )
                rows.append(
                    {
                        "synchronization": synchronization,
                        "embodiment": embodiment,
                        "axis": axis,
                        "metric": "proportion_positive_r",
                        "estimate": float((subset["pearson_r"] > 0).mean()),
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "n_clips": len(subset),
                    }
                )
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path, summary: pd.DataFrame, cv_lags: pd.DataFrame
) -> None:
    def table(sync: str, absolute: bool) -> str:
        metric = "median_abs_pearson_r" if absolute else "median_pearson_r"
        header = "| Embodiment | Pitch | Yaw | Roll |\n|---|---:|---:|---:|"
        body = []
        for embodiment in shared.EMBODIMENTS:
            values = []
            for axis in AXES:
                row = summary.loc[
                    (summary["synchronization"] == sync)
                    & (summary["embodiment"] == embodiment)
                    & (summary["axis"] == axis)
                    & (summary["metric"] == metric)
                ].iloc[0]
                values.append(
                    f"{row.estimate:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]"
                )
            body.append(
                f"| {shared.DISPLAY_NAMES[embodiment]} | " + " | ".join(values) + " |"
            )
        return header + "\n" + "\n".join(body)

    lag_description = []
    for embodiment in shared.EMBODIMENTS:
        values = cv_lags.loc[
            cv_lags["embodiment"] == embodiment, "selected_residual_lag_s"
        ]
        lag_description.append(
            f"{shared.DISPLAY_NAMES[embodiment]} median {values.median():.2f} s "
            f"(range {values.min():.2f} to {values.max():.2f})"
        )

    content = f"""# ARKit–OpenFace head-pose validation

Each value is the median of 64 within-clip Pearson correlations, with a 95%
actor-cluster bootstrap interval. ARKit pitch/yaw/roll were matched to OpenFace
`pose_Rx`/`pose_Ry`/`pose_Rz`. Furhat's documented mean-centering and ×0.5
rotation transform was applied; it does not change correlation magnitude.

## Audio-only synchronization (independent primary check)

Absolute correlations ignore coordinate-direction differences between ARKit and
OpenFace and measure trajectory correspondence.

{table("audio_only", True)}

Signed correlations expose coordinate-direction differences.

{table("audio_only", False)}

## Actor-held-out residual-lag sensitivity analysis

For each actor, the residual lag was selected using only the other actors:
{'; '.join(lag_description)}.

{table("actor_held_out_residual_lag", True)}

## Interpretation boundary

The audio-only result is the independent validation because its timing comes
from the shared audio rather than head-pose agreement. The held-out result shows
what happens after estimating a common residual visual delay on other actors.
Absolute correlation assesses whether trajectory shape and timing are recovered;
the signed table must also be consulted because an axis can track strongly while
using the opposite coordinate direction. Correlation alone does not establish
absolute angular accuracy.
"""
    (output_dir / "SUMMARY.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest.resolve())
    clips = shared.load_clip_data(args.data_root.resolve(), manifest)
    grid = np.arange(-0.50, 0.5001, 0.02)
    cv_lags, lag_scores = cross_validated_lags(clips, grid)
    metrics = calculate_clip_metrics(clips, cv_lags)
    summary = summarize(metrics, args.bootstrap, args.seed)

    cv_lags.to_csv(output_dir / "cross_validated_lags.csv", index=False)
    lag_scores.to_csv(output_dir / "lag_alignment_scores.csv", index=False)
    metrics.to_csv(output_dir / "clip_level_head_pose_metrics.csv", index=False)
    summary.to_csv(output_dir / "head_pose_summary_metrics.csv", index=False)
    write_summary(output_dir, summary, cv_lags)
    print(f"Wrote head-pose validation results to {output_dir}")


if __name__ == "__main__":
    main()
