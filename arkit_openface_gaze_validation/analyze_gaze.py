#!/usr/bin/env python3
"""Synchronize ARKit and OpenFace streams and validate OpenFace gaze.

The ARKit eye-look coefficients are eye-in-head controls. OpenFace gaze angles are
reported in camera/world coordinates. The primary analysis therefore removes the
OpenFace head-pose contribution (first-order angular correction) before comparing
the two streams. Raw OpenFace world-coordinate gaze is retained as a sensitivity
analysis.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr


EMBODIMENTS = ("original", "metahuman", "furhat")
DISPLAY_NAMES = {
    "original": "Human",
    "metahuman": "MetaHuman",
    "furhat": "Furhat",
}
COLORS = {
    "original": "#333333",
    "metahuman": "#2A76DD",
    "furhat": "#E37A24",
}


@dataclass
class ClipData:
    stimulus_id: str
    actor: str
    embodiment: str
    audio_offset_s: float
    arkit_time_s: np.ndarray
    arkit_gaze_x: np.ndarray
    arkit_gaze_y: np.ndarray
    arkit_head_pitch: np.ndarray
    arkit_head_yaw: np.ndarray
    arkit_head_roll: np.ndarray
    openface_time_s: np.ndarray
    openface_gaze_world_x: np.ndarray
    openface_gaze_world_y: np.ndarray
    openface_pose_pitch: np.ndarray
    openface_pose_yaw: np.ndarray
    openface_pose_roll: np.ndarray


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir.parent / "data",
        help="Transfer-study data directory (default: repository data/).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "results",
        help="Directory for generated tables and figures.",
    )
    parser.add_argument(
        "--audio-rate",
        type=int,
        default=2000,
        help="Audio sampling rate used for synchronization.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Actor-cluster bootstrap repetitions.",
    )
    parser.add_argument("--seed", type=int, default=20261019)
    return parser.parse_args()


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [column.strip() for column in frame.columns]
    return frame


def column(frame: pd.DataFrame, name: str) -> pd.Series:
    matches = [item for item in frame.columns if item.lower() == name.lower()]
    if len(matches) != 1:
        raise KeyError(f"Expected one column named {name!r}; found {matches}")
    return frame[matches[0]]


def timecode_seconds(value: str) -> float:
    """Convert LiveLinkFace HH:MM:SS:frame.subframe timecode at a 60-Hz base."""
    hours, minutes, seconds, frame = str(value).split(":")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + float(frame) / 60.0
    )


def relative_timecodes(values: pd.Series) -> np.ndarray:
    absolute = np.array([timecode_seconds(value) for value in values], dtype=float)
    # No selected clip crosses midnight, but support it to keep the parser safe.
    for index in range(1, len(absolute)):
        if absolute[index] < absolute[index - 1] - 3600:
            absolute[index:] += 24 * 3600
    return absolute - absolute[0]


def robust_range(values: np.ndarray) -> float:
    return float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))


def decode_audio(path: Path, sample_rate: int) -> np.ndarray:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True, check=True)
    audio = np.frombuffer(completed.stdout, dtype="<f4").astype(float)
    if audio.size == 0:
        raise ValueError(f"No audio decoded from {path}")
    return audio - np.mean(audio)


def audio_offset(reference: np.ndarray, target: np.ndarray, sample_rate: int) -> dict:
    """Locate target audio inside reference audio using normalized correlation."""
    if len(reference) < len(target):
        raise ValueError(
            f"Reference audio ({len(reference)}) is shorter than target ({len(target)})"
        )
    numerator = correlate(reference, target, mode="valid", method="fft")
    cumulative_energy = np.r_[0.0, np.cumsum(reference * reference)]
    local_energy = (
        cumulative_energy[len(target) :] - cumulative_energy[: -len(target)]
    )
    denominator = np.sqrt(np.maximum(local_energy, 1e-20) * np.sum(target * target))
    normalized = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    peak_index = int(np.argmax(np.abs(normalized)))
    exclusion = max(1, int(round(0.25 * sample_rate)))
    competitors = np.abs(normalized).copy()
    competitors[
        max(0, peak_index - exclusion) : min(len(competitors), peak_index + exclusion + 1)
    ] = -np.inf
    second_peak = float(np.max(competitors)) if np.isfinite(competitors).any() else np.nan
    return {
        "offset_s": peak_index / sample_rate,
        "correlation": float(normalized[peak_index]),
        "peak_margin": float(abs(normalized[peak_index]) - second_peak),
    }


def stimulus_parts(stimulus_id: str) -> tuple[str, str]:
    actor, clip = stimulus_id.split("_", 1)
    return actor, clip


def arkit_paths(data_root: Path, stimulus_id: str) -> tuple[Path, Path]:
    actor, clip = stimulus_parts(stimulus_id)
    csv_path = data_root / "arkit" / "cuts-n" / actor / f"{clip}.face.csv"
    wav_path = csv_path.with_name(csv_path.name.replace(".face.csv", ".face_fade.wav"))
    return csv_path, wav_path


def selected_stimuli(data_root: Path) -> list[str]:
    original = data_root / "openface_files" / "original"
    return sorted(path.stem for path in original.glob("*.csv"))


def require_inputs(data_root: Path, stimuli: list[str]) -> None:
    missing: list[Path] = []
    for stimulus_id in stimuli:
        arkit_csv, arkit_wav = arkit_paths(data_root, stimulus_id)
        for path in (arkit_csv, arkit_wav):
            if not path.exists():
                missing.append(path)
        for embodiment in EMBODIMENTS:
            for path in (
                data_root / "openface_files" / embodiment / f"{stimulus_id}.csv",
                data_root / "videos" / embodiment / f"{stimulus_id}.mov",
            ):
                if not path.exists():
                    missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:20])
        raise FileNotFoundError(f"Missing {len(missing)} required files:\n{preview}")


def derive_arkit_gaze(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Combine the eight EyeLook controls into rightward/upward binocular drivers."""
    horizontal = (
        (column(frame, "EyeLookInLeft") - column(frame, "EyeLookOutLeft"))
        + (column(frame, "EyeLookOutRight") - column(frame, "EyeLookInRight"))
    ) / 2
    vertical = (
        (column(frame, "EyeLookUpLeft") - column(frame, "EyeLookDownLeft"))
        + (column(frame, "EyeLookUpRight") - column(frame, "EyeLookDownRight"))
    ) / 2
    return horizontal.to_numpy(float), vertical.to_numpy(float)


def estimate_manifest(
    data_root: Path, stimuli: list[str], audio_rate: int
) -> pd.DataFrame:
    rows: list[dict] = []
    for stimulus_id in stimuli:
        actor, _ = stimulus_parts(stimulus_id)
        arkit_csv, arkit_wav = arkit_paths(data_root, stimulus_id)
        arkit = clean_columns(pd.read_csv(arkit_csv))
        arkit_time = relative_timecodes(column(arkit, "Timecode"))
        sample_interval = float(np.median(np.diff(arkit_time)))
        reference_audio = decode_audio(arkit_wav, audio_rate)
        for embodiment in EMBODIMENTS:
            video = data_root / "videos" / embodiment / f"{stimulus_id}.mov"
            openface_path = (
                data_root / "openface_files" / embodiment / f"{stimulus_id}.csv"
            )
            sync = audio_offset(reference_audio, decode_audio(video, audio_rate), audio_rate)
            openface = clean_columns(pd.read_csv(openface_path))
            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "actor": actor,
                    "embodiment": embodiment,
                    "arkit_frames": len(arkit),
                    "arkit_duration_s": float(arkit_time[-1] + sample_interval),
                    "arkit_sampling_hz": 1.0 / sample_interval,
                    "openface_frames": len(openface),
                    "openface_duration_s": float(column(openface, "timestamp").max()),
                    "openface_success_rate": float(column(openface, "success").mean()),
                    "openface_mean_confidence": float(column(openface, "confidence").mean()),
                    "audio_offset_s": sync["offset_s"],
                    "audio_correlation": sync["correlation"],
                    "audio_abs_correlation": abs(sync["correlation"]),
                    "audio_peak_margin": sync["peak_margin"],
                }
            )
    return pd.DataFrame(rows)


def load_clip_data(data_root: Path, manifest: pd.DataFrame) -> list[ClipData]:
    clips: list[ClipData] = []
    arkit_cache: dict[str, dict] = {}
    for stimulus_id in sorted(manifest["stimulus_id"].unique()):
        actor, _ = stimulus_parts(stimulus_id)
        arkit_csv, _ = arkit_paths(data_root, stimulus_id)
        arkit = clean_columns(pd.read_csv(arkit_csv))
        gaze_x, gaze_y = derive_arkit_gaze(arkit)
        arkit_cache[stimulus_id] = {
            "actor": actor,
            "time": relative_timecodes(column(arkit, "Timecode")),
            "gaze_x": gaze_x,
            "gaze_y": gaze_y,
            "pitch": column(arkit, "HeadPitch").to_numpy(float),
            "yaw": column(arkit, "HeadYaw").to_numpy(float),
            "roll": column(arkit, "HeadRoll").to_numpy(float),
        }

    for row in manifest.itertuples(index=False):
        source = arkit_cache[row.stimulus_id]
        openface_path = (
            data_root / "openface_files" / row.embodiment / f"{row.stimulus_id}.csv"
        )
        openface = clean_columns(pd.read_csv(openface_path))
        valid = (
            (column(openface, "success") == 1)
            & np.isfinite(column(openface, "timestamp"))
            & np.isfinite(column(openface, "gaze_angle_x"))
            & np.isfinite(column(openface, "gaze_angle_y"))
        )
        clips.append(
            ClipData(
                stimulus_id=row.stimulus_id,
                actor=source["actor"],
                embodiment=row.embodiment,
                audio_offset_s=float(row.audio_offset_s),
                arkit_time_s=source["time"],
                arkit_gaze_x=source["gaze_x"],
                arkit_gaze_y=source["gaze_y"],
                arkit_head_pitch=source["pitch"],
                arkit_head_yaw=source["yaw"],
                arkit_head_roll=source["roll"],
                openface_time_s=column(openface, "timestamp")[valid].to_numpy(float),
                openface_gaze_world_x=column(openface, "gaze_angle_x")[valid].to_numpy(float),
                openface_gaze_world_y=column(openface, "gaze_angle_y")[valid].to_numpy(float),
                openface_pose_pitch=column(openface, "pose_Rx")[valid].to_numpy(float),
                openface_pose_yaw=column(openface, "pose_Ry")[valid].to_numpy(float),
                openface_pose_roll=column(openface, "pose_Rz")[valid].to_numpy(float),
            )
        )
    return clips


def valid_overlap(clip: ClipData, residual_lag_s: float) -> tuple[np.ndarray, np.ndarray]:
    target_time = clip.openface_time_s + clip.audio_offset_s + residual_lag_s
    keep = (target_time >= clip.arkit_time_s[0]) & (
        target_time <= clip.arkit_time_s[-1]
    )
    return target_time[keep], keep


def safe_correlation(left: np.ndarray, right: np.ndarray, method: str) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 20 or np.std(left) <= 1e-6 or np.std(right) <= 1e-6:
        return np.nan
    if method == "pearson":
        return float(pearsonr(left, right).statistic)
    return float(spearmanr(left, right).statistic)


def head_alignment_score(clip: ClipData, lag_s: float) -> float:
    target_time, keep = valid_overlap(clip, lag_s)
    pairs = (
        (clip.arkit_head_pitch, clip.openface_pose_pitch[keep]),
        (clip.arkit_head_yaw, clip.openface_pose_yaw[keep]),
        (clip.arkit_head_roll, clip.openface_pose_roll[keep]),
    )
    correlations = []
    for source, observed in pairs:
        interpolated = np.interp(target_time, clip.arkit_time_s, source)
        result = safe_correlation(interpolated, observed, "pearson")
        if np.isfinite(result):
            correlations.append(abs(result))
    return float(np.median(correlations)) if correlations else np.nan


def estimate_visual_lags(clips: list[ClipData]) -> pd.DataFrame:
    grid = np.arange(-0.50, 0.5001, 0.02)
    rows = []
    for embodiment in EMBODIMENTS:
        subset = [clip for clip in clips if clip.embodiment == embodiment]
        scores = []
        for lag_s in grid:
            per_clip = np.array(
                [head_alignment_score(clip, float(lag_s)) for clip in subset]
            )
            scores.append(float(np.nanmedian(per_clip)))
        best_index = int(np.nanargmax(scores))
        rows.append(
            {
                "embodiment": embodiment,
                "visual_residual_lag_s": float(grid[best_index]),
                "median_head_alignment_abs_r": scores[best_index],
            }
        )
    return pd.DataFrame(rows)


def calculate_clip_metrics(
    clips: list[ClipData], visual_lags: pd.DataFrame
) -> pd.DataFrame:
    lag_lookup = dict(
        zip(visual_lags["embodiment"], visual_lags["visual_residual_lag_s"])
    )
    rows: list[dict] = []
    for clip in clips:
        lag_s = float(lag_lookup[clip.embodiment])
        target_time, keep = valid_overlap(clip, lag_s)
        source_by_axis = {
            "x": np.interp(target_time, clip.arkit_time_s, clip.arkit_gaze_x),
            "y": np.interp(target_time, clip.arkit_time_s, clip.arkit_gaze_y),
        }
        # OpenFace angles are world/camera referenced. Subtracting head yaw/pitch
        # gives a first-order eye-in-head angle, which matches ARKit EyeLook.
        observed = {
            ("world", "x"): clip.openface_gaze_world_x[keep],
            ("world", "y"): -clip.openface_gaze_world_y[keep],
            ("head_relative", "x"): (
                clip.openface_gaze_world_x[keep] - clip.openface_pose_yaw[keep]
            ),
            ("head_relative", "y"): -(
                clip.openface_gaze_world_y[keep] - clip.openface_pose_pitch[keep]
            ),
        }
        for reference_frame in ("head_relative", "world"):
            for axis in ("x", "y"):
                source = source_by_axis[axis]
                output = observed[(reference_frame, axis)]
                rows.append(
                    {
                        "stimulus_id": clip.stimulus_id,
                        "actor": clip.actor,
                        "embodiment": clip.embodiment,
                        "reference_frame": reference_frame,
                        "axis": axis,
                        "audio_offset_s": clip.audio_offset_s,
                        "visual_residual_lag_s": lag_s,
                        "n_overlapping_frames": len(source),
                        "pearson_r": safe_correlation(source, output, "pearson"),
                        "spearman_rho": safe_correlation(source, output, "spearman"),
                        "source_mean": float(np.mean(source)),
                        "openface_mean": float(np.mean(output)),
                        "source_sd": float(np.std(source, ddof=1)),
                        "openface_sd": float(np.std(output, ddof=1)),
                        "source_robust_range": robust_range(source),
                        "openface_robust_range": robust_range(output),
                    }
                )
    return pd.DataFrame(rows)


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    actor_column = frame["actor"].to_numpy()
    actors = pd.unique(actor_column)
    actor_rows = [np.flatnonzero(actor_column == actor) for actor in actors]
    estimates = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(actors), size=len(actors))
        row_indices = np.concatenate([actor_rows[index] for index in sampled])
        replicate = frame.iloc[row_indices]
        value = statistic(replicate)
        if np.isfinite(value):
            estimates.append(value)
    if not estimates:
        return np.nan, np.nan
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def bivariate_statistic(
    frame: pd.DataFrame, left: str, right: str, method: str
) -> float:
    valid = frame[[left, right]].dropna()
    if len(valid) < 3 or valid[left].nunique() < 2 or valid[right].nunique() < 2:
        return np.nan
    if method == "pearson":
        return float(np.corrcoef(valid[left].to_numpy(), valid[right].to_numpy())[0, 1])
    return float(spearmanr(valid[left], valid[right]).statistic)


def summarize_metrics(
    clip_metrics: pd.DataFrame, repetitions: int, seed: int
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    primary = clip_metrics.loc[clip_metrics["reference_frame"] == "head_relative"]
    for axis in ("x", "y"):
        for embodiment in EMBODIMENTS:
            subset = primary.loc[
                (primary["axis"] == axis)
                & (primary["embodiment"] == embodiment)
            ].copy()
            for metric in ("pearson_r", "spearman_rho"):
                valid = subset.loc[np.isfinite(subset[metric])]
                estimate = float(valid[metric].median())
                low, high = cluster_bootstrap(
                    valid,
                    lambda sample, item=metric: float(sample[item].median()),
                    repetitions,
                    rng,
                )
                rows.append(
                    {
                        "analysis": "median_within_clip",
                        "axis": axis,
                        "embodiment": embodiment,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n_clips": len(valid),
                    }
                )
            comparisons = (
                ("clip_mean_pearson", "source_mean", "openface_mean", "pearson"),
                ("clip_mean_spearman", "source_mean", "openface_mean", "spearman"),
                (
                    "clip_range_pearson",
                    "source_robust_range",
                    "openface_robust_range",
                    "pearson",
                ),
                ("clip_sd_pearson", "source_sd", "openface_sd", "pearson"),
            )
            for metric, left, right, method in comparisons:
                estimate = bivariate_statistic(subset, left, right, method)
                low, high = cluster_bootstrap(
                    subset,
                    lambda sample, l=left, r=right, m=method: bivariate_statistic(
                        sample, l, r, m
                    ),
                    repetitions,
                    rng,
                )
                rows.append(
                    {
                        "analysis": "across_clips",
                        "axis": axis,
                        "embodiment": embodiment,
                        "metric": metric,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "n_clips": len(subset),
                    }
                )

    # Same-clip OpenFace dynamic-range ratio relative to the Human rendering.
    ranges = primary.pivot_table(
        index=["stimulus_id", "actor", "axis"],
        columns="embodiment",
        values="openface_robust_range",
    ).reset_index()
    for axis in ("x", "y"):
        for embodiment in ("metahuman", "furhat"):
            subset = ranges.loc[ranges["axis"] == axis].copy()
            subset = subset.loc[subset["original"] > 1e-6]
            subset["ratio"] = subset[embodiment] / subset["original"]
            estimate = float(subset["ratio"].median())
            low, high = cluster_bootstrap(
                subset,
                lambda sample: float(sample["ratio"].median()),
                repetitions,
                rng,
            )
            rows.append(
                {
                    "analysis": "range_retention_vs_human",
                    "axis": axis,
                    "embodiment": embodiment,
                    "metric": "median_openface_range_ratio",
                    "estimate": estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "n_clips": len(subset),
                }
            )
    return pd.DataFrame(rows)


def summarize_data_quality(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for embodiment in EMBODIMENTS:
        subset = manifest.loc[manifest["embodiment"] == embodiment]
        rows.append(
            {
                "embodiment": embodiment,
                "n_clips": len(subset),
                "median_openface_success_rate": subset[
                    "openface_success_rate"
                ].median(),
                "median_openface_confidence": subset[
                    "openface_mean_confidence"
                ].median(),
                "median_audio_offset_s": subset["audio_offset_s"].median(),
                "min_audio_offset_s": subset["audio_offset_s"].min(),
                "max_audio_offset_s": subset["audio_offset_s"].max(),
                "median_audio_abs_correlation": subset[
                    "audio_abs_correlation"
                ].median(),
            }
        )
    return pd.DataFrame(rows)


def plot_results(clip_metrics: pd.DataFrame, output_dir: Path) -> None:
    primary = clip_metrics.loc[clip_metrics["reference_frame"] == "head_relative"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    positions = np.arange(len(EMBODIMENTS)) + 1
    for axis_name, panel in zip(("x", "y"), axes[0]):
        values = [
            primary.loc[
                (primary["axis"] == axis_name)
                & (primary["embodiment"] == embodiment),
                "pearson_r",
            ].dropna()
            for embodiment in EMBODIMENTS
        ]
        box = panel.boxplot(values, positions=positions, patch_artist=True, widths=0.6)
        for patch, embodiment in zip(box["boxes"], EMBODIMENTS):
            patch.set_facecolor(COLORS[embodiment])
            patch.set_alpha(0.65)
        panel.axhline(0, color="#777777", linewidth=0.8)
        panel.set_xticks(positions, [DISPLAY_NAMES[item] for item in EMBODIMENTS])
        panel.set_ylabel("Within-clip Pearson r")
        panel.set_title(f"{axis_name.upper()} gaze trajectory")

    for axis_name, panel in zip(("x", "y"), axes[1]):
        for embodiment in EMBODIMENTS:
            subset = primary.loc[
                (primary["axis"] == axis_name)
                & (primary["embodiment"] == embodiment)
            ]
            panel.scatter(
                subset["source_mean"],
                subset["openface_mean"],
                s=23,
                alpha=0.7,
                color=COLORS[embodiment],
                label=DISPLAY_NAMES[embodiment],
            )
        panel.set_xlabel("Mean ARKit EyeLook driver")
        panel.set_ylabel("Mean head-relative OpenFace gaze (rad)")
        panel.set_title(f"{axis_name.upper()} gaze across clips")
    axes[1, 1].legend(frameon=False)
    fig.suptitle("ARKit–OpenFace gaze correspondence", fontsize=14)
    fig.savefig(output_dir / "gaze_validation.png", dpi=220)
    fig.savefig(output_dir / "gaze_validation.svg")
    plt.close(fig)


def metric_value(
    summary: pd.DataFrame, axis: str, embodiment: str, metric: str
) -> float:
    row = summary.loc[
        (summary["axis"] == axis)
        & (summary["embodiment"] == embodiment)
        & (summary["metric"] == metric)
    ]
    return float(row.iloc[0]["estimate"])


def write_summary(
    output_dir: Path,
    manifest: pd.DataFrame,
    visual_lags: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    sample_rates = (
        manifest.drop_duplicates("stimulus_id")["arkit_sampling_hz"].round().astype(int)
    )
    counts = sample_rates.value_counts().sort_index()
    rate_text = ", ".join(f"{count} at {rate} Hz" for rate, count in counts.items())
    within_rows = []
    mean_rows = []
    range_rows = []
    for embodiment in EMBODIMENTS:
        within_rows.append(
            "| {name} | {x:.3f} | {y:.3f} |".format(
                name=DISPLAY_NAMES[embodiment],
                x=metric_value(summary, "x", embodiment, "pearson_r"),
                y=metric_value(summary, "y", embodiment, "pearson_r"),
            )
        )
        mean_rows.append(
            "| {name} | {x:.3f} | {y:.3f} |".format(
                name=DISPLAY_NAMES[embodiment],
                x=metric_value(summary, "x", embodiment, "clip_mean_pearson"),
                y=metric_value(summary, "y", embodiment, "clip_mean_pearson"),
            )
        )
        range_rows.append(
            "| {name} | {x:.3f} | {y:.3f} |".format(
                name=DISPLAY_NAMES[embodiment],
                x=metric_value(summary, "x", embodiment, "clip_range_pearson"),
                y=metric_value(summary, "y", embodiment, "clip_range_pearson"),
            )
        )
    lag_text = ", ".join(
        f"{DISPLAY_NAMES[row.embodiment]} {row.visual_residual_lag_s:.2f} s"
        for row in visual_lags.itertuples(index=False)
    )
    content = f"""# ARKit–OpenFace gaze-validation results

## Data and synchronization

- All 64 selected stimuli were paired across ARKit, Human, MetaHuman, and Furhat.
- ARKit sampling rates inferred from LiveLinkFace timecodes: {rate_text}.
- Video start offsets were estimated from the shared audio track.
- A single residual visual lag per embodiment was then estimated from head-pose correspondence: {lag_text}.
- OpenFace successful-frame rates and confidence are reported in `data_quality_summary.csv`.

## Primary results

ARKit `EyeLook` is eye-in-head movement, whereas OpenFace gaze angles are camera/world referenced. The primary analysis therefore uses head-relative OpenFace gaze: horizontal gaze minus head yaw, and sign-aligned vertical gaze minus head pitch.

### Median within-clip trajectory correlation

| Embodiment | Horizontal r | Vertical r |
|---|---:|---:|
{chr(10).join(within_rows)}

### Correlation of clip means across the 64 stimuli

| Embodiment | Horizontal r | Vertical r |
|---|---:|---:|
{chr(10).join(mean_rows)}

### Correlation of within-clip dynamic ranges across stimuli

| Embodiment | Horizontal r | Vertical r |
|---|---:|---:|
{chr(10).join(range_rows)}

## Interpretation boundary

The Human condition is the validity benchmark because OpenFace was designed for human faces. Strong Human correspondence supports interpreting the corresponding avatar result; weak Human correspondence means that avatar differences cannot be separated confidently from OpenFace measurement limitations. The results should therefore be interpreted separately for horizontal and vertical gaze rather than as a single general gaze effect.

Actor-cluster bootstrap confidence intervals and all additional metrics are in `summary_metrics.csv`. Raw world-coordinate gaze results are retained in `clip_level_metrics.csv` as a sensitivity analysis.
"""
    (output_dir / "SUMMARY.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for audio synchronization")

    stimuli = selected_stimuli(data_root)
    if len(stimuli) != 64:
        raise ValueError(f"Expected 64 selected stimuli, found {len(stimuli)}")
    require_inputs(data_root, stimuli)

    manifest = estimate_manifest(data_root, stimuli, args.audio_rate)
    clips = load_clip_data(data_root, manifest)
    visual_lags = estimate_visual_lags(clips)
    clip_metrics = calculate_clip_metrics(clips, visual_lags)
    summary = summarize_metrics(clip_metrics, args.bootstrap, args.seed)
    quality = summarize_data_quality(manifest)

    manifest.to_csv(output_dir / "manifest.csv", index=False)
    visual_lags.to_csv(output_dir / "visual_alignment_lags.csv", index=False)
    clip_metrics.to_csv(output_dir / "clip_level_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary_metrics.csv", index=False)
    quality.to_csv(output_dir / "data_quality_summary.csv", index=False)
    plot_results(clip_metrics, output_dir)
    write_summary(output_dir, manifest, visual_lags, summary)
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "data_root": str(data_root),
                "n_stimuli": len(stimuli),
                "audio_rate": args.audio_rate,
                "bootstrap_repetitions": args.bootstrap,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote gaze-validation results to {output_dir}")


if __name__ == "__main__":
    main()
