from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from carla_rl_lab.utils import restore_training_state


ScalarPoint = Tuple[int, float, float]


def moving_average(
    steps: Sequence[int], values: Sequence[float], window: int
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(steps, dtype=np.int64)
    y = np.asarray(values, dtype=np.float64)
    effective_window = min(max(int(window), 1), len(y))
    if len(y) == 0 or effective_window == 1:
        return x, y
    weights = np.ones(effective_window, dtype=np.float64) / effective_window
    return x[effective_window - 1 :], np.convolve(y, weights, mode="valid")


def load_scalars(run_dir: str) -> Dict[str, List[ScalarPoint]]:
    accumulator = EventAccumulator(run_dir)
    accumulator.Reload()
    return {
        tag: [
            (int(event.step), float(event.wall_time), float(event.value))
            for event in accumulator.Scalars(tag)
        ]
        for tag in sorted(accumulator.Tags().get("scalars", []))
    }


def scalar_summary(points: Iterable[ScalarPoint]) -> Dict[str, Any]:
    values = list(points)
    if not values:
        return {"count": 0}
    samples = np.asarray([point[2] for point in values], dtype=np.float64)
    return {
        "count": len(values),
        "first_step": values[0][0],
        "last_step": values[-1][0],
        "first": float(samples[0]),
        "last": float(samples[-1]),
        "min": float(samples.min()),
        "max": float(samples.max()),
        "mean": float(samples.mean()),
    }


def sampled_points(
    points: Sequence[ScalarPoint], max_points: int
) -> Sequence[ScalarPoint]:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return [points[index] for index in np.unique(indices)]


def write_scalar_csv(
    path: str,
    scalars: Dict[str, List[ScalarPoint]],
    max_points_per_tag: int,
) -> None:
    with open(path, "w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(("tag", "step", "wall_time", "value"))
        for tag, points in scalars.items():
            for step, wall_time, value in sampled_points(
                points, max_points_per_tag
            ):
                writer.writerow((tag, step, "{:.6f}".format(wall_time), value))


def plot_reward(
    path: str,
    scalars: Dict[str, List[ScalarPoint]],
    window: int,
    title: str,
) -> bool:
    points = scalars.get("episode/reward", [])
    if not points:
        return False
    steps = [point[0] for point in points]
    values = [point[2] for point in points]
    smooth_steps, smooth_values = moving_average(steps, values, window)

    figure, axis = plt.subplots(figsize=(9.0, 4.8), dpi=160)
    axis.plot(steps, values, color="#94a3b8", linewidth=1.0, alpha=0.7, label="Episode")
    axis.plot(
        smooth_steps,
        smooth_values,
        color="#0f766e",
        linewidth=2.2,
        label="Moving mean ({})".format(min(window, len(values))),
    )
    axis.axhline(0.0, color="#111827", linewidth=0.8, alpha=0.5)
    axis.set_title(title or "Episode return")
    axis.set_xlabel("Environment step")
    axis.set_ylabel("Return")
    axis.ticklabel_format(style="plain", axis="x", useOffset=False)
    axis.grid(True, color="#e5e7eb", linewidth=0.7)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_losses(
    path: str,
    scalars: Dict[str, List[ScalarPoint]],
    window: int,
    title: str,
) -> bool:
    loss_tags = [
        tag for tag in scalars if tag.startswith("train/") and "loss" in tag
    ]
    if not loss_tags:
        return False
    colors = ("#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2")
    column_count = 2
    row_count = int(math.ceil(len(loss_tags) / float(column_count)))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(10.0, 3.2 * row_count),
        dpi=160,
        squeeze=False,
    )
    for index, tag in enumerate(loss_tags):
        axis = axes[index // column_count][index % column_count]
        points = scalars[tag]
        steps = [point[0] for point in points]
        values = [point[2] for point in points]
        smooth_steps, smooth_values = moving_average(steps, values, window)
        color = colors[index % len(colors)]
        axis.plot(
            steps,
            values,
            color=color,
            linewidth=0.6,
            alpha=0.16,
            label="Raw",
        )
        axis.plot(
            smooth_steps,
            smooth_values,
            color=color,
            linewidth=1.7,
            label="Moving mean ({})".format(min(window, len(values))),
        )
        axis.axhline(0.0, color="#111827", linewidth=0.7, alpha=0.4)
        axis.set_title(tag.split("/", 1)[-1])
        axis.set_xlabel("Training step")
        axis.set_ylabel("Loss")
        axis.ticklabel_format(style="plain", axis="x", useOffset=False)
        axis.grid(True, color="#e5e7eb", linewidth=0.7)
        axis.legend(frameon=False)
    for index in range(len(loss_tags), row_count * column_count):
        axes[index // column_count][index % column_count].axis("off")
    figure.suptitle(title or "Training losses", fontsize=14)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return True


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as source:
        return json.load(source)


def result_summary(
    run_dir: str,
    scalars: Dict[str, List[ScalarPoint]],
    benchmark_report_path: str,
) -> Dict[str, Any]:
    run_config_path = os.path.join(run_dir, "run_config.json")
    run_record = read_json(run_config_path) if os.path.isfile(run_config_path) else {}
    record: Dict[str, Any] = {
        "schema_version": 1,
        "run": run_record,
        "scalars": {
            tag: scalar_summary(points) for tag, points in scalars.items()
        },
    }
    if benchmark_report_path:
        report = read_json(benchmark_report_path)
        checkpoint = report.get("checkpoint", {})
        checkpoint_path = checkpoint.get("path", "")
        trainer_state = (
            restore_training_state(checkpoint_path, restore_rng=False)
            if checkpoint_path and os.path.isfile(checkpoint_path)
            else {}
        )
        record["benchmark"] = {
            "name": report.get("benchmark"),
            "protocol": report.get("protocol"),
            "episodes": report.get("episodes"),
            "summary": report.get("summary"),
        }
        record["checkpoint"] = {
            "sha256": checkpoint.get("sha256"),
            "metadata": checkpoint.get("metadata"),
            "carla_versions": trainer_state.get("carla_versions", {}),
        }
    return record


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export reproducible CSV/JSON/PNG curves from TensorBoard"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--benchmark-report", default="")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max-csv-points-per-tag", type=int, default=1000)
    parser.add_argument("--title", default="")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.window <= 0:
        raise ValueError("--window must be positive")
    if args.max_csv_points_per_tag <= 0:
        raise ValueError("--max-csv-points-per-tag must be positive")
    os.makedirs(args.output_dir, exist_ok=True)
    scalars = load_scalars(args.run_dir)
    if not scalars:
        raise ValueError("No TensorBoard scalar events found in {}".format(args.run_dir))

    write_scalar_csv(
        os.path.join(args.output_dir, "scalars.csv"),
        scalars,
        args.max_csv_points_per_tag,
    )
    plot_reward(
        os.path.join(args.output_dir, "episode_reward.png"),
        scalars,
        args.window,
        "{} episode return".format(args.title).strip(),
    )
    plot_losses(
        os.path.join(args.output_dir, "training_losses.png"),
        scalars,
        args.window,
        "{} training losses".format(args.title).strip(),
    )
    summary = result_summary(args.run_dir, scalars, args.benchmark_report)
    summary_path = os.path.join(args.output_dir, "result.json")
    with open(summary_path, "w") as output:
        json.dump(summary, output, indent=2, sort_keys=True)
        output.write("\n")
    print("Curves and scalar data -> {}".format(os.path.abspath(args.output_dir)))


if __name__ == "__main__":
    main()
