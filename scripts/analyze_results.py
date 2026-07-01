from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_metrics(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["iou"] = float(row["iou"])
        row["dice"] = float(row["dice"])
        row["mask_pixels"] = int(row["mask_pixels"])
        row["pred_pixels"] = int(row["pred_pixels"])
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def per_class_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["class_name"], row["method"])].append(row)

    out = []
    for (class_name, method), items in sorted(grouped.items()):
        mean_iou, std_iou = mean_std([x["iou"] for x in items])
        mean_dice, std_dice = mean_std([x["dice"] for x in items])
        out.append(
            {
                "class_name": class_name,
                "method": method,
                "num_samples": len(items),
                "mean_iou": f"{mean_iou:.6f}",
                "std_iou": f"{std_iou:.6f}",
                "mean_dice": f"{mean_dice:.6f}",
                "std_dice": f"{std_dice:.6f}",
            }
        )
    return out


def sample_comparison(rows: list[dict], baseline: str, improved: str) -> list[dict]:
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample_id"]][row["method"]] = row

    out = []
    for sample_id, methods in sorted(by_sample.items()):
        if baseline not in methods or improved not in methods:
            continue
        base = methods[baseline]
        new = methods[improved]
        out.append(
            {
                "sample_id": sample_id,
                "class_name": new["class_name"],
                "baseline_iou": f"{base['iou']:.6f}",
                "improved_iou": f"{new['iou']:.6f}",
                "delta_iou": f"{new['iou'] - base['iou']:.6f}",
                "baseline_dice": f"{base['dice']:.6f}",
                "improved_dice": f"{new['dice']:.6f}",
                "delta_dice": f"{new['dice'] - base['dice']:.6f}",
                "mask_pixels": new["mask_pixels"],
                "pred_pixels": new["pred_pixels"],
            }
        )
    out.sort(key=lambda x: float(x["delta_iou"]), reverse=True)
    return out


def plot_per_class(summary_rows: list[dict], out_path: Path) -> None:
    methods = sorted({row["method"] for row in summary_rows})
    classes = sorted({row["class_name"] for row in summary_rows})
    lookup = {(row["class_name"], row["method"]): float(row["mean_iou"]) for row in summary_rows}
    x = np.arange(len(classes))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(max(9, len(classes) * 0.65), 4.8))
    for idx, method in enumerate(methods):
        values = [lookup.get((class_name, method), 0.0) for class_name in classes]
        ax.bar(x + (idx - (len(methods) - 1) / 2) * width, values, width, label=method)
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean IoU")
    ax.set_title("Per-class prompt segmentation IoU")
    ax.set_xticks(x, classes, rotation=35, ha="right")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_delta_hist(comparison_rows: list[dict], out_path: Path) -> None:
    deltas = [float(row["delta_iou"]) for row in comparison_rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(deltas, bins=10, color="#4c78a8", edgecolor="white")
    ax.axvline(0.0, color="#d62728", linestyle="--", linewidth=1.5)
    ax.set_xlabel("IoU improvement over center_color")
    ax.set_ylabel("number of samples")
    ax.set_title("Sample-level improvement distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_success_failure(path: Path, comparison_rows: list[dict], top_k: int) -> None:
    successes = comparison_rows[:top_k]
    failures = sorted(comparison_rows, key=lambda x: float(x["improved_iou"]))[:top_k]
    regressions = sorted(comparison_rows, key=lambda x: float(x["delta_iou"]))[:top_k]

    lines = [
        "# Success and Failure Analysis",
        "",
        "This analysis compares `robust_superpixel` against the `center_color` baseline.",
        "",
        "## Largest Improvements",
        "",
        "| Sample | Class | Baseline IoU | Robust IoU | Delta IoU |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in successes:
        lines.append(
            f"| {row['sample_id']} | {row['class_name']} | {row['baseline_iou']} | "
            f"{row['improved_iou']} | {row['delta_iou']} |"
        )

    lines.extend(
        [
            "",
            "## Hardest Cases by Robust IoU",
            "",
            "| Sample | Class | Robust IoU | Robust Dice |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    improved_dice_lookup = {row["sample_id"]: row["improved_dice"] for row in comparison_rows}
    for row in failures:
        lines.append(f"| {row['sample_id']} | {row['class_name']} | {row['improved_iou']} | {improved_dice_lookup[row['sample_id']]} |")

    lines.extend(
        [
            "",
            "## Largest Regressions or Ties",
            "",
            "| Sample | Class | Baseline IoU | Robust IoU | Delta IoU |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in regressions:
        lines.append(
            f"| {row['sample_id']} | {row['class_name']} | {row['baseline_iou']} | "
            f"{row['improved_iou']} | {row['delta_iou']} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--baseline", default="center_color")
    parser.add_argument("--improved", default="robust_superpixel")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = read_metrics(args.metrics)
    class_rows = per_class_summary(rows)
    comparison_rows = sample_comparison(rows, args.baseline, args.improved)

    write_csv(
        args.output_dir / "per_class_summary.csv",
        class_rows,
        ["class_name", "method", "num_samples", "mean_iou", "std_iou", "mean_dice", "std_dice"],
    )
    write_csv(
        args.output_dir / "sample_comparison.csv",
        comparison_rows,
        [
            "sample_id",
            "class_name",
            "baseline_iou",
            "improved_iou",
            "delta_iou",
            "baseline_dice",
            "improved_dice",
            "delta_dice",
            "mask_pixels",
            "pred_pixels",
        ],
    )
    plot_per_class(class_rows, args.output_dir / "per_class_iou.png")
    plot_delta_hist(comparison_rows, args.output_dir / "iou_delta_histogram.png")
    write_success_failure(args.output_dir / "success_failure.md", comparison_rows, args.top_k)

    mean_delta = np.mean([float(row["delta_iou"]) for row in comparison_rows])
    wins = sum(float(row["delta_iou"]) > 1e-9 for row in comparison_rows)
    ties = sum(abs(float(row["delta_iou"])) <= 1e-9 for row in comparison_rows)
    losses = len(comparison_rows) - wins - ties
    print(f"Analyzed {len(comparison_rows)} samples.")
    print(f"Mean IoU delta: {mean_delta:.4f}; wins/ties/losses: {wins}/{ties}/{losses}")


if __name__ == "__main__":
    main()
