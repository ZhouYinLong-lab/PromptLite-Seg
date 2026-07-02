from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return float(low), float(high)


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(values.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, len(values)))
    permuted = np.abs((signs * values).mean(axis=1))
    return float((np.count_nonzero(permuted >= observed) + 1) / (n_perm + 1))


def paired_comparison(
    *,
    name: str,
    df: pd.DataFrame,
    key_cols: list[str],
    condition_col: str,
    baseline: str,
    candidate: str,
    metric: str,
    rng: np.random.Generator,
    n_boot: int,
    n_perm: int,
) -> dict:
    table = df.pivot_table(index=key_cols, columns=condition_col, values=metric, aggfunc="mean")
    table = table.dropna(subset=[baseline, candidate])
    deltas = (table[candidate] - table[baseline]).to_numpy(dtype=np.float64)
    low, high = bootstrap_ci(deltas, rng, n_boot)
    p_value = sign_flip_pvalue(deltas, rng, n_perm)
    return {
        "comparison": name,
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "num_pairs": len(deltas),
        "baseline_mean": f"{table[baseline].mean():.6f}",
        "candidate_mean": f"{table[candidate].mean():.6f}",
        "mean_delta": f"{deltas.mean():.6f}",
        "ci95_low": f"{low:.6f}",
        "ci95_high": f"{high:.6f}",
        "paired_sign_flip_p": f"{p_value:.6f}",
        "significant_005": str(p_value < 0.05 and (low > 0 or high < 0)).lower(),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_effects(rows: list[dict], out_path: Path) -> None:
    selected = [
        row
        for row in rows
        if row["metric"] == "iou"
        and row["comparison"]
        in {
            "robust_superpixel_vs_center_color",
            "sam_box_only_vs_point_box",
            "sam_point_box_vs_point_only",
            "moderate_box_noise_vs_point_noise",
            "moderate_score_select_vs_single_noisy",
            "moderate_oracle_vs_single_noisy",
        }
    ]
    labels = [
        "Robust superpixel - center color",
        "SAM box-only - point+box",
        "SAM point+box - point-only",
        "Moderate box noise - point noise",
        "Score select - single noisy",
        "Oracle best - single noisy",
    ]
    order = [row["comparison"] for row in selected]
    label_lookup = dict(zip(order, labels))

    y = np.arange(len(selected))
    means = np.array([float(row["mean_delta"]) for row in selected])
    lows = np.array([float(row["ci95_low"]) for row in selected])
    highs = np.array([float(row["ci95_high"]) for row in selected])

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    colors = ["#4c78a8" if value >= 0 else "#e45756" for value in means]
    ax.barh(y, means, color=colors, alpha=0.88)
    ax.errorbar(
        means,
        y,
        xerr=np.vstack([means - lows, highs - means]),
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.2,
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(y, [label_lookup[row["comparison"]] for row in selected])
    ax.set_xlabel("mean IoU delta with 95% bootstrap CI")
    ax.set_title("Paired statistical effects")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_markdown(path: Path, rows: list[dict]) -> None:
    iou_rows = [row for row in rows if row["metric"] == "iou"]
    lines = [
        "# Statistical Reliability Analysis",
        "",
        "All intervals are paired bootstrap 95% confidence intervals over sample-level deltas.",
        "The p-values use a paired sign-flip permutation test against zero mean delta.",
        "",
        "| Comparison | Pairs | Baseline | Candidate | Delta | 95% CI | p-value |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in iou_rows:
        lines.append(
            f"| {row['comparison']} | {row['num_pairs']} | {row['baseline_mean']} | "
            f"{row['candidate_mean']} | {row['mean_delta']} | "
            f"[{row['ci95_low']}, {row['ci95_high']}] | {row['paired_sign_flip_p']} |"
        )

    def lookup(name: str) -> dict:
        for row in iou_rows:
            if row["comparison"] == name:
                return row
        raise KeyError(name)

    robust = lookup("robust_superpixel_vs_center_color")
    box_noise = lookup("moderate_box_noise_vs_point_noise")
    score = lookup("moderate_score_select_vs_single_noisy")
    box_modality = lookup("sam_box_only_vs_point_box")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- The robust superpixel baseline improves over center-color by {robust['mean_delta']} IoU "
            f"with CI [{robust['ci95_low']}, {robust['ci95_high']}].",
            f"- Box-only SAM is not just visually better than point+box here: the paired delta is "
            f"{box_modality['mean_delta']} IoU with CI [{box_modality['ci95_low']}, {box_modality['ci95_high']}].",
            f"- Moderate box noise is substantially worse than point noise: delta {box_noise['mean_delta']} "
            f"with CI [{box_noise['ci95_low']}, {box_noise['ci95_high']}].",
            f"- Score selection has a positive mean delta of {score['mean_delta']} IoU, "
            f"but its CI [{score['ci95_low']}, {score['ci95_high']}] crosses zero, so the current "
            "30-sample subset should treat this as promising but not statistically confirmed.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-metrics", type=Path, default=Path("outputs/metrics.csv"))
    parser.add_argument(
        "--prompt-uncertainty-metrics",
        type=Path,
        default=Path("outputs/prompt_uncertainty/metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/statistics"))
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--n-perm", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []

    base = pd.read_csv(args.base_metrics)
    prompt = pd.read_csv(args.prompt_uncertainty_metrics)

    for metric in ("iou", "dice"):
        rows.append(
            paired_comparison(
                name="robust_superpixel_vs_center_color",
                df=base,
                key_cols=["sample_id"],
                condition_col="method",
                baseline="center_color",
                candidate="robust_superpixel",
                metric=metric,
                rng=rng,
                n_boot=args.n_boot,
                n_perm=args.n_perm,
            )
        )

    modality = prompt[prompt["experiment"] == "modality"].copy()
    for metric in ("iou", "dice"):
        for name, baseline, candidate in (
            ("sam_box_only_vs_point_box", "point_box", "box_only"),
            ("sam_point_box_vs_point_only", "point_only", "point_box"),
            ("sam_box_only_vs_point_only", "point_only", "box_only"),
        ):
            rows.append(
                paired_comparison(
                    name=name,
                    df=modality,
                    key_cols=["sample_id"],
                    condition_col="condition",
                    baseline=baseline,
                    candidate=candidate,
                    metric=metric,
                    rng=rng,
                    n_boot=args.n_boot,
                    n_perm=args.n_perm,
                )
            )

    noise = prompt[
        (prompt["experiment"] == "noise_decomposition")
        & (prompt["severity"].isin(["clean", "moderate"]))
    ].copy()
    sample_noise = (
        noise.groupby(["sample_id", "condition"], as_index=False)[["iou", "dice"]]
        .mean()
        .reset_index(drop=True)
    )
    for metric in ("iou", "dice"):
        for name, baseline, candidate in (
            ("moderate_point_noise_vs_clean", "clean", "point_noise"),
            ("moderate_box_noise_vs_clean", "clean", "box_noise"),
            ("moderate_point_box_noise_vs_clean", "clean", "point_box_noise"),
            ("moderate_box_noise_vs_point_noise", "point_noise", "box_noise"),
            ("moderate_point_box_noise_vs_point_noise", "point_noise", "point_box_noise"),
        ):
            rows.append(
                paired_comparison(
                    name=name,
                    df=sample_noise,
                    key_cols=["sample_id"],
                    condition_col="condition",
                    baseline=baseline,
                    candidate=candidate,
                    metric=metric,
                    rng=rng,
                    n_boot=args.n_boot,
                    n_perm=args.n_perm,
                )
            )

    ensemble = prompt[
        (prompt["experiment"] == "uncertainty_ensemble")
        & (prompt["severity"] == "moderate")
        & (prompt["condition"] == "point_box_noise")
    ].copy()
    sample_ensemble = (
        ensemble.groupby(["sample_id", "method"], as_index=False)[["iou", "dice"]]
        .mean()
        .reset_index(drop=True)
    )
    for metric in ("iou", "dice"):
        for name, candidate in (
            ("moderate_score_select_vs_single_noisy", "sam_score_select"),
            ("moderate_consistency_medoid_vs_single_noisy", "sam_consistency_medoid"),
            ("moderate_vote_consensus_vs_single_noisy", "sam_vote_consensus"),
            ("moderate_oracle_vs_single_noisy", "sam_oracle_best"),
        ):
            rows.append(
                paired_comparison(
                    name=name,
                    df=sample_ensemble,
                    key_cols=["sample_id"],
                    condition_col="method",
                    baseline="sam_single_noisy",
                    candidate=candidate,
                    metric=metric,
                    rng=rng,
                    n_boot=args.n_boot,
                    n_perm=args.n_perm,
                )
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_effects.csv", rows)
    plot_effects(rows, args.output_dir / "paired_effects.png")
    write_markdown(args.output_dir / "statistical_reliability.md", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "n_boot": args.n_boot,
                "n_perm": args.n_perm,
                "seed": args.seed,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "num_comparisons": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
