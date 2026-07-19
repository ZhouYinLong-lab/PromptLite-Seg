"""Pre-specified confirmatory statistics with Holm family-wise correction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from promptseg.protocol import sha256_file
from promptseg.utils import write_csv


CPU_REQUIRED_COLUMNS = {"sample_id", "class_name", "method", "status", "error", "iou", "dice", "latency_ms"}
SAM_REQUIRED_COLUMNS = {
    "sample_id",
    "class_name",
    "experiment",
    "severity",
    "condition",
    "method",
    "trial",
    "iou",
    "dice",
    "sam_score",
    "point_hit",
    "box_iou",
}
SAM_ENSEMBLE_METHODS = {
    "sam_single_noisy",
    "sam_score_select",
    "sam_consistency_medoid",
    "sam_vote_consensus",
    "sam_oracle_best",
}


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int, batch: int = 1000) -> tuple[float, float]:
    means: list[np.ndarray] = []
    for start in range(0, n_boot, batch):
        count = min(batch, n_boot - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means.append(values[indices].mean(axis=1))
    low, high = np.percentile(np.concatenate(means), [2.5, 97.5])
    return float(low), float(high)


def sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator, n_perm: int, batch: int = 1000) -> float:
    observed = abs(float(values.mean()))
    extreme = 0
    for start in range(0, n_perm, batch):
        count = min(batch, n_perm - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(values)))
        permuted = np.abs((signs * values).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed))
    return float((extreme + 1) / (n_perm + 1))


def effect_row(hypothesis: str, comparison: str, values: np.ndarray, rng, n_boot: int, n_perm: int) -> dict:
    low, high = bootstrap_ci(values, rng, n_boot)
    p_value = sign_flip_pvalue(values, rng, n_perm)
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "num_pairs": len(values),
        "mean_delta_iou": float(values.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "p_raw": p_value,
    }


def holm_adjust(rows: list[dict], alpha: float = 0.05) -> list[dict]:
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["p_raw"])
    running_adjusted = 0.0
    continue_rejecting = True
    for rank, index in enumerate(ordered):
        multiplier = len(rows) - rank
        adjusted = min(1.0, multiplier * rows[index]["p_raw"])
        running_adjusted = max(running_adjusted, adjusted)
        threshold = alpha / multiplier
        reject = continue_rejecting and rows[index]["p_raw"] <= threshold
        if not reject:
            continue_rejecting = False
        rows[index]["p_holm"] = running_adjusted
        rows[index]["holm_threshold"] = threshold
        rows[index]["reject_holm_005"] = reject
    return rows


def pivot_delta(frame: pd.DataFrame, condition_col: str, baseline: str, candidate: str) -> np.ndarray:
    table = frame.pivot(index="sample_id", columns=condition_col, values="iou").dropna(subset=[baseline, candidate])
    return (table[candidate] - table[baseline]).to_numpy(dtype=np.float64)


def load_manifest(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    frame["target_area_quartile"] = pd.qcut(frame["target_area"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return frame[["sample_id", "class_name", "target_area", "target_area_quartile"]]


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} metrics are missing columns: {missing}")


def validate_execution_summaries(
    cpu_summary_path: Path,
    sam_summary_path: Path,
    cpu_metrics_path: Path,
    sam_metrics_path: Path,
    manifest_path: Path,
    protocol_path: Path,
    dataset_fingerprints_path: Path,
) -> dict:
    cpu_summary = json.loads(cpu_summary_path.read_text(encoding="utf-8"))
    sam_summary = json.loads(sam_summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    dataset_spec = json.loads(dataset_fingerprints_path.read_text(encoding="utf-8"))
    expected_dataset = dataset_spec["confirmatory"]
    expected_dataset_sha256 = expected_dataset["dataset_sha256"]
    expected_samples = int(expected_dataset["samples"])

    for label, summary in (("CPU", cpu_summary), ("SAM", sam_summary)):
        if summary.get("confirmatory") is not True:
            raise ValueError(f"{label} summary is not marked as a confirmatory run")
        if summary.get("dataset_sha256") != expected_dataset_sha256:
            raise ValueError(f"{label} dataset fingerprint does not match the frozen confirmatory data")
    if cpu_summary.get("expected_confirmatory_samples") != expected_samples:
        raise ValueError("CPU summary sample count does not match the frozen dataset specification")
    if sam_summary.get("num_samples") != expected_samples:
        raise ValueError("SAM summary sample count does not match the frozen dataset specification")
    manifest_sha256 = sha256_file(manifest_path)
    if {cpu_summary.get("manifest_sha256"), sam_summary.get("manifest_sha256")} != {manifest_sha256}:
        raise ValueError("CPU and SAM summaries are not bound to the selected frozen manifest")

    current_protocol_sha256 = sha256_file(protocol_path)
    accepted_protocols = {
        current_protocol_sha256,
        *protocol.get("integrity", {}).get("accepted_historical_protocol_sha256", []),
    }
    summary_protocols = {cpu_summary.get("protocol_sha256"), sam_summary.get("protocol_sha256")}
    if len(summary_protocols) != 1 or not summary_protocols <= accepted_protocols:
        raise ValueError("CPU and SAM summaries do not share an accepted protocol fingerprint")
    execution_protocol_sha256 = next(iter(summary_protocols))
    historical_metrics = protocol.get("integrity", {}).get(
        "accepted_historical_metric_sha256", {}
    ).get(execution_protocol_sha256, {})
    metric_bindings = (
        ("CPU", cpu_summary, cpu_metrics_path, historical_metrics.get("cpu")),
        ("SAM", sam_summary, sam_metrics_path, historical_metrics.get("sam")),
    )
    for label, summary, metrics_path, historical_hash in metric_bindings:
        if execution_protocol_sha256 == current_protocol_sha256 and summary.get(
            "source_tree_matches"
        ) is not True:
            raise ValueError(f"{label} runtime source tree is not verified")
        expected_metrics_sha256 = summary.get("metrics_sha256") or historical_hash
        if expected_metrics_sha256 != sha256_file(metrics_path):
            raise ValueError(f"{label} metrics do not match their execution summary")
    return {
        "cpu_summary_sha256": sha256_file(cpu_summary_path),
        "sam_summary_sha256": sha256_file(sam_summary_path),
        "dataset_fingerprints_sha256": sha256_file(dataset_fingerprints_path),
        "dataset_sha256": expected_dataset_sha256,
        "manifest_sha256": manifest_sha256,
        "execution_protocol_sha256": execution_protocol_sha256,
    }


def validate_metric_design(
    cpu: pd.DataFrame,
    sam: pd.DataFrame,
    manifest: pd.DataFrame,
    protocol: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate the complete frozen design before any aggregation or coercion."""

    _require_columns(cpu, CPU_REQUIRED_COLUMNS, "CPU")
    _require_columns(sam, SAM_REQUIRED_COLUMNS, "SAM")
    sample_ids = manifest["sample_id"].astype(str).tolist()
    sample_set = set(sample_ids)
    class_lookup = dict(zip(manifest["sample_id"].astype(str), manifest["class_name"].astype(str)))
    if set(cpu["sample_id"].astype(str)) != sample_set or set(sam["sample_id"].astype(str)) != sample_set:
        raise ValueError("CPU, SAM, and frozen manifest sample IDs do not match")
    for label, frame in (("CPU", cpu), ("SAM", sam)):
        mismatched = frame[
            frame["sample_id"].astype(str).map(class_lookup) != frame["class_name"].astype(str)
        ]
        if not mismatched.empty:
            raise ValueError(f"{label} class labels do not match the frozen manifest")

    cpu_methods = [
        *protocol["methods"]["cpu_baselines"],
        protocol["methods"]["proposed"],
        *protocol["methods"]["ablations"],
    ]
    cpu_keys = list(zip(cpu["sample_id"].astype(str), cpu["method"].astype(str)))
    expected_cpu = {(sample_id, method) for sample_id in sample_ids for method in cpu_methods}
    if len(cpu_keys) != len(expected_cpu) or set(cpu_keys) != expected_cpu:
        raise ValueError("CPU metrics do not contain exactly one row per frozen sample and method")
    if set(cpu["status"].astype(str)) - {"ok", "error"}:
        raise ValueError("CPU metrics contain an unknown status")
    if (cpu.loc[cpu["status"] == "ok", "error"].astype(str) != "").any():
        raise ValueError("CPU successful rows unexpectedly contain an error message")
    if (cpu.loc[cpu["status"] == "error", "error"].astype(str) == "").any():
        raise ValueError("CPU failed rows must contain an explicit error message")
    for metric in ("iou", "dice", "latency_ms"):
        numeric = pd.to_numeric(cpu[metric], errors="coerce")
        if numeric[cpu["status"] == "ok"].isna().any():
            raise ValueError(f"CPU successful rows contain non-numeric {metric}")
        if metric in {"iou", "dice"} and numeric[cpu["status"] == "error"].notna().any():
            raise ValueError(f"CPU failed rows unexpectedly contain numeric {metric}")
        cpu[metric] = numeric.fillna(0.0)
    if (cpu["latency_ms"] < 0.0).any():
        raise ValueError("CPU metrics contain negative latency")

    trials = int(protocol["methods"]["sam_confirmatory_trials_per_sample"])
    expected_sam: set[tuple[str, str, str, str, str, int]] = set()
    for sample_id in sample_ids:
        expected_sam.update(
            (sample_id, "modality", "clean", mode, "sam_vit_b", 0)
            for mode in ("point_only", "box_only", "point_box")
        )
        expected_sam.update(
            (sample_id, "noise_decomposition", "moderate", condition, "sam_single_prompt", trial)
            for condition in ("point_noise", "box_noise")
            for trial in range(trials)
        )
        expected_sam.update(
            (sample_id, "uncertainty_ensemble", "moderate", "point_box_noise", method, trial)
            for method in SAM_ENSEMBLE_METHODS
            for trial in range(trials)
        )
    trial_values = pd.to_numeric(sam["trial"], errors="coerce")
    if trial_values.isna().any() or not np.equal(trial_values, np.floor(trial_values)).all():
        raise ValueError("SAM trial values must be integers")
    sam["trial"] = trial_values.astype(int)
    sam_keys = list(
        zip(
            sam["sample_id"].astype(str),
            sam["experiment"].astype(str),
            sam["severity"].astype(str),
            sam["condition"].astype(str),
            sam["method"].astype(str),
            sam["trial"],
        )
    )
    if len(sam_keys) != len(expected_sam) or set(sam_keys) != expected_sam:
        raise ValueError("SAM metrics do not match the complete frozen condition/trial design")
    for metric in ("iou", "dice", "sam_score"):
        numeric = pd.to_numeric(sam[metric], errors="coerce")
        if numeric.isna().any():
            raise ValueError(f"SAM metrics contain non-numeric {metric}")
        sam[metric] = numeric
    noisy = sam["experiment"] != "modality"
    box_iou = pd.to_numeric(sam["box_iou"], errors="coerce")
    if box_iou[noisy].isna().any() or box_iou[~noisy].notna().any():
        raise ValueError("SAM box_iou must be numeric for noisy rows and blank for modality rows")
    sam["box_iou"] = box_iou
    if not sam.loc[noisy, "box_iou"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("SAM metrics contain out-of-range box_iou")
    point_hit_text = sam["point_hit"].astype(str).str.lower()
    if not point_hit_text[noisy].isin({"true", "false"}).all() or (point_hit_text[~noisy] != "").any():
        raise ValueError("SAM point_hit must be boolean for noisy rows and blank for modality rows")
    sam["point_hit"] = point_hit_text.map({"true": True, "false": False})
    for label, frame in (("CPU", cpu), ("SAM", sam)):
        for metric in ("iou", "dice"):
            valid = frame[metric].between(0.0, 1.0, inclusive="both")
            if not valid.all():
                raise ValueError(f"{label} metrics contain out-of-range {metric}")
    return cpu, sam


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-metrics", type=Path, default=Path("outputs_confirmatory/cpu/metrics.csv"))
    parser.add_argument("--sam-metrics", type=Path, default=Path("outputs_confirmatory/sam/metrics.csv"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("protocol/manifests/confirmatory_validation.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_confirmatory/statistics"))
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--n-perm", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--protocol", type=Path, default=Path("protocol/research_protocol.json"))
    parser.add_argument("--cpu-summary", type=Path, default=None)
    parser.add_argument("--sam-summary", type=Path, default=None)
    parser.add_argument(
        "--dataset-fingerprints",
        type=Path,
        default=Path("protocol/dataset_fingerprints.json"),
    )
    args = parser.parse_args()
    if args.n_boot < 1 or args.n_perm < 1:
        parser.error("--n-boot and --n-perm must be at least 1")

    cpu_summary_path = args.cpu_summary or args.cpu_metrics.parent / "summary.json"
    sam_summary_path = args.sam_summary or args.sam_metrics.parent / "summary.json"
    execution_provenance = validate_execution_summaries(
        cpu_summary_path,
        sam_summary_path,
        args.cpu_metrics,
        args.sam_metrics,
        args.manifest,
        args.protocol,
        args.dataset_fingerprints,
    )
    cpu = pd.read_csv(args.cpu_metrics, keep_default_na=False)
    sam = pd.read_csv(args.sam_metrics, keep_default_na=False)
    manifest = load_manifest(args.manifest)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    cpu, sam = validate_metric_design(cpu, sam, manifest, protocol)

    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    rows.append(
        effect_row(
            "H1",
            "robust_superpixel - center_color",
            pivot_delta(cpu, "method", "center_color", "robust_superpixel"),
            rng,
            args.n_boot,
            args.n_perm,
        )
    )

    ensemble = (
        sam[sam["experiment"] == "uncertainty_ensemble"]
        .groupby(["sample_id", "method"], as_index=False)["iou"]
        .mean()
    )
    rows.append(
        effect_row(
            "H2",
            "sam_score_select - sam_single_noisy",
            pivot_delta(ensemble, "method", "sam_single_noisy", "sam_score_select"),
            rng,
            args.n_boot,
            args.n_perm,
        )
    )

    noise = (
        sam[sam["experiment"] == "noise_decomposition"]
        .groupby(["sample_id", "condition"], as_index=False)["iou"]
        .mean()
    )
    rows.append(
        effect_row(
            "H3",
            "point_noise IoU - box_noise IoU",
            pivot_delta(noise, "condition", "box_noise", "point_noise"),
            rng,
            args.n_boot,
            args.n_perm,
        )
    )
    holm_adjust(rows)
    expected_pairs = len(manifest)
    if any(int(row["num_pairs"]) != expected_pairs for row in rows):
        raise RuntimeError(f"Every primary hypothesis must contain exactly {expected_pairs} paired samples")

    cpu_summary = (
        cpu.groupby("method", as_index=False)
        .agg(
            num_samples=("sample_id", "count"),
            num_failures=("status", lambda values: int((values != "ok").sum())),
            mean_iou=("iou", "mean"),
            mean_dice=("dice", "mean"),
            median_latency_ms=("latency_ms", "median"),
        )
        .to_dict("records")
    )
    modality_summary = (
        sam[sam["experiment"] == "modality"]
        .groupby("condition", as_index=False)[["iou", "dice"]]
        .mean()
        .to_dict("records")
    )
    point_rows = sam[(sam["experiment"] == "noise_decomposition") & (sam["condition"] == "point_noise")]
    box_rows = sam[(sam["experiment"] == "noise_decomposition") & (sam["condition"] == "box_noise")]
    quality = {
        "moderate_point_hit_rate": float(point_rows["point_hit"].mean()),
        "moderate_box_mean_iou": float(box_rows["box_iou"].mean()),
    }

    cpu_strata = cpu.merge(manifest, on=["sample_id", "class_name"], how="left")
    strata = (
        cpu_strata.groupby(["method", "class_name", "target_area_quartile"], observed=True, as_index=False)[
            ["iou", "dice"]
        ]
        .mean()
        .to_dict("records")
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "primary_hypotheses.csv", rows)
    write_csv(args.output_dir / "cpu_summary.csv", cpu_summary)
    write_csv(args.output_dir / "sam_modality_summary.csv", modality_summary)
    write_csv(args.output_dir / "cpu_strata.csv", strata)
    payload = {
        "confirmatory_samples": len(manifest),
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "seed": args.seed,
        "input_provenance": {
            **execution_provenance,
            "cpu_metrics_sha256": sha256_file(args.cpu_metrics),
            "sam_metrics_sha256": sha256_file(args.sam_metrics),
        },
        "primary_hypotheses": rows,
        "prompt_quality": quality,
        "cpu_summary": cpu_summary,
        "sam_modality_summary": modality_summary,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Confirmatory Results",
        "",
        "All primary effects use paired sample-level deltas. H1--H3 are corrected together with Holm's method.",
        "Oracle selections are descriptive upper bounds and are not primary hypotheses.",
        "",
        "| Hypothesis | Mean IoU delta | 95% CI | raw p | Holm p | Reject |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['hypothesis']}: {row['comparison']} | {row['mean_delta_iou']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['p_raw']:.6f} | "
            f"{row['p_holm']:.6f} | {row['reject_holm_005']} |"
        )
    lines.extend(
        [
            "",
            "## Prompt-quality check",
            "",
            f"- Moderate point hit rate: {quality['moderate_point_hit_rate']:.4f}.",
            f"- Moderate box mean IoU: {quality['moderate_box_mean_iou']:.4f}.",
        ]
    )
    (args.output_dir / "confirmatory_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
