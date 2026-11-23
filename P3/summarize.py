"""Summarize complete paired runs and export dev diagnostics for P4."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

CLASSES = ["D00", "D10", "D20", "D40"]


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(runs, out):
    manifest = json.loads((runs / "manifest.json").read_text())
    seeds = manifest["seeds"]
    metrics, rows, predictions = {}, [], {}
    for seed in seeds:
        for variant in ["frozen", "layer4"]:
            run = runs / f"{variant}_seed{seed}"
            m = json.loads((run / "dev_metrics.json").read_text())
            cfg = m["config"]
            expected = {**manifest["controls"], "seed": seed, "epochs": manifest["epochs"],
                        "patience": manifest["patience"], "unfreeze_layer4": variant == "layer4",
                        "no_pretrained": False}
            if any(cfg.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Unexpected training settings in {run}")
            with (run / "dev_logits.csv").open(newline="") as f:
                pred = list(csv.DictReader(f))
            if len(pred) != manifest["counts"]["dev"]:
                raise ValueError(f"Wrong dev count in {run}")
            identity = [(p["path"], p["label"], p["country"], p["photo_id"]) for p in pred]
            if predictions and identity != next(iter(predictions.values()))[0]:
                raise ValueError("Dev rows differ across runs")
            predictions[variant, seed] = identity, pred
            metrics[variant, seed] = m
            row = {"variant": variant, "seed": seed, "best_epoch": m["best_epoch"],
                   "macro_f1": m["macro_f1"], "accuracy": m["accuracy"], "d40_recall": m["d40_recall"]}
            row.update({f"recall_{c}": m["recall_per_class"][c] for c in CLASSES})
            row.update({f"f1_{c}": v for c, v in m["f1_per_country"].items()})
            rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "comparison.csv", rows)
    paired = [{"seed": s, **{k: metrics["layer4", s][k] - metrics["frozen", s][k]
                             for k in ["macro_f1", "accuracy", "d40_recall"]}} for s in seeds]
    write_csv(out / "paired_deltas.csv", paired)
    summaries = {}
    numeric = [k for k in rows[0] if k not in {"variant", "seed", "best_epoch"}]
    for variant in ["frozen", "layer4"]:
        summaries[variant] = {k: dict(zip(["mean", "sample_std"], mean_std([r[k] for r in rows if r["variant"] == variant])))
                              for k in numeric}
    deltas = {k: dict(zip(["mean", "sample_std"], mean_std([r[k] for r in paired])))
              for k in ["macro_f1", "accuracy", "d40_recall"]}
    selected = max(["frozen", "layer4"], key=lambda v: summaries[v]["macro_f1"]["mean"])
    # Choose the architecture by mean dev F1; use the first prespecified seed for handoff.
    # Do not pick the luckiest seed or evaluate the test split here.
    handoff_seed = seeds[0]
    selected_run = runs / f"{selected}_seed{handoff_seed}"
    selection = {"variant": selected, "seed": handoff_seed,
                 "criterion": "higher mean dev macro-F1 over paired seeds; ties favor frozen",
                 "checkpoint": str((selected_run / "best_model.pt").resolve()),
                 "dev_logits": str((selected_run / "dev_logits.csv").resolve()),
                 "test_evaluated": False}
    summary = {"seeds": seeds, "variants": summaries, "paired_deltas": deltas,
               "selection": selection, "provenance": manifest}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    shutil.copyfile(selected_run / "dev_logits.csv", out / "selected_dev_logits.csv")
    shutil.copyfile(selected_run / "dev_metrics.json", out / "selected_dev_metrics.json")
    errors = []
    for variant in ["frozen", "layer4"]:
        for row in predictions[variant, handoff_seed][1]:
            if row["label"] != row["pred"]:
                errors.append({"variant": variant, "seed": handoff_seed, **row})
    if errors:
        write_csv(out / "dev_errors.csv", errors)
    lines = ["# P3: Layer-4 fine-tuning comparison", "",
             "Completed paired experiments using the unchanged P2 training script. "
             "Only layer4 trainability and its optimizer group differ within each pair.", "",
             "Seeds: " + ", ".join(map(str, seeds)) + ". Same source-photo split for every run; final test images were not evaluated.", "",
             "## Dev results", "", "Values are mean +/- sample standard deviation across seeds (0-1 scale).", "",
             "| Metric | Frozen ResNet-50 | Fine-tuned layer4 |", "|---|---:|---:|"]
    for k in numeric:
        vals = [summaries[v][k] for v in ["frozen", "layer4"]]
        lines.append(f"| {k} | " + " | ".join(f"{v['mean']:.4f} +/- {v['sample_std']:.4f}" for v in vals) + " |")
    lines += ["", "## Paired changes (layer4 minus frozen)", "", "| Metric | Mean change | Sample std |", "|---|---:|---:|"]
    for k, v in deltas.items():
        lines.append(f"| {k} | {v['mean']:+.4f} | {v['sample_std']:.4f} |")
    wins = sum(r["macro_f1"] > 0 for r in paired)
    pothole_deltas = [r["d40_recall"] for r in paired]
    lines += ["", "## Findings", "",
              f"Layer4 improves dev macro-F1 in {wins}/{len(seeds)} paired seeds, by a mean "
              f"{100 * deltas['macro_f1']['mean']:.2f} percentage points. "
              f"Mean D40 recall changes by {100 * deltas['d40_recall']['mean']:+.2f} points; "
              f"the within-seed changes range from {100 * min(pothole_deltas):+.2f} to "
              f"{100 * max(pothole_deltas):+.2f} points, so inspect the per-seed tradeoff.", ""]
    lines += ["", "These seeds measure training variability on one fixed split, not uncertainty over new photos or countries. "
              "A small number of seeds does not establish statistical significance.", "",
              "## P4 handoff", "", f"Selected architecture: **{selected}**, using prespecified seed **{handoff_seed}**. "
              "Selection uses mean dev macro-F1; D40 recall is reported as a separate operational tradeoff.", "",
              "`selected_dev_logits.csv` and `selected_dev_metrics.json` are copied here for P4. "
              "The checkpoint path is recorded in `summary.json`; large checkpoints remain under the ignored `P3/runs/` directory. "
              "P4 must fit temperature and review threshold on dev, freeze choices, then evaluate final test once.", "",
              "## Diagnostics", "", "`comparison.csv` contains every run; `paired_deltas.csv` contains within-seed differences. "
              "`dev_errors.csv` contains misclassified dev patches for both variants at the handoff seed. "
              "Figures show dev learning curves, paired F1, and normalized confusion matrices.", "",
              "![Dev learning curves and paired seeds](learning_curves.png)", "",
              "![Seed-0 normalized confusion matrices](confusion_matrices.png)", ""]
    (out / "report.md").write_text("\n".join(lines))
    plot(runs, out, seeds, metrics)
    if manifest.get("data_root"):
        from review_errors import review
        summary["error_transitions"] = review(runs, out, manifest, selection)
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))
    return summary


def plot(runs, out, seeds, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for variant, color in [("frozen", "#386cb0"), ("layer4", "#e6550d")]:
        for i, seed in enumerate(seeds):
            with (runs / f"{variant}_seed{seed}" / "history.csv").open() as f:
                history = list(csv.DictReader(f))
            axes[0].plot([int(r["epoch"]) for r in history], [float(r["dev_macro_f1"]) for r in history],
                         color=color, alpha=0.7, label=variant if i == 0 else None)
    for seed in seeds:
        axes[1].plot([0, 1], [metrics[v, seed]["macro_f1"] for v in ["frozen", "layer4"]], "o-", label=f"seed {seed}")
    axes[0].set(xlabel="Epoch", ylabel="Dev macro-F1", title="Learning curves")
    axes[1].set(xticks=[0, 1], xticklabels=["Frozen", "Layer4"], ylabel="Best dev macro-F1", title="Paired seeds")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, variant in zip(axes, ["frozen", "layer4"]):
        matrix = metrics[variant, seeds[0]]["confusion_matrix"]
        normalized = [[v / sum(row) if sum(row) else 0 for v in row] for row in matrix]
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{normalized[i][j]:.2f}", ha="center", va="center",
                        color="white" if normalized[i][j] > 0.5 else "black")
        ax.set(xticks=range(4), xticklabels=CLASSES, yticks=range(4), yticklabels=CLASSES,
               xlabel="Predicted", ylabel="True", title=f"{variant}, seed {seeds[0]}")
    fig.colorbar(im, ax=axes, label="Fraction within true class")
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path(__file__).parent / "runs")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results")
    args = ap.parse_args()
    summarize(args.runs, args.out)
