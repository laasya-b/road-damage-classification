"""Review paired dev predictions and show confident errors by true class."""
import csv
import json
import math
from pathlib import Path

CLASSES = ["D00", "D10", "D20", "D40"]


def review(runs, out, manifest, selection):
    seed = selection["seed"]
    def read(variant):
        with (runs / f"{variant}_seed{seed}" / "dev_logits.csv").open() as f:
            return list(csv.DictReader(f))
    frozen, layer4 = read("frozen"), read("layer4")
    transitions, counts = [], {c: {"corrected": 0, "regressed": 0} for c in CLASSES}
    for a, b in zip(frozen, layer4):
        if a["path"] != b["path"]:
            raise ValueError("Paired dev rows are not aligned")
        before, after = a["label"] == a["pred"], b["label"] == b["pred"]
        status = "corrected" if after and not before else "regressed" if before and not after else "both_correct" if after else "both_wrong"
        if status in counts[a["label"]]:
            counts[a["label"]][status] += 1
        transitions.append({"path": a["path"], "label": a["label"], "country": a["country"],
                            "frozen_pred": a["pred"], "layer4_pred": b["pred"], "status": status})
    with (out / "paired_dev_predictions.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(transitions[0]))
        w.writeheader()
        w.writerows(transitions)
    lines = ["", f"## Error review (seed {seed})", "", "Counts compare layer4 predictions with frozen predictions on the same dev patches.", "",
             "| True class | Wrong -> correct | Correct -> wrong |", "|---|---:|---:|"]
    for c, n in counts.items():
        lines.append(f"| {c} | {n['corrected']} | {n['regressed']} |")
    data_root = Path(manifest["data_root"])
    predictions = layer4 if selection["variant"] == "layer4" else frozen
    for row in predictions:
        logits = [float(row["logit_" + c]) for c in CLASSES]
        largest = max(logits)
        row["raw_confidence"] = 1 / sum(math.exp(x - largest) for x in logits)
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, axes = plt.subplots(4, 3, figsize=(9, 12), layout="constrained")
    for i, c in enumerate(CLASSES):
        errors = sorted((r for r in predictions if r["label"] == c and r["pred"] != c),
                        key=lambda r: (-r["raw_confidence"], r["path"]))[:3]
        for j, ax in enumerate(axes[i]):
            ax.axis("off")
            if j >= len(errors):
                continue
            row = errors[j]
            with Image.open(data_root / row["path"]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"True {c} / predicted {row['pred']}\n{row['country']} | raw score {row['raw_confidence']:.3f}", fontsize=10)
    fig.suptitle(f"Highest-confidence dev errors per class: {selection['variant']}, seed {seed}\nSoftmax scores are uncalibrated; XML labels are treated as ground truth", fontsize=12)
    fig.savefig(out / "dev_error_examples.png", dpi=140)
    plt.close(fig)
    lines += ["", "`dev_error_examples.png` shows the three highest-confidence errors per true class for the selected checkpoint. "
              "Scores are raw, uncalibrated softmax values; these examples are diagnostic selections, not a representative sample. "
              "`paired_dev_predictions.csv` identifies every correction and regression for further inspection.", "",
              "![Highest-confidence dev errors by true class](dev_error_examples.png)", ""]
    with (out / "report.md").open("a") as f:
        f.write("\n".join(lines))
    return counts
