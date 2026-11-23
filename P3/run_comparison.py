"""Run paired P2/P3 experiments without using the final test split."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"D00", "D10", "D20", "D40"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(csv_path, data_root):
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"path", "label", "country", "photo_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Split CSV must contain {sorted(required)}")
        rows = list(reader)
    photos, counts, missing, seen = {}, {}, [], set()
    for row in rows:
        if row["label"] not in CLASSES or row["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid label or split: {row}")
        photo = row["photo_id"]
        previous = photos.setdefault(photo, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Photo leaks between partitions: {photo}")
        if row["path"] in seen:
            raise ValueError(f"Duplicate patch: {row['path']}")
        seen.add(row["path"])
        counts[row["split"]] = counts.get(row["split"], 0) + 1
        # Check train/dev files only; do not load final test images.
        if row["split"] != "test":
            path = (data_root / row["path"]).resolve()
            if not path.is_relative_to(data_root):
                raise ValueError(f"Patch is outside data root: {row['path']}")
            if not path.is_file():
                missing.append(row["path"])
    if missing:
        raise FileNotFoundError(f"{len(missing)} train/dev patches missing under {data_root}: {missing[:3]}")
    for split in ["train", "dev"]:
        if {r["label"] for r in rows if r["split"] == split} != CLASSES:
            raise ValueError(f"{split} must contain all four classes")
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=ROOT / "P1/splits.csv")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "P3/runs")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true", help="skip completed runs with matching provenance")
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    args.csv, args.data_root, args.out = args.csv.resolve(), args.data_root.resolve(), args.out.resolve()
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        ap.error("Seeds must be distinct nonnegative integers")
    if args.epochs < 1 or args.patience < 1 or args.workers < 0:
        ap.error("Epochs/patience must be positive; workers must be nonnegative")
    counts = preflight(args.csv, args.data_root)
    print(f"Preflight passed: {counts}; source photos disjoint across all splits", flush=True)
    if args.check_only:
        return
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for full runs; use the Colab notebook")
    # Keep CPU preprocessing from oversubscribing WSL cores.
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    versions = {p: importlib.metadata.version(p) for p in
                ["torch", "torchvision", "numpy", "pandas", "scikit-learn", "pillow"]}
    manifest = {
        "split_sha256": digest(args.csv),
        "trainer_sha256": digest(ROOT / "P2/train_baseline.py"),
        "data_root": str(args.data_root), "counts": counts,
        "seeds": args.seeds, "epochs": args.epochs, "patience": args.patience,
        "workers": args.workers, "versions": versions,
        "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda,
        "controls": {"batch_size": 32, "head_lr": 0.001, "layer4_lr": 0.0001,
                     "weight_decay": 0.0001, "hidden": 256, "dropout": 0.3},
        "test_evaluated": False,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(f"{args.out} already has a manifest; use --resume or a new output directory")
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run provenance changed; use a new output directory")
    elif any(args.out.iterdir()):
        raise ValueError("Output directory is nonempty without a manifest; choose a new one")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for seed in args.seeds:
        for variant in ["frozen", "layer4"]:
            out = args.out / f"{variant}_seed{seed}"
            outputs = ["best_model.pt", "dev_metrics.json", "dev_logits.csv", "history.csv"]
            if args.resume and all((out / f).is_file() for f in outputs):
                print(f"Already complete: {out.name}", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(ROOT / "P2/train_baseline.py"),
                       "--csv", str(args.csv), "--data-root", str(args.data_root),
                       "--out", str(out), "--seed", str(seed), "--epochs", str(args.epochs),
                       "--patience", str(args.patience), "--workers", str(args.workers)]
            for key, value in manifest["controls"].items():
                command += ["--" + key.replace("_", "-"), str(value)]
            if variant == "layer4":
                command.append("--unfreeze-layer4")
            print(f"Starting {out.name}", flush=True)
            with (out / "train.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True)
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                if process.wait():
                    raise RuntimeError(f"{out.name} failed; see {out / 'train.log'}")
    subprocess.run([sys.executable, str(ROOT / "P3/summarize.py"), "--runs", str(args.out),
                    "--out", str(ROOT / "P3/results")], check=True, env=env)


if __name__ == "__main__":
    main()
