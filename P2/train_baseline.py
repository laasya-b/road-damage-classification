"""
P2 baseline: frozen ResNet-50 features + MLP head for RDD2022 patch classification.

This script also runs P3's change (unfreeze layer4) with --unfreeze-layer4,
so both runs share the split, head, augmentation, epochs and seed.

This is still a baseline run and will update once P1 pushes the preliminary inputs

Expected input: a CSV from P1 with these columns
    path      patch file, relative to --data-root (already padded + resized to 224x224)
    label     D00 / D10 / D20 / D40
    country   e.g. Japan, India, Norway
    photo_id  source photo the patch was cropped from
    split     train / dev / test

"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["D00", "D10", "D20", "D40"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# setup

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms(train):
    # No 90-degree rotations: they would turn a longitudinal crack (D00)
    # into a transverse one (D10) and silently corrupt the label.

    #augmentations meant to learn shape of the crack
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PatchDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = Image.open(self.root / row["path"]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[row["label"]], i


# model

def build_model(pretrained, unfreeze_layer4, hidden, dropout):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    for p in model.parameters():
        p.requires_grad = False
    if unfreeze_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True

    # Replace 1000-class layer with our 4-class MLP head (trainable).
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, hidden),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, len(CLASSES)),
    )
    return model


def set_train_mode(model, unfreeze_layer4):
    # Frozen parts stay in eval mode so BatchNorm running stats don't change 
    model.eval()
    model.fc.train()
    if unfreeze_layer4:
        model.layer4.train()


def class_weights(train_df):
    counts = train_df["label"].value_counts().reindex(CLASSES, fill_value=0).values
    if (counts == 0).any():
        raise ValueError(f"A class has no training patches: {dict(zip(CLASSES, counts))}")
    w = counts.sum() / (len(CLASSES) * counts)  # inverse frequency, mean ~1
    return torch.tensor(w, dtype=torch.float32), dict(zip(CLASSES, counts.tolist()))


# train / eval

def run_epoch(model, loader, loss_fn, optimizer, device, unfreeze_layer4):
    set_train_mode(model, unfreeze_layer4)
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels, idx = [], [], []
    for x, y, i in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
        idx.append(i)
    return torch.cat(logits), torch.cat(labels), torch.cat(idx)


def compute_metrics(logits, labels, df):
    y_true = labels.numpy()
    y_pred = logits.argmax(1).numpy()
    rec = recall_score(y_true, y_pred, labels=range(len(CLASSES)), average=None, zero_division=0)
    out = {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=range(len(CLASSES)), zero_division=0)),
        "accuracy": float((y_true == y_pred).mean()),
        "recall_per_class": dict(zip(CLASSES, map(float, rec))),
        "d40_recall": float(rec[CLASS_TO_IDX["D40"]]),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(CLASSES))).tolist(),
        "f1_per_country": {},
    }
    for country in sorted(df["country"].unique()):
        m = (df["country"] == country).values
        out["f1_per_country"][country] = float(
            f1_score(y_true[m], y_pred[m], average="macro", labels=range(len(CLASSES)), zero_division=0)
        )
    return out


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="P2/runs/baseline")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=5, help="early stop on dev macro-F1")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--layer4-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--unfreeze-layer4", action="store_true", help="P3's change run")
    ap.add_argument("--no-pretrained", action="store_true", help="smoke tests only")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    bad = set(df["label"]) - set(CLASSES)
    if bad:
        raise ValueError(f"Unexpected labels in CSV: {bad}")
    train_df = df[df["split"] == "train"]
    dev_df = df[df["split"] == "dev"].reset_index(drop=True)

    leak = set(train_df["photo_id"]) & set(dev_df["photo_id"])
    if leak:
        raise ValueError(f"{len(leak)} photos appear in both train and dev; check P1's split")

    weights, counts = class_weights(train_df)
    print(f"device={device}  train={len(train_df)}  dev={len(dev_df)}  train counts={counts}")

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(PatchDataset(train_df, args.data_root, get_transforms(True)),
                              batch_size=args.batch_size, shuffle=True, generator=g,
                              num_workers=args.workers, pin_memory=device == "cuda")
    dev_loader = DataLoader(PatchDataset(dev_df, args.data_root, get_transforms(False)),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=device == "cuda")

    model = build_model(not args.no_pretrained, args.unfreeze_layer4, args.hidden, args.dropout).to(device)
    groups = [{"params": model.fc.parameters(), "lr": args.head_lr}]
    if args.unfreeze_layer4:
        groups.append({"params": model.layer4.parameters(), "lr": args.layer4_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    best_f1, best_epoch, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, device, args.unfreeze_layer4)
        logits, labels, _ = predict(model, dev_loader, device)
        dev_loss = nn.functional.cross_entropy(logits, labels, weight=weights).item()
        m = compute_metrics(logits, labels, dev_df)
        history.append({"epoch": epoch, "train_loss": train_loss, "dev_loss": dev_loss,
                        "dev_macro_f1": m["macro_f1"], "dev_d40_recall": m["d40_recall"]})
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  dev_loss {dev_loss:.4f}  "
              f"dev_macroF1 {m['macro_f1']:.4f}  D40_recall {m['d40_recall']:.3f}  ({time.time()-t0:.0f}s)")

        if m["macro_f1"] > best_f1:
            best_f1, best_epoch = m["macro_f1"], epoch
            torch.save(model.state_dict(), out / "best_model.pt")
        elif epoch - best_epoch >= args.patience:
            print(f"early stop: no dev improvement for {args.patience} epochs")
            break

    # Final dev  with the best checkpoint.
    model.load_state_dict(torch.load(out / "best_model.pt", map_location=device))
    logits, labels, _ = predict(model, dev_loader, device)
    metrics = compute_metrics(logits, labels, dev_df)
    metrics.update({"best_epoch": best_epoch, "train_counts": counts, "config": vars(args)})

    # Dev logits - P4 needs to fit temperature scaling and review threshold.
    dev_out = dev_df[["path", "label", "country", "photo_id"]].copy()
    dev_out[[f"logit_{c}" for c in CLASSES]] = logits.numpy()
    dev_out["pred"] = [CLASSES[i] for i in logits.argmax(1).tolist()]
    dev_out.to_csv(out / "dev_logits.csv", index=False)

    (out / "dev_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    print(f"\nbest epoch {best_epoch}  dev macro-F1 {metrics['macro_f1']:.4f}  "
          f"D40 recall {metrics['d40_recall']:.3f}\nsaved to {out}/")


if __name__ == "__main__":
    main()
