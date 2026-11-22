"""Split patches by source photo (70/15/15 per country), cap TRAIN per class, write splits.csv."""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLASSES = ["D00", "D10", "D20", "D40"]

def split_one_country(df, seed, dev_frac, test_frac):
    g1 = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tv_idx, te_idx = next(g1.split(df, groups=df["photo_id"]))
    tv, te = df.iloc[tv_idx], df.iloc[te_idx]
    g2 = GroupShuffleSplit(n_splits=1, test_size=dev_frac / (1 - test_frac), random_state=seed)
    tr_idx, dv_idx = next(g2.split(tv, groups=tv["photo_id"]))
    return pd.concat([tv.iloc[tr_idx].assign(split="train"),
                      tv.iloc[dv_idx].assign(split="dev"),
                      te.assign(split="test")])

ap = argparse.ArgumentParser()
ap.add_argument("--meta", required=True)
ap.add_argument("--out", default="splits.csv")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dev-frac", type=float, default=0.15)
ap.add_argument("--test-frac", type=float, default=0.15)
ap.add_argument("--cap-per-class", type=int, default=1500)
a = ap.parse_args()

df = pd.read_csv(a.meta)
df = df[df["label"].isin(CLASSES)].copy()
df["photo_id"] = df["country"] + "_" + df["photo_id"].astype(str)
out = pd.concat([split_one_country(g, a.seed, a.dev_frac, a.test_frac)
                 for _, g in df.groupby("country")]).reset_index(drop=True)

if a.cap_per_class > 0:
    train = out[out["split"] == "train"]
    capped = pd.concat([g.sample(min(len(g), a.cap_per_class), random_state=a.seed)
                        for _, g in train.groupby("label")])
    out = pd.concat([capped, out[out["split"] != "train"]]).reset_index(drop=True)

assert (out.groupby("photo_id")["split"].nunique() == 1).all(), "a photo is in two splits"
by_split = pd.crosstab(out["label"], out["split"]).reindex(CLASSES, fill_value=0)
by_country = pd.crosstab(out["label"], [out["split"], out["country"]]).reindex(CLASSES, fill_value=0)
print("Patches per class and split:\n", by_split)
print("\nPatches per class, split and country:\n", by_country)
out.to_csv(a.out, index=False)
stem = str(Path(a.out).with_suffix(""))
by_split.to_csv(stem + "_counts_by_split.csv")
by_country.to_csv(stem + "_counts_by_country.csv")
print("\nwrote", a.out, len(out), "patches")
