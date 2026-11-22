"""Crop one square 224x224 patch per box (10% margin) for the 4 classes.
Reads P1/boxes.csv, writes patches to /content/patches and P1/meta.csv."""
import os
from multiprocessing import Pool
import pandas as pd
from PIL import Image

ROOT = "/content/rdd/data"
OUT = "/content/patches"
CLASSES = {"D00", "D10", "D20", "D40"}
MARGIN, SIZE = 1.10, 224

def crop_photo(args):
    (country, photo_id), g = args
    src = f"{ROOT}/{country}/train/images/{photo_id}"
    if not os.path.exists(src):
        return []
    im = Image.open(src).convert("RGB")
    W, H = im.size
    os.makedirs(f"{OUT}/{country}", exist_ok=True)
    rows = []
    for k, r in enumerate(g.itertuples()):
        bw, bh = r.xmax - r.xmin, r.ymax - r.ymin
        if bw < 2 or bh < 2:
            continue
        side = min(max(bw, bh) * MARGIN, W, H)
        cx, cy = (r.xmin + r.xmax) / 2, (r.ymin + r.ymax) / 2
        x0 = min(max(cx - side / 2, 0), W - side)
        y0 = min(max(cy - side / 2, 0), H - side)
        patch = im.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((SIZE, SIZE), Image.LANCZOS)
        rel = f"{country}/{os.path.splitext(photo_id)[0]}_{k}.jpg"
        patch.save(f"{OUT}/{rel}", quality=95)
        rows.append(dict(path=rel, label=r.label, country=country, photo_id=photo_id))
    return rows

if __name__ == "__main__":
    df = pd.read_csv("/content/repo/P1/boxes.csv")
    df = df[df["label"].isin(CLASSES)]
    groups = list(df.groupby(["country", "photo_id"]))
    print("photos to process:", len(groups), " patches expected:", len(df))
    with Pool(os.cpu_count()) as p:
        results = p.map(crop_photo, groups, chunksize=20)
    meta = pd.DataFrame([r for rs in results for r in rs])
    meta.to_csv("/content/repo/P1/meta.csv", index=False)
    print("patches written:", len(meta))
    print(pd.crosstab(meta["label"], meta["country"], margins=True))
