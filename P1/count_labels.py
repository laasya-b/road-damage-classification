"""Read every RDD2022 XML for the chosen countries and list all boxes.
Writes P1/boxes.csv and prints counts per class and country."""
import glob, os
import xml.etree.ElementTree as ET
import pandas as pd

ROOT = "/content/rdd/data"
COUNTRIES = ["Japan", "India", "Norway"]
rows = []
for c in COUNTRIES:
    for f in sorted(glob.glob(f"{ROOT}/{c}/train/annotations/xmls/*.xml")):
        r = ET.parse(f).getroot()
        fname = r.findtext("filename") or (os.path.splitext(os.path.basename(f))[0] + ".jpg")
        w = int(float(r.findtext("size/width", "0")))
        h = int(float(r.findtext("size/height", "0")))
        for o in r.findall("object"):
            b = o.find("bndbox")
            rows.append(dict(country=c, photo_id=os.path.basename(f).replace(".xml", ".jpg"),
                             label=(o.findtext("name") or "").strip(),
                             xmin=float(b.findtext("xmin")), ymin=float(b.findtext("ymin")),
                             xmax=float(b.findtext("xmax")), ymax=float(b.findtext("ymax")),
                             img_w=w, img_h=h))
df = pd.DataFrame(rows)
df.to_csv("/content/repo/P1/boxes.csv", index=False)
print("total boxes:", len(df), " photos:", df.groupby("country")["photo_id"].nunique().to_dict())
print("\nALL labels found, by country:")
print(pd.crosstab(df["label"], df["country"], margins=True))
keep = df[df["label"].isin(["D00", "D10", "D20", "D40"])]
print("\nKept 4 classes, by country:")
print(pd.crosstab(keep["label"], keep["country"], margins=True))
