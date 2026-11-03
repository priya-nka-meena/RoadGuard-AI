"""
RoadGuard AI - train our OWN pothole detector and evaluate it.

Run in Google Colab (Runtime > Change runtime type > T4 GPU):
  1. Upload your Kaggle zip ("archive (3).zip") to /content
  2. !pip install ultralytics
  3. !python train_roadguard.py

Dataset: Kaggle "annotated-potholes-dataset" (665 images, Pascal VOC XML)
  - splits.json test list (133 imgs) -> kept untouched for final testing
  - splits.json train list (532 imgs) -> 85% train / 15% validation
"""
import json
import random
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from ultralytics import YOLO

# ---------------- settings ----------------
ZIP_PATH = Path("/content/archive (3).zip")
WORK = Path("/content/work")
RAW = WORK / "potholes_raw"
FROM_SCRATCH = False      # True = random weights (yolov8s.yaml). False = start from yolov8s.pt
EPOCHS = 60
IMGSZ = 640
BATCH = 16
SEED = 42
# ------------------------------------------


def find_raw_dir():
    """Locate the folder that contains splits.json and annotated-images/."""
    for p in RAW.rglob("splits.json"):
        return p.parent
    raise FileNotFoundError("splits.json not found after unzipping")


def convert(xml_names, src_dir, out_img, out_lbl):
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for xml_name in xml_names:
        root = ET.parse(src_dir / xml_name).getroot()
        image_name = root.findtext("filename")
        size = root.find("size")
        if not image_name or size is None:
            continue
        image_path = src_dir / image_name
        if not image_path.exists():
            continue
        w, h = int(size.findtext("width")), int(size.findtext("height"))
        if w <= 0 or h <= 0:
            continue
        lines = []
        for obj in root.findall("object"):
            if obj.findtext("name") != "pothole":
                continue
            b = obj.find("bndbox")
            if b is None:
                continue
            x1, y1 = float(b.findtext("xmin")), float(b.findtext("ymin"))
            x2, y2 = float(b.findtext("xmax")), float(b.findtext("ymax"))
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(
                f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        shutil.copy2(image_path, out_img / image_name)
        (out_lbl / (Path(image_name).stem + ".txt")).write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def main():
    random.seed(SEED)

    # 1. unzip
    if not RAW.exists():
        RAW.mkdir(parents=True)
        with zipfile.ZipFile(ZIP_PATH) as z:
            z.extractall(RAW)
    base = find_raw_dir()
    src = base / "annotated-images"
    splits = json.loads((base / "splits.json").read_text())

    train_all = list(splits["train"])
    test_list = list(splits["test"])
    random.shuffle(train_all)
    n_val = int(0.15 * len(train_all))
    val_list, train_list = train_all[:n_val], train_all[n_val:]

    # 2. build YOLO folders
    ds = WORK / "dataset"
    if ds.exists():
        shutil.rmtree(ds)
    for name, lst in [("train", train_list), ("val", val_list), ("test", test_list)]:
        n_img, n_box = convert(lst, src, ds / name / "images", ds / name / "labels")
        print(f"{name}: {n_img} images, {n_box} boxes")

    # 3. data.yaml (absolute paths, so it works anywhere)
    yaml_path = ds / "data.yaml"
    yaml_path.write_text(
        f"path: {ds}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n  0: pothole\n"
    )

    # 4. TRAIN
    model = YOLO("yolov8s.yaml" if FROM_SCRATCH else "yolov8s.pt")
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=15,
        seed=SEED,
        project=str(WORK / "runs"),
        name="roadguard_train",
        exist_ok=True,
    )

    # 5. EVALUATE best weights on the untouched TEST split
    best = WORK / "runs" / "roadguard_train" / "weights" / "best.pt"
    trained = YOLO(str(best))
    m = trained.val(
        data=str(yaml_path),
        split="test",
        imgsz=IMGSZ,
        project=str(WORK / "runs"),
        name="roadguard_test_eval",
        exist_ok=True,
    )
    p, r = m.box.mp, m.box.mr
    f1 = 2 * p * r / (p + r) if (p + r) else 0
    print("\n===== TEST RESULTS (our own model) =====")
    print(f"Precision : {p:.3f}")
    print(f"Recall    : {r:.3f}")
    print(f"F1        : {f1:.3f}")
    print(f"mAP@50    : {m.box.map50:.3f}")
    print(f"mAP@50-95 : {m.box.map:.3f}")
    print(f"\nWeights   : {best}")
    print(f"Training logs/graphs: {WORK / 'runs' / 'roadguard_train'} (results.csv, results.png)")
    print(f"Test eval plots     : {WORK / 'runs' / 'roadguard_test_eval'}")


if __name__ == "__main__":
    main()
