"""
Pack the training set:
1. Read all VRB_*.json under output/train/reviewed/
2. Strip internal fields (_solution / _review / _pipeline_metadata / video_path)
3. Rewrite input_image to relative path images/{task}/{variant}/{diff}/images/{filename}
4. Write to VRBench-Spatial-v2/train/questions/
5. Copy corresponding images to VRBench-Spatial-v2/train/images/
6. Pack into VRBench-Spatial-v2-train.tar.gz
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REVIEWED_DIR = PROJECT_ROOT / "output" / "train" / "reviewed"
OUT_DIR      = PROJECT_ROOT / "VRBench-Spatial-v2" / "train"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"

STRIP_KEYS = {"_solution", "_review", "_pipeline_metadata", "video_path"}


def abs_to_rel(abs_path: str) -> str:
    """
    /...../raw_data/maze/1/hard/images/hard_0058.png
    ->  images/maze/1/hard/images/hard_0058.png
    """
    p = Path(abs_path)
    raw_idx = None
    for i, part in enumerate(p.parts):
        if part == "raw_data":
            raw_idx = i
            break
    if raw_idx is None:
        raise ValueError(f"Cannot find 'raw_data' in path: {abs_path}")
    rel = Path("images").joinpath(*p.parts[raw_idx + 1:])
    return str(rel)


def main():
    qa_files = sorted(REVIEWED_DIR.glob("VRB_*.json"))
    if not qa_files:
        print(f"[ERROR] No VRB_*.json found in {REVIEWED_DIR}", file=sys.stderr)
        sys.exit(1)

    questions_out = OUT_DIR / "questions"
    images_out    = OUT_DIR / "images"
    questions_out.mkdir(parents=True, exist_ok=True)

    missing_images = []
    copied_images  = 0
    written_qa     = 0

    print(f"Processing {len(qa_files)} QA files ...")
    for qa_file in qa_files:
        data = json.load(open(qa_file, encoding="utf-8"))

        # 1. Convert image path
        abs_img = data.get("input_image", "")
        rel_img = abs_to_rel(abs_img)
        data["input_image"] = rel_img

        # 2. Strip internal fields
        for k in STRIP_KEYS:
            data.pop(k, None)

        # 3. Write the cleaned JSON
        out_json = questions_out / qa_file.name
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        written_qa += 1

        # 4. Copy image (only once)
        src_img = RAW_DATA_DIR / Path(*Path(rel_img).parts[1:])  # strip leading "images/"
        dst_img = images_out / Path(*Path(rel_img).parts[1:])
        if not dst_img.exists():
            if src_img.exists():
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img, dst_img)
                copied_images += 1
            else:
                missing_images.append(str(src_img))

    print(f"  QA files written : {written_qa}")
    print(f"  Images copied    : {copied_images}")
    if missing_images:
        print(f"  [WARN] Missing images ({len(missing_images)}):")
        for p in missing_images[:10]:
            print(f"    {p}")
        if len(missing_images) > 10:
            print(f"    ... and {len(missing_images)-10} more")

    # 5. Pack
    tar_path = PROJECT_ROOT / "VRBench-Spatial-v2-train.tar.gz"
    print(f"\nPacking → {tar_path} ...")
    import subprocess
    result = subprocess.run(
        ["tar", "-czf", str(tar_path), "-C", str(OUT_DIR.parent), "train"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[ERROR] tar failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    size_mb = tar_path.stat().st_size / 1024 / 1024
    print(f"Done! {tar_path.name}  ({size_mb:.1f} MB)")
    print(f"\nStructure inside tar:")
    print(f"  train/")
    print(f"    questions/  ({written_qa} JSON files)")
    print(f"    images/     (organized by task/variant/difficulty/images/)")


if __name__ == "__main__":
    main()
