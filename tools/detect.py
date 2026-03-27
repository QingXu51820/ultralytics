"""Detection script with editable in-file hyperparameters.

Edit `DETECT_ARGS` below to override defaults from `ultralytics/cfg/default.yaml`.
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_PATH, YAML

ROOT = Path(__file__).resolve().parents[1]

# 1) Choose your model checkpoint (required)
MODEL = "yolo11n.pt"

# 2) Override only the hyperparameters you want to change
#    All unspecified keys will inherit from ultralytics/cfg/default.yaml.
DETECT_ARGS = {
    "source": "ultralytics/assets",
    "imgsz": 640,
    "conf": 0.25,
    "iou": 0.7,
    "project": "runs/detect",
    "name": "exp",
    "device": 0,
    "save": True,
}


def main() -> None:
    default_args = YAML.load(DEFAULT_CFG_PATH)
    detect_args = {**default_args, **DETECT_ARGS}

    if isinstance(detect_args.get("source"), str) and not str(detect_args["source"]).startswith(("http://", "https://")):
        source_path = Path(detect_args["source"])
        if not source_path.is_absolute() and (ROOT / source_path).exists():
            detect_args["source"] = str(ROOT / source_path)

    model = YOLO(MODEL)
    model.predict(**detect_args)


if __name__ == "__main__":
    main()
