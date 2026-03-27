"""Validation script with editable in-file hyperparameters.

Edit `VAL_ARGS` below to override defaults from `ultralytics/cfg/default.yaml`.
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_PATH, YAML

ROOT = Path(__file__).resolve().parents[1]

# 1) Choose your model checkpoint (required)
MODEL = "yolo11n.pt"

# 2) Override only the hyperparameters you want to change
#    All unspecified keys will inherit from ultralytics/cfg/default.yaml.
VAL_ARGS = {
    "data": "coco8.yaml",
    "imgsz": 640,
    "batch": 16,
    "split": "val",
    "project": "runs/val",
    "name": "exp",
    "device": 0,
}


def main() -> None:
    default_args = YAML.load(DEFAULT_CFG_PATH)
    val_args = {**default_args, **VAL_ARGS}

    if isinstance(val_args.get("data"), str) and not str(val_args["data"]).startswith(("http://", "https://")):
        data_path = Path(val_args["data"])
        if not data_path.is_absolute() and (ROOT / data_path).exists():
            val_args["data"] = str(ROOT / data_path)

    model = YOLO(MODEL)
    model.val(**val_args)


if __name__ == "__main__":
    main()
