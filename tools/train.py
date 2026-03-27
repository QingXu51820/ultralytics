"""Train script with editable in-file hyperparameters.

Edit `TRAIN_ARGS` below to override defaults from `ultralytics/cfg/default.yaml`.
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_PATH, YAML

ROOT = Path(__file__).resolve().parents[1]

# 1) Choose your model checkpoint or YAML (required)
MODEL = "yolo11n.pt"

# 2) Override only the hyperparameters you want to change
#    All unspecified keys will inherit from ultralytics/cfg/default.yaml.
TRAIN_ARGS = {
    "data": "coco8.yaml",
    "epochs": 100,
    "imgsz": 640,
    "batch": 16,
    "project": "runs/train",
    "name": "exp",
    "device": 0,
}


def main() -> None:
    default_args = YAML.load(DEFAULT_CFG_PATH)
    train_args = {**default_args, **TRAIN_ARGS}

    # Optional: ensure local relative paths are resolved from repository root.
    if isinstance(train_args.get("data"), str) and not str(train_args["data"]).startswith(("http://", "https://")):
        data_path = Path(train_args["data"])
        if not data_path.is_absolute() and (ROOT / data_path).exists():
            train_args["data"] = str(ROOT / data_path)

    model = YOLO(MODEL)
    model.train(**train_args)


if __name__ == "__main__":
    main()
