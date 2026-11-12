"""Generate the Ultralytics dataset config from the canonical class taxonomy."""

from pathlib import Path
from typing import Any

import yaml


def generate_data_yaml(
    classes_path: Path | None = None,
    dataset_dir: Path | None = None,
    output_path: Path | None = None,
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str | None = "images/test",
) -> Path:
    """Write configs/data.yaml and return its path."""
    project_root = Path(__file__).resolve().parents[2]
    classes_path = classes_path or project_root / "configs" / "classes.yaml"
    dataset_dir = dataset_dir or project_root / "data" / "labeled"
    output_path = output_path or project_root / "configs" / "data.yaml"

    taxonomy: Any = yaml.safe_load(classes_path.read_text(encoding="utf-8"))
    classes = taxonomy.get("classes") if isinstance(taxonomy, dict) else None
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{classes_path} must define a non-empty 'classes' list")

    names: dict[int, str] = {}
    for index, item in enumerate(classes):
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"Class at index {index} must have a string 'name'")
        names[index] = item["name"]
    if len(set(names.values())) != len(names):
        raise ValueError("Class names must be unique")

    config = {
        "path": str(dataset_dir.resolve()),
        "train": train_path,
        "val": val_path,
    }
    if test_path is not None:
        config["test"] = test_path
    config["nc"] = len(names)
    config["names"] = names
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


if __name__ == "__main__":
    print(generate_data_yaml())
