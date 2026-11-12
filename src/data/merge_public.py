"""Audit Roboflow metadata, merge reviewed exports, and optionally download them."""

import argparse
import csv
import math
import os
import re
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = PROJECT_ROOT / "configs" / "roboflow_sources.yaml"
TAXONOMY_CONFIG = PROJECT_ROOT / "configs" / "classes.yaml"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FOOD_CLASS_HINTS = {
    "bacon",
    "beverage",
    "bean",
    "beans",
    "burger",
    "butter",
    "bread",
    "cereal",
    "cheese",
    "chocolate",
    "cream",
    "curd",
    "dairy",
    "date",
    "dessert",
    "drink",
    "flour",
    "food",
    "fruit",
    "grain",
    "grocery",
    "ham",
    "ingredient",
    "jelly",
    "legume",
    "labneh",
    "juice",
    "milk",
    "meat",
    "mutton",
    "noodle",
    "nut",
    "oil",
    "pasta",
    "paste",
    "pastry",
    "pepper",
    "peppers",
    "fasol",
    "bazlama",
    "patisson",
    "produce",
    "salad",
    "salami",
    "sauce",
    "seafood",
    "sausage",
    "snack",
    "spice",
    "sugar",
    "water",
    "creme",
    "vegetable",
    "wine",
    "yogurt",
    "yoghurt",
}
NON_FOOD_CLASSES = {"bottle", "bottles", "person", "water bottle", "water bottles"}


@dataclass(frozen=True)
class ExportReport:
    total_images: int
    kept_images: int
    instances_per_class: dict[str, int]
    top_unmapped: list[tuple[str, int]]


@dataclass(frozen=True)
class MergeSample:
    image: Path
    source: str
    scene_type: str
    boxes: tuple[tuple[int, float, float, float, float], ...]
    perceptual_hash: int


def normalize_label(label: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", label.casefold())).strip()


def load_taxonomy(path: Path = TAXONOMY_CONFIG) -> tuple[list[str], dict[str, str]]:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = data.get("classes") if isinstance(data, dict) else None
    if not isinstance(classes, list) or not classes:
        raise ValueError(f"{path} must define a non-empty 'classes' list")

    names: list[str] = []
    aliases: dict[str, str] = {}
    for item in classes:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError(f"Every class in {path} must have a string name")
        name = item["name"]
        names.append(name)
        for alias in [name, *item.get("synonyms", [])]:
            key = normalize_label(alias.replace("_", " "))
            previous = aliases.get(key)
            if previous is not None and previous != name:
                raise ValueError(f"Alias {alias!r} maps to both {previous!r} and {name!r}")
            aliases[key] = name
    if len(set(names)) != len(names):
        raise ValueError("Class names must be unique")
    return names, aliases


def map_source_class(label: str, aliases: dict[str, str]) -> str | None:
    normalized = normalize_label(label)
    if normalized in aliases:
        return aliases[normalized]
    if normalized in NON_FOOD_CLASSES:
        return None
    if normalized == "other food" or any(
        word in FOOD_CLASS_HINTS for word in normalized.split()
    ):
        return "other_food"
    return None


def _read_class_names(data_yaml: Path) -> list[str]:
    data: Any = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list) and all(isinstance(name, str) for name in names):
        return names
    if isinstance(names, dict):
        indexed: list[tuple[int, str]] = []
        for key, name in names.items():
            if not isinstance(name, str):
                raise ValueError(f"Invalid class name in {data_yaml}: {name!r}")
            indexed.append((int(key), name))
        indexed.sort()
        if [index for index, _ in indexed] != list(range(len(indexed))):
            raise ValueError(f"Class IDs in {data_yaml} must be contiguous from zero")
        return [name for _, name in indexed]
    raise ValueError(f"{data_yaml} must define class names as a list or indexed mapping")


def _export_image_pairs(export_dir: Path) -> list[tuple[Path, Path]]:
    image_pairs: list[tuple[Path, Path]] = []
    images_dir = export_dir / "images"
    labels_dir = export_dir / "labels"
    if images_dir.is_dir() and labels_dir.is_dir():
        image_pairs.extend(
            (path, labels_dir / path.relative_to(images_dir).with_suffix(".txt"))
            for path in images_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
    else:
        for split in ("train", "valid", "val", "test"):
            split_images = export_dir / split / "images"
            split_labels = export_dir / split / "labels"
            if split_images.is_dir() and split_labels.is_dir():
                image_pairs.extend(
                    (
                        path,
                        split_labels / path.relative_to(split_images).with_suffix(".txt"),
                    )
                    for path in split_images.rglob("*")
                    if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
                )
    return sorted(image_pairs, key=lambda pair: str(pair[0]).casefold())


def _suppress_box_polygon_duplicates(
    boxes: list[tuple[int, float, float, float, float]],
    polygons: list[tuple[int, float, float, float, float]],
    overlap_threshold: float = 0.9,
) -> tuple[list[tuple[int, float, float, float, float]], int]:
    retained: list[tuple[int, float, float, float, float]] = []
    suppressed = 0
    for box in boxes:
        class_id, center_x, center_y, width, height = box
        edges = (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )
        box_area = width * height
        is_duplicate = False
        for polygon in polygons:
            if polygon[0] != class_id:
                continue
            p_center_x, p_center_y, p_width, p_height = polygon[1:]
            p_edges = (
                p_center_x - p_width / 2,
                p_center_y - p_height / 2,
                p_center_x + p_width / 2,
                p_center_y + p_height / 2,
            )
            intersection = max(
                0.0, min(edges[2], p_edges[2]) - max(edges[0], p_edges[0])
            ) * max(0.0, min(edges[3], p_edges[3]) - max(edges[1], p_edges[1]))
            smaller_area = min(box_area, p_width * p_height)
            if (
                smaller_area > 0
                and intersection / smaller_area >= overlap_threshold
            ):
                is_duplicate = True
                suppressed += 1
                break
        if not is_duplicate:
            retained.append(box)
    return retained, suppressed


def inspect_yolo_export(
    export_dir: Path,
    taxonomy_names: list[str],
    aliases: dict[str, str],
) -> ExportReport:
    """Count mapped boxes and images in an already-present YOLOv8 export."""
    data_yaml = export_dir / "data.yaml"
    image_pairs = _export_image_pairs(export_dir)
    if not data_yaml.is_file() or not image_pairs:
        raise ValueError(
            f"{export_dir} must contain data.yaml and images/labels in a supported layout"
        )

    source_names = _read_class_names(data_yaml)
    mapped_ids = {
        index: map_source_class(name, aliases)
        for index, name in enumerate(source_names)
    }
    image_pairs.sort(key=lambda pair: str(pair[0]).casefold())
    instance_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()
    kept_images = 0

    for image_path, label_path in image_pairs:
        retained = False
        if label_path.is_file():
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                fields = line.split()
                if not fields:
                    continue
                try:
                    class_id = int(fields[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid class ID in {label_path}:{line_number}"
                    ) from exc
                if class_id not in mapped_ids:
                    raise ValueError(
                        f"Class ID {class_id} is absent from {data_yaml}"
                    )
                source_name = source_names[class_id]
                target = mapped_ids[class_id]
                if target is None:
                    unmapped_counts[source_name] += 1
                    continue
                if target not in taxonomy_names:
                    raise ValueError(f"Mapped class {target!r} is absent from taxonomy")
                instance_counts[target] += 1
                retained = True
        if retained:
            kept_images += 1

    return ExportReport(
        total_images=len(image_pairs),
        kept_images=kept_images,
        instances_per_class={name: instance_counts[name] for name in taxonomy_names},
        top_unmapped=unmapped_counts.most_common(10),
    )


def _read_merge_samples(
    source: dict[str, Any],
    taxonomy_names: list[str],
    aliases: dict[str, str],
) -> tuple[list[MergeSample], Counter[str], list[tuple[str, str]], int, int]:
    from PIL import Image
    import imagehash

    export_dir = _local_export(source)
    if not (export_dir / "data.yaml").is_file():
        raise ValueError(f"Missing YOLO export for {_source_id(source)}: {export_dir}")
    source_names = _read_class_names(export_dir / "data.yaml")
    target_ids = {name: index for index, name in enumerate(taxonomy_names)}
    mapped_ids = {
        index: map_source_class(name, aliases)
        for index, name in enumerate(source_names)
    }
    samples: list[MergeSample] = []
    dropped_classes: Counter[str] = Counter()
    label_flags: list[tuple[str, str]] = []
    polygon_rows = 0
    duplicate_boxes_suppressed = 0
    image_pairs = _export_image_pairs(export_dir)
    if not image_pairs:
        raise ValueError(f"No images found in {export_dir}")

    for image_path, label_path in image_pairs:
        if not label_path.is_file():
            raise ValueError(f"Missing label file for image: {image_path}")
        box_rows: list[tuple[int, float, float, float, float]] = []
        polygon_boxes: list[tuple[int, float, float, float, float]] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            fields = line.split()
            if not fields:
                continue
            if len(fields) < 5:
                raise ValueError(
                    f"Invalid YOLO annotation row in {label_path}:{line_number}"
                )
            try:
                class_id = int(fields[0])
                coordinates = tuple(float(value) for value in fields[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid YOLO box in {label_path}:{line_number}"
                ) from exc
            if class_id not in mapped_ids:
                raise ValueError(f"Unknown class ID {class_id} in {label_path}")
            is_polygon = len(fields) != 5
            if not is_polygon:
                center_x, center_y, width, height = coordinates
            elif len(fields) >= 7 and len(fields) % 2 == 1:
                if not all(math.isfinite(value) for value in coordinates):
                    raise ValueError(
                        f"Non-finite polygon coordinate in {label_path}:{line_number}"
                    )
                if any(value < -1e-4 or value > 1 + 1e-4 for value in coordinates):
                    raise ValueError(
                        f"Out-of-range polygon coordinate in {label_path}:{line_number}"
                    )
                xs = [min(1.0, max(0.0, value)) for value in coordinates[::2]]
                ys = [min(1.0, max(0.0, value)) for value in coordinates[1::2]]
                left, right = min(xs), max(xs)
                top, bottom = min(ys), max(ys)
                center_x, center_y = (left + right) / 2, (top + bottom) / 2
                width, height = right - left, bottom - top
                polygon_rows += 1
            else:
                raise ValueError(
                    f"Invalid YOLO annotation row in {label_path}:{line_number}"
                )
            target_name = mapped_ids[class_id]
            if target_name is None:
                dropped_classes[source_names[class_id]] += 1
                continue
            if target_name not in target_ids:
                raise ValueError(
                    f"Mapped class {target_name!r} is absent from the taxonomy"
                )
            if not all(
                math.isfinite(value)
                for value in (center_x, center_y, width, height)
            ):
                raise ValueError(f"Out-of-range YOLO box in {label_path}:{line_number}")
            if width < 0 or width > 1 or height < 0 or height > 1:
                raise ValueError(f"Out-of-range YOLO box in {label_path}:{line_number}")
            if width == 0 or height == 0:
                label_flags.append(
                    (
                        image_path.name,
                        f"degenerate box dropped in {source_names[class_id]}",
                    )
                )
                continue
            left, top = center_x - width / 2, center_y - height / 2
            right, bottom = center_x + width / 2, center_y + height / 2
            overflow = max(0.0, -left, -top, right - 1, bottom - 1)
            if overflow > 1e-4:
                raise ValueError(f"Out-of-range YOLO box in {label_path}:{line_number}")
            if overflow:
                left, top = max(0.0, left), max(0.0, top)
                right, bottom = min(1.0, right), min(1.0, bottom)
                center_x, center_y = (left + right) / 2, (top + bottom) / 2
                width, height = right - left, bottom - top
                label_flags.append(
                    (
                        image_path.name,
                        f"rounded box clipped to image bounds in {source_names[class_id]}",
                    )
                )
            target_box = (
                target_ids[target_name],
                center_x,
                center_y,
                width,
                height,
            )
            (polygon_boxes if is_polygon else box_rows).append(target_box)
        retained_boxes, suppressed = _suppress_box_polygon_duplicates(
            box_rows, polygon_boxes
        )
        boxes = polygon_boxes + retained_boxes
        duplicate_boxes_suppressed += suppressed
        if not boxes:
            continue
        try:
            with Image.open(image_path) as image:
                perceptual_hash = int(str(imagehash.phash(image.convert("RGB"))), 16)
        except Exception as exc:
            raise ValueError(f"Unable to hash image {image_path}: {exc}") from exc
        samples.append(
            MergeSample(
                image=image_path,
                source=_source_id(source),
                scene_type=str(source.get("scene_type", "to verify")),
                boxes=tuple(boxes),
                perceptual_hash=perceptual_hash,
            )
        )
    return (
        samples,
        dropped_classes,
        label_flags,
        polygon_rows,
        duplicate_boxes_suppressed,
    )


def _cluster_sample_hashes(
    hashes: list[int], max_hamming_distance: int = 6
) -> list[list[int]]:
    parents = list(range(len(hashes)))
    ranks = [0] * len(hashes)

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if ranks[first_root] < ranks[second_root]:
            first_root, second_root = second_root, first_root
        parents[second_root] = first_root
        if ranks[first_root] == ranks[second_root]:
            ranks[first_root] += 1

    bands: list[dict[int, list[int]]] = [{} for _ in range(8)]
    for index, value in enumerate(hashes):
        candidates: set[int] = set()
        for band_index, band in enumerate(bands):
            part = (value >> (band_index * 8)) & 0xFF
            candidates.update(band.get(part, []))
        for candidate in candidates:
            if (value ^ hashes[candidate]).bit_count() <= max_hamming_distance:
                union(index, candidate)
        for band_index, band in enumerate(bands):
            part = (value >> (band_index * 8)) & 0xFF
            band.setdefault(part, []).append(index)

    clusters: dict[int, list[int]] = {}
    for index in range(len(hashes)):
        clusters.setdefault(find(index), []).append(index)
    return sorted(clusters.values(), key=lambda cluster: min(cluster))


def _render_label_samples(
    names: list[str],
    sample_count: int,
    seed: int,
    suspicious: list[tuple[str, str]],
) -> int:
    import cv2

    image_rows: list[tuple[Path, Path]] = []
    for split in ("train", "val"):
        images_dir = PROJECT_ROOT / "data" / "public" / split / "images"
        labels_dir = PROJECT_ROOT / "data" / "public" / split / "labels"
        image_rows.extend(
            (path, labels_dir / path.relative_to(images_dir).with_suffix(".txt"))
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_SUFFIXES
        )
    if not image_rows:
        return 0
    random.Random(seed).shuffle(image_rows)
    chosen = image_rows[: min(sample_count, len(image_rows))]
    output_dir = PROJECT_ROOT / "runs" / "label_check"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, label_path) in enumerate(chosen, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"OpenCV could not read {image_path}")
        height, width = image.shape[:2]
        for line in label_path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if not fields:
                continue
            class_id = int(fields[0])
            center_x, center_y, box_width, box_height = map(float, fields[1:])
            left = round((center_x - box_width / 2) * width)
            top = round((center_y - box_height / 2) * height)
            right = round((center_x + box_width / 2) * width)
            bottom = round((center_y + box_height / 2) * height)
            color = (
                (37 * (class_id + 1)) % 255,
                (97 * (class_id + 1)) % 255,
                (173 * (class_id + 1)) % 255,
            )
            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(
                image,
                names[class_id],
                (left, max(18, top - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
            area = box_width * box_height
            if area < 0.0001 or area > 0.9:
                suspicious.append((image_path.name, f"class={names[class_id]} area={area:.6f}"))
        output_path = output_dir / f"{index:02d}_{image_path.name}"
        if not cv2.imwrite(str(output_path), image):
            raise ValueError(f"Could not write label-check image {output_path}")
    return len(chosen)


def merge_public_sources(
    sources: list[dict[str, Any]],
    taxonomy_names: list[str],
    aliases: dict[str, str],
    max_images_per_class: int = 2500,
    val_fraction: float = 0.2,
    seed: int = 42,
    label_sample_count: int = 40,
) -> None:
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one")
    if max_images_per_class < 1:
        raise ValueError("max_images_per_class must be positive")
    public_dir = PROJECT_ROOT / "data" / "public"
    if public_dir.exists() and any(public_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output: {public_dir}")
    if METADATA_CSV.exists() and len(METADATA_CSV.read_text(encoding="utf-8").splitlines()) > 1:
        raise ValueError(f"Refusing to overwrite populated metadata file: {METADATA_CSV}")

    samples: list[MergeSample] = []
    dropped_classes: Counter[str] = Counter()
    raw_class_images: Counter[str] = Counter()
    raw_instances: Counter[str] = Counter()
    suspicious: list[tuple[str, str]] = []
    converted_polygon_rows = 0
    duplicate_boxes_suppressed = 0
    for source in sources:
        (
            source_samples,
            dropped,
            clipped,
            polygon_rows,
            suppressed_boxes,
        ) = _read_merge_samples(source, taxonomy_names, aliases)
        samples.extend(source_samples)
        dropped_classes.update(dropped)
        suspicious.extend(clipped)
        converted_polygon_rows += polygon_rows
        duplicate_boxes_suppressed += suppressed_boxes
        for sample in source_samples:
            sample_classes = {box[0] for box in sample.boxes}
            for class_id in sample_classes:
                raw_class_images[taxonomy_names[class_id]] += 1
            for class_id, *_ in sample.boxes:
                raw_instances[taxonomy_names[class_id]] += 1
        print(
            f"{_source_id(source)}: {len(source_samples)} labeled images "
            f"after mapping; dropped classes={dropped.most_common(10) or 'none'}"
        )
    if not samples:
        raise ValueError("No labeled, mapped images found in selected exports")

    clusters = _cluster_sample_hashes(
        [sample.perceptual_hash for sample in samples]
    )
    representatives = [cluster[0] for cluster in clusters]
    dedup_class_images: Counter[str] = Counter()
    dedup_instances: Counter[str] = Counter()
    for index in representatives:
        for class_id in {box[0] for box in samples[index].boxes}:
            dedup_class_images[taxonomy_names[class_id]] += 1
        for class_id, *_ in samples[index].boxes:
            dedup_instances[taxonomy_names[class_id]] += 1

    rng = random.Random(seed)
    rng.shuffle(clusters)
    val_cluster_count = round(len(clusters) * val_fraction)
    val_cluster_ids = {
        cluster[0] for cluster in clusters[:val_cluster_count]
    }
    class_image_counts: Counter[int] = Counter()
    final_instance_counts: Counter[str] = Counter()
    final_class_image_counts: Counter[str] = Counter()
    metadata_rows: list[dict[str, str]] = []
    output_count = 0

    for cluster_number, cluster in enumerate(clusters, start=1):
        sample_index = cluster[0]
        sample = samples[sample_index]
        retained_boxes = tuple(
            box
            for box in sample.boxes
            if class_image_counts[box[0]] < max_images_per_class
        )
        if not retained_boxes:
            continue
        class_ids_in_image = {box[0] for box in retained_boxes}
        class_image_counts.update(class_ids_in_image)
        for class_id in class_ids_in_image:
            final_class_image_counts[taxonomy_names[class_id]] += 1
        for class_id, *_ in retained_boxes:
            final_instance_counts[taxonomy_names[class_id]] += 1

        split = "val" if cluster[0] in val_cluster_ids else "train"
        filename = f"image_{cluster_number:06d}{sample.image.suffix.lower()}"
        images_dir = public_dir / split / "images"
        labels_dir = public_dir / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample.image, images_dir / filename)
        label_path = labels_dir / Path(filename).with_suffix(".txt")
        label_path.write_text(
            "".join(
                f"{class_id} {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}\n"
                for class_id, center_x, center_y, width, height in retained_boxes
            ),
            encoding="utf-8",
        )
        relative_image = Path(split) / "images" / filename
        metadata_rows.append(
            {
                "image": relative_image.as_posix(),
                "source": sample.source,
                "scene_type": sample.scene_type,
                "cluster_id": f"cluster_{cluster_number:06d}",
            }
        )
        output_count += 1

    if output_count == 0:
        raise ValueError("No images remained after deduplication and class caps")
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    with METADATA_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "source", "scene_type", "cluster_id"]
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    from src.data.config_utils import generate_data_yaml

    generate_data_yaml(
        classes_path=TAXONOMY_CONFIG,
        dataset_dir=public_dir,
        output_path=public_dir / "data.yaml",
        train_path="train/images",
        val_path="val/images",
        test_path=None,
    )
    rendered = _render_label_samples(
        taxonomy_names, label_sample_count, seed, suspicious
    )
    label_check_dir = PROJECT_ROOT / "runs" / "label_check"
    with (label_check_dir / "suspicious_labels.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "reason"])
        writer.writerows(suspicious)

    report_path = PROJECT_ROOT / "runs" / "step3_merge_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = [
        f"Seed: {seed}",
        f"Perceptual hash: pHash 64-bit, cluster threshold <= 6 bits",
        f"Input labeled images after mapping: {len(samples)}",
        f"Duplicate images removed: {len(samples) - len(clusters)}",
        f"Unique clusters: {len(clusters)}",
        f"Output images after class cap: {output_count}",
        f"Segmentation polygon rows converted to boxes: {converted_polygon_rows}",
        f"Overlapping duplicate boxes suppressed: {duplicate_boxes_suppressed}",
        f"Train images: {sum(row['image'].startswith('train/') for row in metadata_rows)}",
        f"Val images: {sum(row['image'].startswith('val/') for row in metadata_rows)}",
        f"Images rendered for label check: {rendered}",
        f"Suspicious labels flagged: {len(suspicious)}",
        f"Top dropped classes: {dropped_classes.most_common(10) or 'none'}",
        "Per-class image and instance counts (before dedup / after dedup / final):",
    ]
    for name in taxonomy_names:
        report_lines.append(
            f"  {name}: images {raw_class_images[name]}/{dedup_class_images[name]}/"
            f"{final_class_image_counts[name]}; instances "
            f"{raw_instances[name]}/{dedup_instances[name]}/"
            f"{final_instance_counts[name]}"
        )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines[:14]))
    print(f"Merge report: {report_path}")
    print(f"Public data written only under {public_dir}")


def _load_sources(path: Path) -> list[dict[str, Any]]:
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    sources = data.get("datasets") if isinstance(data, dict) else None
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{path} must define a non-empty 'datasets' list")
    for source in sources:
        if not isinstance(source, dict) or not source.get("workspace") or not source.get(
            "project"
        ):
            raise ValueError(f"Every dataset in {path} needs workspace and project")
    return sources


def _source_id(source: dict[str, Any]) -> str:
    return f"{source['workspace']}/{source['project']}"


def _get_roboflow_version(
    rf: Any, source: dict[str, Any]
) -> tuple[Any, Any, list[str]]:
    project = rf.workspace(source["workspace"]).project(source["project"])
    raw_classes = project.classes
    if isinstance(raw_classes, dict):
        class_names = [str(name) for name in raw_classes.keys()]
    elif isinstance(raw_classes, list):
        class_names = [str(name) for name in raw_classes]
    else:
        raise ValueError(f"Unexpected class metadata for {_source_id(source)}")
    print(f"  source classes ({len(class_names)}): {', '.join(class_names)}")
    version_number = source.get("version")
    available_versions: list[Any] = []
    if version_number is not None:
        try:
            version = project.version(int(version_number))
        except RuntimeError as exc:
            print(f"  requested version unavailable: {exc}")
            available_versions = project.versions()
            print(
                "  available versions: "
                + ", ".join(
                    f"v{item.version} ({item.images} images)"
                    for item in available_versions
                )
            )
            return project, None, class_names
    else:
        available_versions = project.versions()
        print(
            "  available versions (not selected): "
            + ", ".join(
                f"v{item.version} ({item.images} images)"
                for item in available_versions
            )
        )
        version = None
    return project, version, class_names


def _local_export(source: dict[str, Any]) -> Path:
    version = source.get("version")
    suffix = f"v{version}" if version is not None else "unversioned"
    return PROJECT_ROOT / "data" / "raw" / "roboflow" / (
        f"{source['workspace']}__{source['project']}__{suffix}"
    )


def run_dry_run(
    sources: list[dict[str, Any]],
    taxonomy_names: list[str],
    aliases: dict[str, str],
    api_key: str | None,
    local_only: bool = False,
) -> None:
    rf: Any = None
    if api_key and not local_only:
        from roboflow import Roboflow

        rf = Roboflow(api_key=api_key)

    for source in sources:
        print(f"\n{_source_id(source)} (version={source.get('version') or 'to verify'})")
        export_dir = _local_export(source)
        class_names = source.get("observed_classes", [])
        if export_dir.is_dir():
            report = inspect_yolo_export(export_dir, taxonomy_names, aliases)
            print(f"  local export: {export_dir}")
            print(f"  total images: {report.total_images} (local export)")
            print(f"  kept images: {report.kept_images}")
            print(f"  instances per class: {report.instances_per_class}")
            print(f"  top unmapped classes: {report.top_unmapped or 'none'}")
        else:
            listing_images = source.get("listing_images")
            print(
                f"  total images: {listing_images if listing_images is not None else 'unavailable'} "
                "(listing/manifest; unique count unverified)"
            )
            print("  kept images: unavailable (requires annotation audit)")
            print("  instances per class: unavailable (requires annotation audit)")
            print("  top unmapped classes: unavailable (requires annotation audit)")
            if class_names:
                mapping = {
                    name: map_source_class(str(name), aliases) or "drop / review"
                    for name in class_names
                }
                print(f"  source class mapping (names only): {mapping}")

        if rf is None:
            reason = (
                "local-only dry run"
                if local_only
                else "ROBOFLOW_API_KEY is not set"
            )
            if not class_names:
                print(f"  Roboflow class names unavailable ({reason})")
            else:
                print(f"  using recorded class metadata ({reason})")
            if source.get("observed_splits") is not None:
                print(f"  recorded version splits: {source['observed_splits']}")
            if source.get("observed_augmentation") is not None:
                print(
                    "  recorded augmentation: "
                    f"{source['observed_augmentation']}"
                )
            if source.get("version_status"):
                print(f"  version status: {source['version_status']}")
            if source.get("observed_splits") is None and source.get(
                "observed_augmentation"
            ) is None:
                print(f"  version splits/augmentation unavailable ({reason})")
            print(f"  scene_type: {source.get('scene_type', 'to verify')}")
            continue

        try:
            _, version, class_names = _get_roboflow_version(rf, source)
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"  metadata error: {exc}")
            print(f"  scene_type: {source.get('scene_type', 'to verify')}")
            continue

        if version is not None:
            print(
                f"  version listing total: {version.images} "
                "(unique count unverified)"
            )
            print(f"  version splits: {version.splits}")
            print(f"  augmentation: {version.augmentation}")
        else:
            print("  version image counts/splits/augmentation: not selected")
        print(f"  scene_type: {source.get('scene_type', 'to verify')}")
        possible_mapping = {
            name: map_source_class(name, aliases) or "drop / review"
            for name in class_names
        }
        print(f"  source class mapping: {possible_mapping}")


def download_reviewed_sources(
    sources: list[dict[str, Any]],
    rf: Any,
) -> None:
    """Download only after the manifest documents size/licence and the user confirms."""
    unresolved = [
        _source_id(source)
        for source in sources
        if not source.get("version")
        or not source.get("license")
        or str(source["license"]).casefold() == "to verify"
        or source.get("expected_size_gb") is None
    ]
    if unresolved:
        raise ValueError(
            "Download blocked until version, licence, and expected_size_gb are verified "
            "for: " + ", ".join(unresolved)
        )

    total_gb = sum(float(source["expected_size_gb"]) for source in sources)
    if total_gb > 60:
        raise ValueError(f"Expected download size {total_gb:.2f} GB exceeds the 60 GB cap")
    free_gb = shutil.disk_usage(PROJECT_ROOT).free / (1024**3)

    print("Download preflight:")
    for source in sources:
        print(
            f"  source={_source_id(source)} v{source['version']}; "
            f"expected_size_gb={source['expected_size_gb']}; "
            f"license={source['license']}"
        )
    print(f"  total expected size: {total_gb:.2f} GB (public-data cap: 60 GB)")
    print(f"  free disk space: {free_gb:.2f} GB")
    if total_gb > free_gb:
        raise ValueError("Insufficient free disk space for the reviewed downloads")
    confirmation = input("Type DOWNLOAD to approve these exact downloads: ")
    if confirmation != "DOWNLOAD":
        print("Download cancelled; no dataset files were requested.")
        return

    for source in sources:
        _, version, _ = _get_roboflow_version(rf, source)
        version.download(
            "yolov8",
            location=str(_local_export(source)),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--download-reviewed", action="store_true")
    mode.add_argument("--merge-public", action="store_true")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--sources", type=Path, default=SOURCE_CONFIG)
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_CONFIG)
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Limit to workspace/project IDs from the source manifest",
    )
    parser.add_argument("--max-images-per-class", type=int, default=2500)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    sources = _load_sources(args.sources)
    if args.datasets:
        requested = set(args.datasets)
        sources = [source for source in sources if _source_id(source) in requested]
        found = {_source_id(source) for source in sources}
        missing = requested - found
        if missing:
            raise ValueError(f"Unknown dataset IDs: {', '.join(sorted(missing))}")
    taxonomy_names, aliases = load_taxonomy(args.taxonomy)
    print(f"Taxonomy classes: {len(taxonomy_names)}")
    if args.dry_run:
        run_dry_run(
            sources,
            taxonomy_names,
            aliases,
            os.environ.get("ROBOFLOW_API_KEY"),
            local_only=args.local_only,
        )
        return
    if args.merge_public:
        merge_public_sources(
            sources,
            taxonomy_names,
            aliases,
            max_images_per_class=args.max_images_per_class,
            val_fraction=args.val_fraction,
            seed=args.seed,
        )
        return

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Set ROBOFLOW_API_KEY before requesting reviewed downloads")
    from roboflow import Roboflow

    download_reviewed_sources(sources, Roboflow(api_key=api_key))


if __name__ == "__main__":
    main()
