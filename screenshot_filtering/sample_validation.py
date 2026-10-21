"""Prepare a fresh whole-clip sample, excluding screenshots in earlier samples.

Run from the project root:
    python -m screenshot_filtering.sample_validation --dry-run
    python -m screenshot_filtering.sample_validation
    python -m screenshot_filtering.sample_validation --mapping-only

Keeps the original largest-fitting-clip selection method. The target is approximate
because clips are never split. Existing output directories are never overwritten.
Also writes screenshot_mapping.xlsx with participant_id, clip_id and screenshot_id
stored as text. Use --mapping-only to export this workbook for an existing sample.
"""

import argparse
import ast
import json
import random
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def identifier(value):
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def key(enrol, filename):
    return identifier(enrol), identifier(Path(str(filename).replace("\\", "/")).stem)


def read_table(path):
    return pd.read_excel(path, dtype={"enrol_number": str})


def write_screenshot_mapping(directory):
    """Export one row per screenshot, preserving the manifests' frame order."""
    output = directory / "screenshot_mapping.xlsx"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    manifests = sorted(directory.glob("*/frames.xlsx"))
    if not manifests:
        raise ValueError(f"No frame manifests found in {directory}")
    rows = []
    for manifest in manifests:
        frames = pd.read_excel(manifest, dtype=str, keep_default_na=False)
        for row in frames.to_dict("records"):
            rows.append({
                "participant_id": str(row["enrol_number"]),
                "clip_id": str(row["clip_id"]),
                "screenshot_id": Path(str(row["filename"]).replace("\\", "/")).stem,
            })
    mapping = pd.DataFrame(rows, columns=["participant_id", "clip_id", "screenshot_id"])
    if mapping.empty or mapping.eq("").any().any():
        raise ValueError("Screenshot mapping contains no rows or has missing IDs")
    if mapping.duplicated(["participant_id", "screenshot_id"]).any():
        raise ValueError("Screenshot mapping contains duplicate participant/screenshot pairs")
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        mapping.to_excel(writer, sheet_name="Screenshots", index=False)
        sheet = writer.sheets["Screenshots"]
        # Set both the stored type and Excel's Text format, including numeric-looking IDs.
        for cells in sheet.iter_rows(min_row=2):
            for cell in cells:
                cell.value = str(cell.value)
                cell.data_type = "s"
                cell.number_format = "@"
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column, width in (("A", 22), ("B", 34), ("C", 22)):
            sheet.column_dimensions[column].width = width
    print(f"Saved {len(mapping)} screenshot associations: {output}", flush=True)


def exclusions(directories):
    used = set()
    for directory in directories:
        manifests = sorted(directory.glob("*/frames.xlsx"))
        if not manifests:
            raise ValueError(f"No frame manifests found in {directory}")
        for path in manifests:
            for row in read_table(path).to_dict("records"):
                enrol = identifier(row.get("enrol_number", path.parent.name))
                if enrol != path.parent.name:
                    raise ValueError(f"Participant mismatch in {path}")
                used.add(key(enrol, row["filename"]))
        # Also exclude any actual images absent from the saved metadata.
        for path in directory.glob("*/frames/*"):
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
                used.add(key(path.parent.parent.name, path.name))
    return used


def local_path(value):
    text = str(value).replace("\\", "/")
    for prefix in ("/workspace/persistent/kidad/", "/workspace/persistent/kidad_validation/"):
        if text.startswith(prefix):
            return ROOT / text[len(prefix):]
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def quotas(counts, target):
    """Equal allocations, repeatedly redistributing exhausted participants' shares."""
    result = {}
    remaining = dict(counts)
    budget = target
    while remaining:
        share = budget / len(remaining)
        small = [p for p, n in remaining.items() if n <= share]
        if not small:
            result.update({p: share for p in remaining})
            break
        for p in small:
            result[p] = remaining[p]
            budget -= remaining.pop(p)
    return result


def select(candidates, allocations, seed):
    rng = random.Random(seed)
    selected = []
    for enrol in sorted(candidates):
        remaining = list(candidates[enrol])
        rng.shuffle(remaining)
        total = 0
        while remaining and total < allocations[enrol]:
            fits = [r for r in remaining if r["n_frames"] <= allocations[enrol] - total]
            row = max(fits, key=lambda r: r["n_frames"]) if fits else min(
                remaining, key=lambda r: r["n_frames"])
            selected.append(row)
            total += row["n_frames"]
            remaining.remove(row)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=6)
    parser.add_argument("--exclude", type=Path, action="append",
                        help="Previous sample directory; repeat for multiple rounds")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation 2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mapping-only", action="store_true",
                        help="Create only the mapping workbook from an existing sample's frame manifests")
    args = parser.parse_args()
    if args.mapping_only:
        if args.dry_run:
            parser.error("--mapping-only cannot be combined with --dry-run")
        write_screenshot_mapping(args.output)
        return
    if args.target <= 0:
        parser.error("--target must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    previous = args.exclude or [ROOT / "data/validation 1"]
    used = exclusions(previous)
    used_clips = set()
    for directory in previous:
        combined = directory / "all_clips.xlsx"
        clip_tables = [combined] if combined.exists() else sorted(directory.glob("*/clips.xlsx"))
        for table in clip_tables:
            for row in read_table(table).to_dict("records"):
                used_clips.add((identifier(row.get("enrol_number", table.parent.name)),
                                str(row["clip_id"])))
    metadata = read_table(ROOT / "data/metadata.xlsx")
    metadata["enrol_number"] = metadata["enrol_number"].map(identifier)
    eligible = set(metadata.loc[
        ~metadata["enrol_number"].str.startswith("32"), "enrol_number"])
    timestamps = {
        (r["enrol_number"], identifier(r["image"])): r["Time"]
        for r in metadata.to_dict("records")
    }
    candidates = {}
    seen = set()
    excluded_clips = 0
    invalid_clips = []
    for path in sorted((ROOT / "data/results_videos").glob("*/frames.xlsx")):
        enrol = path.parent.name
        if enrol not in eligible:
            continue
        rows = []
        clip_ids = set()
        for row in read_table(path).to_dict("records"):
            try:
                paths = ast.literal_eval(row["screenshot_paths"])
            except (SyntaxError, ValueError):
                invalid_clips.append({"participant": enrol, "clip_id": row["clip_id"],
                                      "reason": "Unparseable screenshot list",
                                      "n_frames": int(row["n_frames"])})
                continue
            if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
                raise ValueError(f"Invalid screenshot paths: {path}, {row['clip_id']}")
            n = int(row["n_frames"])
            if n != row["n_frames"] or n != len(paths):
                raise ValueError(f"Frame count mismatch: {path}, {row['clip_id']}")
            keys = [key(enrol, p) for p in paths]
            if len(set(keys)) != n or seen.intersection(keys):
                raise ValueError(f"Repeated screenshot in source clips: {path}, {row['clip_id']}")
            seen.update(keys)
            if row["clip_id"] in clip_ids:
                raise ValueError(f"Duplicate clip ID in {path}")
            clip_ids.add(row["clip_id"])
            if used.intersection(keys) or (enrol, str(row["clip_id"])) in used_clips:
                used.update(keys)
                excluded_clips += 1
                continue
            row.update(enrol_number=enrol, n_frames=n, paths=[local_path(p) for p in paths],
                       video=local_path(row["video_path"]), keys=keys)
            rows.append(row)
        candidates[enrol] = rows
    counts = {p: sum(r["n_frames"] for r in rows) for p, rows in candidates.items()}
    if sum(counts.values()) < args.target:
        raise ValueError(f"Only {sum(counts.values())} eligible screenshots remain")
    selected = select(candidates, quotas(counts, args.target), args.seed)
    new_keys = [k for row in selected for k in row["keys"]]
    if len(set(new_keys)) != len(new_keys) or used.intersection(new_keys):
        raise ValueError("Sample contains duplicate or previously selected screenshots")
    missing = [str(p) for row in selected for p in [row["video"], *row["paths"]]
               if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source files missing; first ten: {missing[:10]}")
    if any(k not in timestamps or pd.isna(timestamps[k]) for k in new_keys):
        raise ValueError("Selected screenshots have missing metadata timestamps")
    summary = {
        "target_screenshots": args.target, "actual_screenshots": len(new_keys),
        "clips": len(selected), "participants": len({r["enrol_number"] for r in selected}),
        "seed": args.seed, "excluded_directories": [str(p.resolve()) for p in previous],
        "previous_screenshots": len(used), "excluded_clips": excluded_clips,
        "eligible_screenshots": sum(counts.values()), "overlap": 0, "duplicate_screenshots": 0,
        "invalid_source_clips_excluded": invalid_clips,
        "method": "Whole clips, equal participant quotas with redistribution; largest fitting clip, seeded tie order",
        "per_participant": {
            p: sum(r["n_frames"] for r in selected if r["enrol_number"] == p)
            for p in sorted(candidates)
        },
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return
    # All source files and sample invariants have passed before creating output.
    args.output.mkdir(parents=True, exist_ok=False)
    all_clips = []
    for enrol in sorted(candidates):
        rows = [r for r in selected if r["enrol_number"] == enrol]
        if not rows:
            continue
        destination = args.output / enrol
        (destination / "videos").mkdir(parents=True)
        (destination / "frames").mkdir()
        frames, clips = [], []
        for row in rows:
            shutil.copy2(row["video"], destination / "videos" / row["video"].name)
            for path, frame_key in zip(row["paths"], row["keys"]):
                shutil.copy2(path, destination / "frames" / path.name)
                frames.append(dict(enrol_number=enrol, clip_id=row["clip_id"],
                                   filename=path.name, timestamp=timestamps[frame_key]))
            clips.append(dict(enrol_number=enrol, clip_id=row["clip_id"],
                              n_frames=row["n_frames"], start_time=row["start_time"],
                              end_time=row["end_time"], video_filename=row["video"].name))
        pd.DataFrame(frames).to_excel(destination / "frames.xlsx", index=False)
        pd.DataFrame(clips).to_excel(destination / "clips.xlsx", index=False)
        all_clips.extend(clips)
        print(f"Copied {enrol}: {len(frames)} screenshots", flush=True)
    pd.DataFrame(all_clips).to_excel(args.output / "all_clips.xlsx", index=False)
    # Verify the saved manifests and physical output before writing the completion report.
    saved_keys = []
    for path in args.output.glob("*/frames.xlsx"):
        for row in read_table(path).to_dict("records"):
            saved_keys.append(key(row["enrol_number"], row["filename"]))
            if not (path.parent / "frames" / row["filename"]).is_file():
                raise RuntimeError(f"Missing exported screenshot: {row['filename']}")
    if set(saved_keys) != set(new_keys) or len(saved_keys) != len(new_keys):
        raise RuntimeError("Saved manifests do not match selection")
    if used.intersection(saved_keys) or len(set(saved_keys)) != len(saved_keys):
        raise RuntimeError("Saved output failed overlap/duplicate checks")
    write_screenshot_mapping(args.output)
    (args.output / "sampling_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Verified sample ready: {args.output}", flush=True)


if __name__ == "__main__":
    main()
