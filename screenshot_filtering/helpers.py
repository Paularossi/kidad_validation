from contextlib import contextmanager
from functools import lru_cache
import ast
import tempfile
import json
import math
from PIL import Image
import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# change only this path to switch datasets (e.g. "data/participants" for full dataset)
DATA_DIR = os.path.join(BASE_DIR, "data/validation 1")
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata.xlsx")
PLAYBACK_FPS = 2.0
FRAME_BOUNDS_VERSION = "inclusive_displayed_frames_v2"
WINDOW_MINUTES = 5.0
MODEL = "Qwen/Qwen3.8-27B"
_dataset = os.path.basename(os.path.normpath(DATA_DIR))

RESULTS_DIR = os.path.join(os.path.dirname(DATA_DIR), "results_validation" if _dataset == "validation 1" else f"results_{_dataset}")
AGGREGATED_DIR = os.path.join(os.path.dirname(DATA_DIR), "aggregated validation" if _dataset == "validation 1" else f"aggregated_{_dataset}")


def inclusive_frame_bounds(start_sec, end_sec, n_frames, fps=PLAYBACK_FPS):
    """Both endpoints identify included displayed frames on the playback timeline.

    Frame i is displayed during [i/fps, (i+1)/fps). Exact frame timestamps map
    to themselves; fractional timestamps inside a frame map to that frame.
    Video duration is outside the valid frame range, not an inclusive endpoint.
    """
    start_sec, end_sec = float(start_sec), float(end_sec)
    if (not math.isfinite(start_sec) or not math.isfinite(end_sec)
            or not math.isfinite(fps) or fps <= 0
            or isinstance(n_frames, bool) or int(n_frames) != n_frames or n_frames <= 0):
        raise ValueError("Invalid timestamps, frame count or FPS")
    start_position, end_position = round(start_sec * fps, 9), round(end_sec * fps, 9)
    if not 0 <= start_sec <= end_sec or not 0 <= start_position <= end_position < n_frames:
        raise ValueError(f"Interval {start_sec}..{end_sec}s outside {n_frames} frames at {fps} FPS")
    return math.floor(start_position), math.floor(end_position)


@lru_cache(maxsize=1)
def screenshot_metadata(path):
    return pd.read_excel(path, dtype={"enrol_number": str, "image": str})


def read_frame_manifest(source_dir):
    """Canonical frame order shared by inference, aggregation and similarity."""
    frames = pd.read_excel(os.path.join(source_dir, "frames.xlsx"),
                           dtype={"clip_id": str, "filename": str})
    if not {"clip_id", "filename", "timestamp"}.issubset(frames.columns):
        raise ValueError(f"Invalid frame manifest in {source_dir}")
    if frames[["clip_id", "filename", "timestamp"]].isna().any().any():
        raise ValueError(f"Missing frame metadata in {source_dir}")
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], format="mixed", utc=True).dt.tz_localize(None)
    if frames["timestamp"].isna().any():
        raise ValueError(f"Missing timestamps in {source_dir}")
    frames["screenshot_path"] = [resolve_frame_path(source_dir, name) for name in frames["filename"]]
    if frames["screenshot_path"].duplicated().any():
        raise ValueError(f"Duplicate screenshots in {source_dir}")
    return frames.sort_values("timestamp", kind="stable").reset_index(drop=True)


def resolve_frame_path(source_dir, filename):
    """Resolve prepared or raw images locally, including manifests moved from DSRI."""
    filename = str(filename).replace("\\", "/")
    name = filename.rsplit("/", 1)[-1]
    candidates = [os.path.join(source_dir, "frames", name), os.path.join(source_dir, name),
                  os.path.join(source_dir, filename)]
    for path in candidates:
        path = os.path.abspath(os.path.normpath(path))
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Missing screenshot {filename} in {source_dir}")


def load_participant(source_dir):
    """Read prepared media only; never create manifests or MP4 files."""
    if not all(os.path.isfile(os.path.join(source_dir, name)) for name in ("clips.xlsx", "frames.xlsx")):
        raise FileNotFoundError(f"Missing manifests in {source_dir}; run python -m screenshot_filtering.prepare first")
    clips = pd.read_excel(os.path.join(source_dir, "clips.xlsx"), dtype={"clip_id": str})
    required = {"clip_id", "video_filename", "n_frames", "start_time", "end_time"}
    if not required.issubset(clips.columns) or clips.empty or clips["clip_id"].isna().any() or clips["clip_id"].duplicated().any():
        raise ValueError(f"Invalid or duplicate clips in {source_dir}")
    frames = read_frame_manifest(source_dir)
    if not set(frames["clip_id"]).issubset(set(clips["clip_id"])):
        raise ValueError(f"Frames refer to unknown clips in {source_dir}")
    groups = {key: group for key, group in frames.groupby("clip_id", sort=False)}
    records = []
    for row in clips.to_dict("records"):
        group = groups.get(row["clip_id"], frames.iloc[:0])
        if group.empty or len(group) != row["n_frames"]:
            raise ValueError(f"Frame count invalid for {row['clip_id']}")
        row["n_frames"] = len(group)
        video_name = str(row["video_filename"]).replace("\\", "/").rsplit("/", 1)[-1]
        records.append({**row, "video_path": os.path.join(source_dir, "videos", video_name),
                        "screenshot_paths": group["screenshot_path"].tolist()})
    return records


def prepare_participant(source_dir, enrol_number, videos=False):
    """Create missing manifests; optionally create missing or empty MP4s."""
    manifests = [os.path.join(source_dir, name) for name in ("clips.xlsx", "frames.xlsx")]
    if not any(os.path.exists(path) for path in manifests):
        metadata = screenshot_metadata(METADATA_PATH)
        metadata = metadata.loc[metadata["enrol_number"].eq(str(enrol_number))].copy()
        metadata["Time"] = pd.to_datetime(metadata["Time"], format="mixed", utc=True).dt.tz_localize(None)
        metadata = metadata.sort_values("Time", kind="stable")
        if metadata.empty or metadata[["image", "Time"]].isna().any().any() or metadata["image"].duplicated().any():
            raise ValueError(f"Missing or duplicate screenshot metadata for {enrol_number}")
        metadata["filename"] = metadata["image"] + ".png"
        for name in metadata["filename"]:
            resolve_frame_path(source_dir, name)
        # keep the original fixed five-minute windows, skipping empty windows directly.
        windows = (metadata["Time"] - metadata["Time"].iloc[0]).dt.total_seconds() // (WINDOW_MINUTES * 60)
        clips, frames = [], []
        for index, (_, group) in enumerate(metadata.groupby(windows, sort=True)):
            start, end = group["Time"].iloc[0], group["Time"].iloc[-1]
            clip_id = f"clip_{index:04d}_{start:%H%M%S}_{end:%H%M%S}"
            clips.append(dict(clip_id=clip_id, n_frames=len(group), start_time=start,
                              end_time=end, video_filename=clip_id + ".mp4"))
            frames.extend(dict(clip_id=clip_id, filename=r.filename, timestamp=r.Time)
                          for r in group.itertuples())
        save_excel(pd.DataFrame(frames), manifests[1])
        save_excel(pd.DataFrame(clips), manifests[0])
    media = load_participant(source_dir)
    if videos:
        for clip in media:
            path = clip["video_path"]
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                write_video(clip["screenshot_paths"], path, PLAYBACK_FPS)
    return media


def validate_intervals(ads, label, n_frames):
    """Reject conflicting groups rather than silently merging or dropping them."""
    ads = ast.literal_eval(ads) if isinstance(ads, str) else ads
    if label not in {"AD", "NON_AD", "UNCLEAR", "UNCERTAIN"}:
        raise ValueError(f"Invalid clip classification: {label}")
    if not isinstance(ads, list) or (label == "AD") != bool(ads):
        raise ValueError("AD requires intervals; other labels must have none")
    bounds = []
    for ad in ads:
        if not isinstance(ad, dict) or not {"start_sec", "end_sec"}.issubset(ad):
            raise ValueError("Ad interval must contain start_sec and end_sec")
        bounds.append(inclusive_frame_bounds(ad["start_sec"], ad["end_sec"], n_frames))
    ordered = sorted(bounds)
    if any(start <= previous_end for (_, previous_end), (start, _) in zip(ordered, ordered[1:])):
        raise ValueError("Duplicate or overlapping ad intervals; review/retry this clip")
    return ads


def load_annotation_model():
    """Load GPU dependencies and credentials only when inference is requested."""
    from huggingface_hub import login
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    with open(os.path.join(BASE_DIR, "keys.txt"), encoding="utf-8") as stream:
        login(json.load(stream)["huggingface"])
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL, device_map="cuda:0", dtype="auto", attn_implementation="sdpa").eval()
    return model, AutoProcessor.from_pretrained(MODEL)


def write_video(paths, output_path, playback_fps, video_codec="mp4v"):
    import cv2
    import numpy as np

    with Image.open(paths[0]) as first:
        size = first.size
    with atomic_output(output_path) as temporary:
        writer = cv2.VideoWriter(temporary, cv2.VideoWriter_fourcc(*video_codec), playback_fps, size)
        try:
            if not writer.isOpened():
                raise RuntimeError(f"Could not create video: {output_path}")
            for path in paths:
                with Image.open(path) as source:
                    img = source.convert("RGB")
                if img.size != size:
                    img = img.resize(size, Image.LANCZOS)
                writer.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        if os.path.getsize(temporary) == 0:
            raise RuntimeError(f"Empty video: {output_path}")


@contextmanager
def atomic_output(path):
    """Replace a file only after its temporary copy has been written successfully."""
    path = os.path.abspath(os.fspath(path))
    fd, temporary = tempfile.mkstemp(dir=os.path.dirname(path), suffix=os.path.splitext(path)[1])
    os.close(fd)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_excel(data, path):
    with atomic_output(path) as temporary:
        data.to_excel(temporary, index=False)


def save_json(data, path):
    with atomic_output(path) as temporary:
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=4)
