from contextlib import contextmanager
from functools import lru_cache
import tempfile
import json
from PIL import Image
from datetime import timedelta
import os
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# change only this path to switch datasets (e.g. "data/participants" for full dataset)
DATA_DIR = os.path.join(BASE_DIR, "data/validation 1")
METADATA_PATH = os.path.join(BASE_DIR, "data/metadata.xlsx")
PLAYBACK_FPS = 2.0
WINDOW_MINUTES = 5.0
_dataset = os.path.basename(os.path.normpath(DATA_DIR))

RESULTS_DIR = os.path.join(os.path.dirname(DATA_DIR), "results_validation" if _dataset == "validation 1" else f"results_{_dataset}")
AGGREGATED_DIR = os.path.join(os.path.dirname(DATA_DIR), "aggregated validation" if _dataset == "validation 1" else f"aggregated_{_dataset}")


@lru_cache(maxsize=1)
def screenshot_metadata(path):
    return pd.read_excel(path, dtype={"enrol_number": str, "image": str})


def prepare_participant(source_dir, enrol_number):
    """Reuse clip metadata, or prepare raw screenshots without copying them."""
    clips_path = os.path.join(source_dir, "clips.xlsx")
    frames_path = os.path.join(source_dir, "frames.xlsx")
    if os.path.exists(clips_path) and os.path.exists(frames_path):
        clips = pd.read_excel(clips_path)
        frames = pd.read_excel(frames_path)
    elif os.path.exists(clips_path) or os.path.exists(frames_path):
        raise ValueError(f"Incomplete preparation in {source_dir}: both clips.xlsx and frames.xlsx are required")
    else:
        metadata = screenshot_metadata(METADATA_PATH)
        metadata = metadata.loc[metadata["enrol_number"].eq(str(enrol_number))].copy()
        if metadata.empty:
            raise ValueError(f"No screenshot metadata for participant {enrol_number}")
        metadata["Time"] = pd.to_datetime(metadata["Time"], errors="raise")
        metadata = metadata.sort_values("Time", kind="stable")
        if metadata["image"].duplicated().any() or metadata["Time"].isna().any():
            raise ValueError(f"Duplicate screenshot IDs or missing timestamps for {enrol_number}")
        paths = [os.path.abspath(os.path.join(source_dir, f"{sid}.png")) for sid in metadata["image"]]
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} screenshots for {enrol_number}: {missing[:3]}")
        timestamps = dict(zip(metadata["image"], metadata["Time"]))
        videos = screenshots_to_videos(paths, timestamps, os.path.join(source_dir, "videos"),
                                       window_minutes=WINDOW_MINUTES, playback_fps=PLAYBACK_FPS)
        clips = videos.drop(columns=["screenshot_paths", "video_path"]).copy()
        clips["video_filename"] = videos["video_path"].map(os.path.basename)
        frames = pd.DataFrame([
            {"clip_id": row["clip_id"], "filename": path,
             "timestamp": timestamps[os.path.splitext(os.path.basename(path))[0]]}
            for row in videos.to_dict("records") for path in row["screenshot_paths"]
        ])
        for col in ("start_time", "end_time"):
            clips[col] = clips[col].dt.tz_localize(None)
        frames["timestamp"] = pd.to_datetime(frames["timestamp"]).dt.tz_localize(None)
        save_excel(frames, frames_path)
        save_excel(clips, clips_path)

    clip_columns = {"clip_id", "video_filename", "n_frames", "start_time", "end_time"}
    if not clip_columns.issubset(clips.columns) or not {"clip_id", "filename", "timestamp"}.issubset(frames.columns):
        raise ValueError(f"Invalid clip/frame metadata in {source_dir}")
    if clips.empty or clips["clip_id"].duplicated().any():
        raise ValueError(f"Empty or duplicate clips in {source_dir}")
    if not set(frames["clip_id"]).issubset(set(clips["clip_id"])):
        raise ValueError(f"Frames refer to unknown clips in {source_dir}")
    records = []
    groups = {key: group.sort_values("timestamp", kind="stable") for key, group in frames.groupby("clip_id", sort=False)}
    for row in clips.to_dict("records"):
        group = groups.get(row["clip_id"], frames.iloc[:0])
        paths = [os.path.normpath(os.path.join(source_dir, "frames", str(name))) for name in group["filename"]]
        if len(paths) != row["n_frames"] or not paths or group["timestamp"].isna().any():
            raise ValueError(f"Frame count or timestamps invalid for {row['clip_id']}")
        if any(not os.path.isfile(path) for path in paths):
            raise FileNotFoundError(f"Missing screenshots for {row['clip_id']}")
        video_path = os.path.join(source_dir, "videos", row["video_filename"])
        if row["n_frames"] > 1 and (not os.path.isfile(video_path) or os.path.getsize(video_path) == 0):
            print(f"Recreating missing video: {video_path}")
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            write_video(paths, video_path, PLAYBACK_FPS)
        records.append({**row, "video_path": video_path, "screenshot_paths": paths})
    return records


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


def screenshots_to_videos(meta_images, id_to_ts, output_folder, window_minutes = 10.0, playback_fps = 2.0, video_codec = "mp4v"):
    os.makedirs(output_folder, exist_ok=True)
    ext = ".mp4"

    timed = []
    for p in meta_images:
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid in id_to_ts:
            timed.append((p, id_to_ts[sid]))
    timed.sort(key=lambda x: x[1])
 
    print(f"Loaded {len(timed)} screenshots. Window: {window_minutes} min.")
 
    if not timed:
        raise ValueError("No screenshots with timestamps to prepare")

    # slice into windows 
    window_seconds = window_minutes * 60.0
    t_start = timed[0][1]
    clips = []
    clip_idx = 0
 
    i = 0
    while i < len(timed):
        window_end = t_start + timedelta(seconds=window_seconds)
 
        # collect all frames within [t_start, window_end)
        batch = []
        while i < len(timed) and timed[i][1] < window_end:
            batch.append(timed[i])
            i += 1
 
        if not batch:
            # shouldn't happen, but guard against empty windows
            t_start = window_end
            continue
 
        actual_start = batch[0][1]
        actual_end   = batch[-1][1]
 
        out_name = f"clip_{clip_idx:04d}_{actual_start.strftime('%H%M%S')}_{actual_end.strftime('%H%M%S')}{ext}"
        out_path = os.path.join(output_folder, out_name)
 
        write_video([path for path, _ in batch], out_path, playback_fps, video_codec)

        clips.append({
            "video_path": out_path,
            "clip_id": f"clip_{clip_idx:04d}_{actual_start.strftime('%H%M%S')}_{actual_end.strftime('%H%M%S')}",
            "start_time": actual_start,
            "end_time": actual_end,
            "n_frames": len(batch),
            "screenshot_paths": [p for p, _ in batch],
        })
 
        print(f"  Clip {clip_idx:04d}: {len(batch):3d} frames → {out_name}")
 
        clip_idx += 1
        t_start = window_end
 
    clips_df = pd.DataFrame(clips)
    print(f"\nDone. {len(clips)} clips written to {output_folder}")
 
    return clips_df


def group_screenshots(meta_images, id_to_ts, window_minutes = 10.0):
    """
    Group screenshots into time windows and return lists of PIL images (frames are passed directly to Qwen)
    """
    # build (path, timestamp) list sorted by time
    timed = []
    for p in meta_images:
        sid = os.path.splitext(os.path.basename(p))[0]
        if sid in id_to_ts:
            timed.append((p, pd.to_datetime(id_to_ts[sid])))
    timed.sort(key=lambda x: x[1])

    window_seconds = window_minutes * 60.0
    t_start = timed[0][1]
    windows = []
    i = 0

    while i < len(timed):
        window_end = t_start + timedelta(seconds=window_seconds)
        batch = []
        while i < len(timed) and timed[i][1] < window_end:
            batch.append(timed[i])
            i += 1
        if batch:
            windows.append({
                "clip_id": f"clip_{len(windows):04d}_{batch[0][1].strftime('%H%M%S')}_{batch[-1][1].strftime('%H%M%S')}",
                "start_time": batch[0][1],
                "end_time": batch[-1][1],
                "n_frames": len(batch),
                "frames": [Image.open(p).convert("RGB") for p, _ in batch],  # just PIL images, no timestamps needed
                "screenshot_paths": [p for p, _ in batch],
            })
        t_start = window_end

    return windows


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
