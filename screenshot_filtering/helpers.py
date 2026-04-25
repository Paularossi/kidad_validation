import re
import math
from PIL import Image
import cv2
from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd


# normalize OCR text into a set of tokens
def normalize_ocr_text(text: str) -> set:
    # lowercase, keep words/numbers, drop very short tokens
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 3}

# compute Jaccard similarity between two token sets
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# extract first number found in a string
def extract_num(fn: str) -> int:
    m = re.search(r"(\d+)", fn)
    return int(m.group(1)) if m else 10**18


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
 
    # get output resolution from first image 
    first_img = Image.open(timed[0][0]).convert("RGB")
    w, h = first_img.size
    fourcc = cv2.VideoWriter_fourcc(*video_codec)
 
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
 
        writer = cv2.VideoWriter(out_path, fourcc, playback_fps, (w, h))
 
        for path, _ in batch:
            img = Image.open(path).convert("RGB")
            if img.size != (w, h):
                img = img.resize((w, h), Image.LANCZOS)
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            writer.write(frame)
 
        writer.release()
 
        clips.append({
            "video_path": out_path,
            "clip_index": clip_idx,
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
