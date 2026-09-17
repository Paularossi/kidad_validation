import os
import math
import pandas as pd
from screenshot_filtering.helpers import save_excel, DATA_DIR, RESULTS_DIR, AGGREGATED_DIR, PLAYBACK_FPS

OUTPUT_DIR = AGGREGATED_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 1. LOAD ALL RESULTS AND ADS ACROSS PARTICIPANTS
# ============================================================

all_results = []
all_ads = []
all_clips = []
failed_clips = []
all_frames = []

for enrol_nr in os.listdir(RESULTS_DIR):
    participant_dir = os.path.join(RESULTS_DIR, enrol_nr) # results data
    if not os.path.isdir(participant_dir):
        continue

    source_dir = os.path.join(DATA_DIR, str(enrol_nr)) # original data
    results_path = os.path.join(participant_dir, "results.xlsx")
    ads_path = os.path.join(participant_dir, "ads.xlsx")
    clips_path = os.path.join(source_dir, "clips.xlsx")
    frames_path = os.path.join(source_dir, "frames.xlsx")

    if not os.path.exists(results_path):
        print(f"WARNING: No results.xlsx for participant {enrol_nr}, skipping.")
        continue

    results = pd.read_excel(results_path)
    results["enrol_number"] = str(enrol_nr)
    all_results.append(results)

    if not os.path.isfile(clips_path) or not os.path.isfile(frames_path):
        raise FileNotFoundError(f"Missing prepared clip/frame metadata for {enrol_nr}; run Round 1 first")

    # collect failed clips (UNCERTAIN label)
    failed = results[results["label"] == "UNCERTAIN"].copy()
    if not failed.empty:
        failed["enrol_number"] = str(enrol_nr)
        failed_clips.append(failed)

    if results["n_ads"].fillna(0).gt(0).any() and not os.path.isfile(ads_path):
        raise FileNotFoundError(f"Missing ads.xlsx for {enrol_nr}; rerun Round 1 to rebuild it")

    if os.path.exists(ads_path):
        ads = pd.read_excel(ads_path)
        ads["enrol_number"] = str(enrol_nr)
        all_ads.append(ads)

    if os.path.exists(frames_path):
        frames = pd.read_excel(frames_path)
        frames["enrol_number"] = str(enrol_nr)
        all_frames.append(frames)

    if os.path.exists(clips_path):
        clips = pd.read_excel(clips_path)
        clips["enrol_number"] = str(enrol_nr)
    
        clips["video_path"] = clips["video_filename"].apply(
            lambda filename: os.path.join(
                source_dir,
                "videos",
                os.path.basename(str(filename)),
            )
        )
        all_clips.append(clips)

if not all_results:
    raise SystemExit("No Round 1 results found. Run round1_vid first.")

results_all = pd.concat(all_results, ignore_index=True)
ads_all = pd.concat(all_ads, ignore_index=True) if all_ads else pd.DataFrame()
clips_all = pd.concat(all_clips, ignore_index=True) if all_clips else pd.DataFrame()
frames_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
failed_all = pd.concat(failed_clips, ignore_index=True) if failed_clips else pd.DataFrame()


# sort each clip once (all ads in that clip reuse the same frame sequence)
frames_by_clip = {}
if not frames_all.empty:
    frames_all["enrol_number"] = frames_all["enrol_number"].astype(str)
    frames_all["clip_id"] = frames_all["clip_id"].astype(str)
    for key, group in frames_all.groupby(["enrol_number", "clip_id"], sort=False):
        if "timestamp" in group.columns:
            group = group.sort_values("timestamp", kind="stable")
        frames_by_clip[key] = group.reset_index(drop=True)


def get_ad_screenshot_paths(ad_row):
    enrol_number = str(ad_row["enrol_number"])
    clip_id = str(ad_row["id"])

    clip_frames = frames_by_clip.get((enrol_number, clip_id))
    if clip_frames is None or clip_frames.empty:
        print(f"WARNING: No frames found for {enrol_number}, {clip_id}")
        return []

    start = max(0, int(ad_row["start_frame"]))
    end = min(int(ad_row["end_frame"]), len(clip_frames) - 1)

    if start > end:
        print(f"WARNING: Invalid frame range for {enrol_number}, {clip_id}")
        return []

    paths = []

    for filename in clip_frames.iloc[start : end + 1]["filename"]:
        filename = str(filename)

        if os.path.isabs(filename):
            frame_path = filename
        else:
            frame_path = os.path.join(
                DATA_DIR,
                enrol_number,
                "frames",
                os.path.basename(filename),
            )

        paths.append(frame_path)

    return paths


# ============================================================
# 2. LABEL DISTRIBUTION
# ============================================================

print("\n" + "="*60)
print("LABEL DISTRIBUTION")
print("="*60)

label_counts = results_all["label"].value_counts()
print(f"\nTotal clips processed: {len(results_all)}")
for label, count in label_counts.items():
    print(f"  {label}: {count} ({count/len(results_all)*100:.1f}%)")

print(f"\nPer participant:")
label_dist = results_all.groupby(["enrol_number", "label"]).size().unstack(fill_value=0)
print(label_dist.to_string())


# ============================================================
# 3. FAILED CLIPS OVERVIEW
# ============================================================

print("\n" + "="*60)
print("FAILED CLIPS OVERVIEW")
print("="*60)

print(f"\nTotal failed clips (UNCERTAIN): {len(failed_all)}")
if not failed_all.empty:
    print(f"Failure rate: {len(failed_all) / len(results_all) * 100:.1f}%")

    per_participant = failed_all.groupby("enrol_number").size().reset_index(name="n_failed")
    total_per_participant = results_all.groupby("enrol_number").size().reset_index(name="n_total")
    failed_summary = per_participant.merge(total_per_participant, on="enrol_number")
    failed_summary["failure_rate_%"] = (failed_summary["n_failed"] / failed_summary["n_total"] * 100).round(1)
    failed_summary = failed_summary.sort_values("n_failed", ascending=False)
    print("\nPer participant:")
    print(failed_summary.to_string(index=False))

    failed_cols = ["enrol_number", "id", "label", "confidence", "signals", "n_frames", "start_time", "end_time"]
    save_excel(failed_all[failed_cols], os.path.join(OUTPUT_DIR, "failed_clips.xlsx"))
    save_excel(failed_summary, os.path.join(OUTPUT_DIR, "failed_clips_summary.xlsx"))
    print(f"\nFailed clips saved to {OUTPUT_DIR}/failed_clips.xlsx")


# ============================================================
# 4. BUILD ADS TABLE FOR ROUND 2
# ============================================================
# ads_all already has one row per ad (exploded from the ads JSON list in round1_vid.py)
# each row has: id (clip_id), enrol_number, start_sec, end_sec, platform, label, signals, etc.
# we join with frames_all to get the video_path for each clip

print("\n" + "="*60)
print("BUILDING ADS TABLE FOR ROUND 2")
print("="*60)

if not ads_all.empty and (frames_all.empty or clips_all.empty):
    raise ValueError("Missing clip/frame metadata. Run Round 1 preparation first.")

if not ads_all.empty:
    # join ads with frames to get video_path
    frames_lookup = clips_all[["clip_id", "enrol_number", "video_path", "n_frames"]].copy()
    ads_round2 = ads_all.merge(
        frames_lookup,
        left_on=["id", "enrol_number"],
        right_on=["clip_id", "enrol_number"],
        how="left",
        validate="many_to_one"
    )

    # seconds use [start, end)
    # select frames whose timestamps satisfy start_sec <= index / FPS < end_sec.
    def frame_bounds(row):
        n = int(row["n_frames_y"])
        start_sec, end_sec = float(row["start_sec"]), float(row["end_sec"])
        if not math.isfinite(start_sec) or not math.isfinite(end_sec) or n <= 0 or end_sec < start_sec:
            raise ValueError(f"Invalid ad interval: {row['enrol_number']} {row['id']} ({start_sec}, {end_sec})")
        # equal timestamps are interpreted as a point detection
        if start_sec == end_sec:
            if not 0 <= start_sec < n / PLAYBACK_FPS:
                raise ValueError(f"Point detection outside clip: {row['enrol_number']} {row['id']} ({start_sec})")
            # select the frame displayed at that time
            frame = min(n - 1, math.floor(round(start_sec * PLAYBACK_FPS, 9)))
            return pd.Series({"start_frame": frame, "end_frame": frame})
        start = max(0, math.ceil(round(start_sec * PLAYBACK_FPS, 9)))
        end = min(n, math.ceil(round(end_sec * PLAYBACK_FPS, 9))) - 1
        if start > end:
            raise ValueError(f"Ad interval contains no frames: {row['enrol_number']} {row['id']}")
        return pd.Series({"start_frame": start, "end_frame": end})

    ads_round2[["start_frame", "end_frame"]] = ads_round2.apply(frame_bounds, axis=1)
    ads_round2["frame_bounds_version"] = "exclusive_seconds_v1"
    ads_round2["point_detection"] = ads_round2["start_sec"].eq(ads_round2["end_sec"])
    print(f"Point detections mapped to one frame: {ads_round2['point_detection'].sum()}")

    # drop columns not needed for round 2
    drop_cols = ["clip_id", "n_screenshots", "response_time", "n_frames_y"]
    ads_round2 = ads_round2.drop(columns=[c for c in drop_cols if c in ads_round2.columns])

    ads_round2["ad_screenshot_paths"] = ads_round2.apply(get_ad_screenshot_paths, axis=1)

    print(f"\nTotal ads found: {len(ads_round2)}")
    print(f"\nPlatform distribution:")
    print(ads_round2["platform"].value_counts().to_string())

    print(f"\nAds per participant:")
    print(ads_round2.groupby("enrol_number").size().sort_values(ascending=False).to_string())

    save_excel(ads_round2, os.path.join(OUTPUT_DIR, "all_ads_round2.xlsx"))
    print(f"\nAds table saved to {OUTPUT_DIR}/all_ads_round2.xlsx")

else:
    save_excel(pd.DataFrame(columns=["enrol_number", "id", "start_frame", "end_frame",
                                    "ad_screenshot_paths", "frame_bounds_version"]),
               os.path.join(OUTPUT_DIR, "all_ads_round2.xlsx"))
    print("No ads found.")


# ============================================================
# 5. SAVE COMBINED RESULTS
# ============================================================

save_excel(results_all, os.path.join(OUTPUT_DIR, "all_results.xlsx"))
print(f"\nCombined results saved to {OUTPUT_DIR}/all_results.xlsx")

print("\n" + "="*60)
print("Aggregation complete!")
print("="*60)