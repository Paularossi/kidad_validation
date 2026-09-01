"""
Prepares the validation dataset for dietician annotation.
Run this on the DSRI after aggregate.py.
 
Samples clips per participant such that the total number of screenshots
across all participants is approximately TARGET_SCREENSHOTS.
 
Output per participant (data/validation/<enrol_number>/):
  - videos/         — sampled .mp4 clips
  - frames/         — individual screenshots for those clips
  - clips.xlsx      — metadata for sampled clips (clip_id, n_frames, start_time, end_time)
  - frames.xlsx     — frame-level metadata (clip_id, screenshot filename, timestamp)
 
data/validation/
  - all_clips.xlsx  — combined clips metadata across all participants
"""

import os
import ast
import shutil
import random
import pandas as pd

# load initial metadata
metadata = pd.read_excel("data/metadata.xlsx")
metadata = metadata[~metadata['enrol_number'].astype(str).str.startswith("32")]
screenshots_by_participant = metadata.groupby('enrol_number').size().sort_values(ascending=True)


def normalize_image_id(value):
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


# Fast lookup for screenshot timestamp by (enrol_number, image_id)
metadata_lookup = metadata.copy()
metadata_lookup["enrol_number"] = metadata_lookup["enrol_number"].astype(str)
metadata_lookup["image_id"] = metadata_lookup["image"].apply(normalize_image_id)
timestamp_lookup = metadata_lookup.set_index(["enrol_number", "image_id"])["Time"].to_dict()

# load all results
all_results = pd.read_excel("data/aggregated/all_results.xlsx")
all_results["enrol_number"] = all_results["enrol_number"].astype(str)
all_results.groupby('enrol_number').size().sort_values(ascending=False)


# ── configuration ─────────────────────────────────────────────────────────────
TARGET_SCREENSHOTS = 4000   # change to 10000 for main validation
RANDOM_SEED = 6          # for reproducibility

RESULTS_DIR = "data/results_videos"
PARTICIPANTS_DIR = "data/participants"
VALIDATION_DIR = "data/validation 1" # small = 4000
os.makedirs(VALIDATION_DIR, exist_ok=True)
 
random.seed(RANDOM_SEED)

# ── load participant list ─────────────────────────────────────────────────────
enrol_numbers = [
    d for d in os.listdir(RESULTS_DIR)
    if os.path.isdir(os.path.join(RESULTS_DIR, d))
    and os.path.exists(os.path.join(RESULTS_DIR, d, "frames.xlsx"))
]
n_participants = len(enrol_numbers)
base_quota = TARGET_SCREENSHOTS / n_participants
ordered_enrol_numbers = screenshots_by_participant.index.astype(str).tolist()



# participants with fewer screenshots than the base quota are fully included
under_quota_counts = screenshots_by_participant[screenshots_by_participant < base_quota]
remaining_counts = screenshots_by_participant[screenshots_by_participant >= base_quota]

allocated_under_quota = int(under_quota_counts.sum())
remaining_target = TARGET_SCREENSHOTS - allocated_under_quota

if len(remaining_counts) > 0:
    redistributed_quota = max(0.0, remaining_target / len(remaining_counts))
else:
    redistributed_quota = 0.0
 
print(f"Participants: {n_participants}")
print(f"Target screenshots: {TARGET_SCREENSHOTS}")
print(f"Base quota per participant: {base_quota:.1f} screenshots")
print(f"Participants under base quota: {len(under_quota_counts)}")
print(f"Allocated to under-quota participants: {allocated_under_quota}")
print(f"Redistributed quota for remaining participants: {redistributed_quota:.1f}")
print("=" * 60)
 
# ── sample clips per participant ──────────────────────────────────────────────
all_sampled_clips = []
total_screenshots = 0

# as strings, because enrol_number is sometimes int and sometimes str in the metadata
enrol_nr =  ordered_enrol_numbers[0]

# take participants in sorted order of screenshot counts from metadata
for enrol_nr in ordered_enrol_numbers:
    frames_path = os.path.join(RESULTS_DIR, enrol_nr, "frames.xlsx")
    frames_df = pd.read_excel(frames_path)
 
    # shuffle clips randomly
    frames_df = frames_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
 
    sampled = []
    cumulative = 0
    remaining = frames_df.copy()
 
    while len(remaining) > 0 and cumulative < redistributed_quota:
        headroom = redistributed_quota - cumulative
 
        # prefer clips that fit within remaining headroom
        fits = remaining[remaining["n_frames"] <= headroom]
 
        if not fits.empty:
            # take the largest clip that still fits
            row = fits.loc[fits["n_frames"].idxmax()]
        else:
            # nothing fits cleanly — take the smallest available to minimise overshoot
            row = remaining.loc[remaining["n_frames"].idxmin()]
 
        sampled.append(row)
        cumulative += int(row["n_frames"])
        remaining = remaining.drop(index=row.name)
 
    sampled_df = pd.DataFrame(sampled)
    sampled_df["enrol_number"] = enrol_nr
    all_sampled_clips.append(sampled_df)
    total_screenshots += cumulative
 
    print(f"Participant {enrol_nr}: {len(sampled)} clips, {cumulative} screenshots")
 
print("=" * 60)
print(f"Total clips sampled: {sum(len(df) for df in all_sampled_clips)}")
print(f"Total screenshots: {total_screenshots}")
 
# ── copy files and build metadata ─────────────────────────────────────────────
all_clips_rows = []
missing_videos = []
missing_frames = []

# one to test
sampled_df = all_sampled_clips[3]

for sampled_df in all_sampled_clips:
    enrol_nr = sampled_df["enrol_number"].iloc[0]
    participant_validation_dir = os.path.join(VALIDATION_DIR, enrol_nr)
    videos_out = os.path.join(participant_validation_dir, "videos")
    frames_out = os.path.join(participant_validation_dir, "frames")
    os.makedirs(videos_out, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)
 
    clips_rows = []
    frames_rows = []

    # convert absolute DSRI paths to project-relative paths
    dsri_prefix = "/workspace/persistent/kidad/"
 
    for idx, row in sampled_df.iterrows():
        clip_id = row["clip_id"]
        video_path = str(row["video_path"])
        if video_path.startswith(dsri_prefix):
            video_path = video_path[len(dsri_prefix):]
        sampled_df.at[idx, "video_path"] = video_path
        n_frames = int(row["n_frames"])
        start_time = row["start_time"]
        end_time = row["end_time"]
 
        # parse screenshot paths
        screenshot_paths = row["screenshot_paths"]
        if isinstance(screenshot_paths, str):
            try:
                screenshot_paths = ast.literal_eval(screenshot_paths)
            except Exception:
                screenshot_paths = []

        screenshot_paths = [
            p[len(dsri_prefix):] if isinstance(p, str) and p.startswith(dsri_prefix) else p
            for p in screenshot_paths
        ]
        sampled_df.at[idx, "screenshot_paths"] = str(screenshot_paths)
 
        # copy video
        if os.path.exists(video_path):
            dst_video = os.path.join(videos_out, os.path.basename(video_path))
            if not os.path.exists(dst_video):
                shutil.copy2(video_path, dst_video)
        else:
            missing_videos.append(f"{enrol_nr}/{clip_id}")
 
        # copy frames and build frame rows
        for p in screenshot_paths:
            filename = os.path.basename(p)
            image_id = normalize_image_id(os.path.splitext(filename)[0])
            timestamp = timestamp_lookup.get((str(enrol_nr), image_id))
            src = p  # already full path on pod
            dst = os.path.join(frames_out, filename)
            if os.path.exists(src):
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
            else:
                missing_frames.append(f"{enrol_nr}/{filename}")
 
            frames_rows.append({
                "enrol_number": enrol_nr,
                "clip_id": clip_id,
                "filename": filename,
                "timestamp": timestamp,
            })
 
        clips_rows.append({
            "enrol_number": enrol_nr,
            "clip_id": clip_id,
            "n_frames": n_frames,
            "start_time": start_time,
            "end_time": end_time,
            "video_filename": os.path.basename(video_path),
        })
 
        all_clips_rows.append(clips_rows[-1])
 
    # save per-participant excels
    pd.DataFrame(clips_rows).to_excel(
        os.path.join(participant_validation_dir, "clips.xlsx"), index=False
    )
    pd.DataFrame(frames_rows).to_excel(
        os.path.join(participant_validation_dir, "frames.xlsx"), index=False
    )
 
# save combined clips excel
all_clips_df = pd.DataFrame(all_clips_rows)
all_clips_df["enrol_number"] = all_clips_df["enrol_number"].astype(str)

# merge with all_results to include label, n_ads, ads columns
all_clips_df = all_clips_df.merge(
    all_results[["enrol_number", "id", "label", "n_ads", "ads"]],
    left_on=["enrol_number", "clip_id"],
    right_on=["enrol_number", "id"],
    how="left"
).drop(columns=["id"])

all_clips_df.to_excel(os.path.join(VALIDATION_DIR, "all_clips.xlsx"), index=False)

# summary statistics of the prepared dataset
print(
    f"\nSummary statistics of the prepared dataset:\n"
    f"Total participants: {len(ordered_enrol_numbers)}\n"
    f"Total clips: {len(all_clips_rows)}\n"
    f"Total screenshots: {total_screenshots}\n"
    f"Label distribution: {all_clips_df['label'].value_counts().to_dict()}\n"
    f"Total ads: {all_clips_df['n_ads'].fillna(0).sum()}"
)

# ========== THIS IS FOR AI ANNOTATION OF ALL THE VIDEOS IN ROUND 2 ==========

"""
Prepares the validation dataset for dietician annotation.
Run this on the DSRI after aggregate.py.

Output:
- data/validation/videos/<enrol_number>/  — .mp4 clips where ads were found
- data/validation/ads_for_annotation.xlsx — metadata Excel with video_filename column added
"""

# AGGREGATED_DIR = os.path.join("data/aggregated")
# VALIDATION_DIR = os.path.join("data/validation")
# VIDEOS_DIR = os.path.join(VALIDATION_DIR, "videos")
# ROUND1_DIR = os.path.join("data/results_videos")

# os.makedirs(VALIDATION_DIR, exist_ok=True)
# os.makedirs(VIDEOS_DIR, exist_ok=True)

# # load ads table
# ads_path = os.path.join(AGGREGATED_DIR, "all_ads_round2.xlsx")
# if not os.path.exists(ads_path):
#     print(f"ERROR: {ads_path} not found. Run aggregate.py first.")
#     exit(1)

# ads = pd.read_excel(ads_path)
# print(f"Loaded {len(ads)} ads from {ads['enrol_number'].nunique()} participants.")


# # copy unique video clips organized by participant
# copied = 0
# missing = []

# unique_clips = ads.drop_duplicates(subset=["enrol_number", "id"])
# print(f"Copying {len(unique_clips)} unique video clips...")

# for _, row in unique_clips.iterrows():
#     enrol_nr = str(row["enrol_number"])
#     clip_id = str(row["id"])
#     video_path = os.path.join(ROUND1_DIR, enrol_nr, "videos", f"{clip_id}.mp4")

#     if not os.path.exists(video_path):
#         missing.append(video_path)
#         continue

#     dst_dir = os.path.join(VIDEOS_DIR, enrol_nr)
#     os.makedirs(dst_dir, exist_ok=True)
#     dst = os.path.join(dst_dir, os.path.basename(video_path))

#     if not os.path.exists(dst):
#         shutil.copy2(video_path, dst)
#         copied += 1

# print(f"Copied {copied} video clips.")
# if missing:
#     print(f"Missing video paths: {len(missing)}")
#     print("First 10:", missing[:10])

# # save clean Excel for annotation
# output_excel = os.path.join(VALIDATION_DIR, "ads_for_annotation.xlsx")
# ads.to_excel(output_excel, index=False)
# print(f"\nSaved annotation Excel to {output_excel}")

# print(f"\n{'='*50}")
# print(f"Validation dataset ready in {VALIDATION_DIR}")
# print(f"  Videos: {VIDEOS_DIR}")
# print(f"  Excel:  {output_excel}")
# print(f"{'='*50}")