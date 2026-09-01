import os
import ast
import shutil
import json
import pandas as pd

RESULTS_DIR = os.path.join("data/results_videos")
OUTPUT_DIR = os.path.join("data/aggregated")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD ALL RESULTS AND ADS ACROSS PARTICIPANTS
# ============================================================

all_results = []
all_ads = []
all_frames = []
failed_clips = []

for enrol_nr in os.listdir(RESULTS_DIR):
    participant_dir = os.path.join(RESULTS_DIR, enrol_nr)
    if not os.path.isdir(participant_dir):
        continue

    results_path = os.path.join(participant_dir, "results.xlsx")
    ads_path = os.path.join(participant_dir, "ads.xlsx")
    frames_path = os.path.join(participant_dir, "frames.xlsx")

    if not os.path.exists(results_path):
        print(f"WARNING: No results.xlsx for participant {enrol_nr}, skipping.")
        continue

    results = pd.read_excel(results_path)
    results["enrol_number"] = str(enrol_nr)
    all_results.append(results)

    # collect failed clips (UNCERTAIN label)
    failed = results[results["label"] == "UNCERTAIN"].copy()
    if not failed.empty:
        failed["enrol_number"] = str(enrol_nr)
        failed_clips.append(failed)

    if os.path.exists(ads_path):
        ads = pd.read_excel(ads_path)
        ads["enrol_number"] = str(enrol_nr)
        all_ads.append(ads)

    if os.path.exists(frames_path):
        frames = pd.read_excel(frames_path)
        frames["enrol_number"] = str(enrol_nr)
        all_frames.append(frames)

results_all = pd.concat(all_results, ignore_index=True)
ads_all = pd.concat(all_ads, ignore_index=True) if all_ads else pd.DataFrame()
frames_all = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
failed_all = pd.concat(failed_clips, ignore_index=True) if failed_clips else pd.DataFrame()


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
    failed_all[failed_cols].to_excel(os.path.join(OUTPUT_DIR, "failed_clips.xlsx"), index=False)
    failed_summary.to_excel(os.path.join(OUTPUT_DIR, "failed_clips_summary.xlsx"), index=False)
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

if not ads_all.empty and not frames_all.empty:
    # join ads with frames to get video_path
    frames_lookup = frames_all[["clip_id", "enrol_number", "video_path", "n_frames"]].copy()
    ads_round2 = ads_all.merge(
        frames_lookup,
        left_on=["id", "enrol_number"],
        right_on=["clip_id", "enrol_number"],
        how="left"
    )
    ads_round2 = ads_round2.drop(columns=["clip_id"], errors="ignore")

    # drop columns not needed for round 2
    drop_cols = ["n_screenshots", "response_time"]
    ads_round2 = ads_round2.drop(columns=[c for c in drop_cols if c in ads_round2.columns])

    print(f"\nTotal ads found: {len(ads_round2)}")
    print(f"\nPlatform distribution:")
    print(ads_round2["platform"].value_counts().to_string())

    print(f"\nAds per participant:")
    print(ads_round2.groupby("enrol_number").size().sort_values(ascending=False).to_string())

    ads_round2.to_excel(os.path.join(OUTPUT_DIR, "all_ads_round2.xlsx"), index=False)
    print(f"\nAds table saved to {OUTPUT_DIR}/all_ads_round2.xlsx")

else:
    print("No ads found — check that results_videos/ contains ads.xlsx files.")


# ============================================================
# 5. SAVE COMBINED RESULTS
# ============================================================

results_all.to_excel(os.path.join(OUTPUT_DIR, "all_results.xlsx"), index=False)
print(f"\nCombined results saved to {OUTPUT_DIR}/all_results.xlsx")

print("\n" + "="*60)
print("Aggregation complete!")
print("="*60)