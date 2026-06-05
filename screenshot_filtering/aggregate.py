import pandas as pd
import shutil
import os
import ast

# just to split all the images into individual folders for each participant
# i did this to make it easier to upload data into the DSRI
metadata = pd.read_excel("data/metadata.xlsx")
metadata = metadata[~metadata['enrol_number'].astype(str).str.startswith("32")]
image_folder = "data/All_images"

for enrol_nr in metadata['enrol_number'].unique():
    participant_images = metadata[metadata['enrol_number'] == enrol_nr]['image'].astype(str).tolist()
    participant_images = [f"{img}.png" for img in participant_images]
    
    out_folder = f"data/participants/{enrol_nr}"
    os.makedirs(out_folder, exist_ok=True)
    
    for img in participant_images:
        src = os.path.join(image_folder, img)
        dst = os.path.join(out_folder, img)
        if os.path.exists(src):
            shutil.move(src, dst)
    
    print(f"\n{'='*50}\nMoved images for participant {enrol_nr}\n{'='*50}")


# select only the food ads for round 2 annotation
food_ads = pd.read_excel("data/aggregated/food_ads_round2.xlsx")
base_output = "data/food_ads"
os.makedirs(base_output, exist_ok=True)

missing = []
copied = 0

for _, row in food_ads.iterrows():
    enrol_nr = str(row["enrol_number"])
    paths = ast.literal_eval(row["ad_screenshot_paths"]) if isinstance(row["ad_screenshot_paths"], str) else row["ad_screenshot_paths"]
    
    for p in paths:
        filename = os.path.basename(p)
        # reconstruct local source path
        src = os.path.join("data/participants", enrol_nr, filename)
        # mirror the pod folder structure
        dst_dir = os.path.join(base_output, enrol_nr)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, filename)
        
        if os.path.exists(src):
            if not os.path.exists(dst):  # skip if already copied
                shutil.copy2(src, dst)
                copied += 1
        else:
            missing.append(f"{enrol_nr}/{filename}")

print(f"Copied: {copied} images")
print(f"Missing: {len(missing)} images")
if missing:
    print("First 10 missing:", missing[:10])


# to group results from all participants from first round together for second round annotation
import os
import json
import pandas as pd

RESULTS_DIR = "data/results"
OUTPUT_DIR = "data/aggregated"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD ALL RESULTS AND ADS
# ============================================================

all_results = []
all_ads = []
failed_clips = []

for enrol_nr in os.listdir(RESULTS_DIR):
    participant_dir = os.path.join(RESULTS_DIR, enrol_nr)
    if not os.path.isdir(participant_dir):
        continue

    results_path = os.path.join(participant_dir, "results.xlsx")
    ads_path = os.path.join(participant_dir, "ads.xlsx")

    if not os.path.exists(results_path):
        print(f"WARNING: No results.xlsx for participant {enrol_nr}, skipping.")
        continue

    results = pd.read_excel(results_path)
    results["enrol_number"] = enrol_nr
    all_results.append(results)

    # collect failed clips for this participant
    failed = results[results["label"] == "UNCERTAIN"].copy()
    failed["enrol_number"] = enrol_nr
    if not failed.empty:
        failed_clips.append(failed)

    if os.path.exists(ads_path):
        ads = pd.read_excel(ads_path)
        ads["enrol_number"] = enrol_nr
        all_ads.append(ads)

# combine all participants
results_all = pd.concat(all_results, ignore_index=True)
ads_all = pd.concat(all_ads, ignore_index=True) if all_ads else pd.DataFrame()
failed_all = pd.concat(failed_clips, ignore_index=True) if failed_clips else pd.DataFrame()


# ============================================================
# 2. FAILED CLIPS OVERVIEW
# ============================================================

print("\n" + "="*60)
print("FAILED CLIPS OVERVIEW")
print("="*60)

# per participant
print(f"\nTotal clips processed: {len(results_all)}")
print(f"Total failed clips (UNCERTAIN): {len(failed_all)}")
print(f"Failure rate: {len(failed_all) / len(results_all) * 100:.1f}%\n")

if not failed_all.empty:
    per_participant = failed_all.groupby("enrol_number").size().reset_index(name="n_failed")
    total_per_participant = results_all.groupby("enrol_number").size().reset_index(name="n_total")
    failed_summary = per_participant.merge(total_per_participant, on="enrol_number")
    failed_summary["failure_rate_%"] = (failed_summary["n_failed"] / failed_summary["n_total"] * 100).round(1)
    failed_summary = failed_summary.sort_values("n_failed", ascending=False)
    print("Per participant:")
    print(failed_summary.to_string(index=False))
    
    # save failed clips detail
    failed_cols = ["enrol_number", "id", "label", "confidence", "signals", "n_frames", "start_time", "end_time"]
    failed_all[failed_cols].to_excel(os.path.join(OUTPUT_DIR, "failed_clips.xlsx"), index=False)
    failed_summary.to_excel(os.path.join(OUTPUT_DIR, "failed_clips_summary.xlsx"), index=False)
    print(f"\nFailed clips saved to {OUTPUT_DIR}/failed_clips.xlsx")


# ============================================================
# 3. OVERALL LABEL DISTRIBUTION
# ============================================================

print("\n" + "="*60)
print("LABEL DISTRIBUTION")
print("="*60)

label_counts = results_all["label"].value_counts()
print(f"\nOverall:")
for label, count in label_counts.items():
    print(f"  {label}: {count} ({count/len(results_all)*100:.1f}%)")

print(f"\nPer participant:")
label_dist = results_all.groupby(["enrol_number", "label"]).size().unstack(fill_value=0)
print(label_dist.to_string())


# ============================================================
# 4. COMBINED ADS TABLE FOR ROUND 2 ANNOTATION
# ============================================================

print("\n" + "="*60)
print("ADS TABLE FOR ROUND 2")
print("="*60)

if not ads_all.empty:
    # drop the wrongly flattened columns (always show first ad only)
    # and rename the correctly exploded .1 columns to their proper names
    duplicate_cols = [c for c in ads_all.columns if c.endswith(".1")]
    rename_map = {c: c.replace(".1", "") for c in duplicate_cols}
    original_wrong_cols = list(rename_map.values())
    
    ads_all = ads_all.drop(columns=original_wrong_cols)
    ads_all = ads_all.rename(columns=rename_map)

    # now drop actual duplicate rows (same clip_id + start_frame combination)
    ads_all = ads_all.drop_duplicates(subset=["enrol_number", "id", "start_frame", "end_frame"])

    # take all columns except for response_time and n_screenshots
    keep_cols = [c for c in ads_all.columns if c not in ["response_time", "n_screenshots"]]
    ads_round2 = ads_all[keep_cols].copy()

    print(f"\nTotal ads found: {len(ads_round2)}")
    print(f"Food ads (YES): {(ads_round2['food_ad'] == 'YES').sum()}")
    print(f"Food ads (UNSURE): {(ads_round2['food_ad'] == 'UNSURE').sum()}")
    print(f"Non-food ads: {(ads_round2['food_ad'] == 'NO').sum()}")
    print(f"\nAds per participant:")
    print(ads_round2.groupby("enrol_number").size().sort_values(ascending=False).to_string())

    ads_round2.to_excel(os.path.join(OUTPUT_DIR, "all_ads_round2.xlsx"), index=False)
    print(f"\nCombined ads table saved to {OUTPUT_DIR}/all_ads_round2.xlsx")

    # save only food ads for round 2 annotation
    food_ads = ads_round2[ads_round2["food_ad"].isin(["YES", "UNSURE"])]
    food_ads.to_excel(os.path.join(OUTPUT_DIR, "food_ads_round2.xlsx"), index=False)
    print(f"Food ads for round 2 saved to {OUTPUT_DIR}/food_ads_round2.xlsx")

else:
    print("No ads found across all participants.")


# ============================================================
# 5. SAVE COMBINED RESULTS
# ============================================================

results_all.to_excel(os.path.join(OUTPUT_DIR, "all_results.xlsx"), index=False)
print(f"\nCombined results saved to {OUTPUT_DIR}/all_results.xlsx")

print("\n" + "="*60)
print("Aggregation complete!")
print("="*60)
