"""Combine Round 1 results and map inclusive ad intervals to original screenshots."""
import os
import pandas as pd
from screenshot_filtering.helpers import (save_excel, DATA_DIR, RESULTS_DIR, AGGREGATED_DIR,
                                         load_participant, validate_intervals,
                                         inclusive_frame_bounds, FRAME_BOUNDS_VERSION)


def aggregate(data_dir=DATA_DIR, results_dir=RESULTS_DIR, output_dir=AGGREGATED_DIR):
    all_results, all_ads = [], []
    for participant in sorted(os.listdir(results_dir)):
        result_path = os.path.join(results_dir, participant, "results.xlsx")
        if not participant.isdigit() or not os.path.isdir(os.path.dirname(result_path)):
            continue
        if not os.path.isfile(result_path):
            print(f"WARNING: No results for participant {participant}; skipping.")
            continue
        clips = {c["clip_id"]: c for c in load_participant(os.path.join(data_dir, participant))}
        results = pd.read_excel(result_path, dtype={"id": str})
        if results["id"].duplicated().any():
            raise ValueError(f"Duplicate results for participant {participant}")
        missing = set(clips) - set(results["id"])
        if missing:
            print(f"WARNING: {participant} has {len(missing)} clips without Round 1 results.")
        results["enrol_number"] = participant
        for row in results.to_dict("records"):
            clip = clips.get(row["id"])
            if clip is None or row["n_frames"] != clip["n_frames"]:
                raise ValueError(f"Clip metadata mismatch: {participant}/{row['id']}")
            if row["label"] in {"AD", "NON_AD"} and row.get("frame_bounds_version") != FRAME_BOUNDS_VERSION:
                raise ValueError("Results use an older timing convention; rerun Round 1 in fresh results.")
            intervals = validate_intervals(row["ads"], row["label"], clip["n_frames"])
            if len(intervals) != row["n_ads"]:
                raise ValueError(f"Ad count mismatch: {participant}/{row['id']}")
            for interval in intervals:
                start, end = inclusive_frame_bounds(interval["start_sec"], interval["end_sec"], clip["n_frames"])
                ad = {key: value for key, value in row.items()
                      if key not in {"ads", "n_screenshots", "response_time"}}
                ad.update(interval, start_frame=start, end_frame=end,
                          video_path=clip["video_path"], point_detection=start == end,
                          ad_screenshot_paths=clip["screenshot_paths"][start:end + 1])
                all_ads.append(ad)
        all_results.append(results)
    if not all_results:
        raise ValueError("No Round 1 results found. Run round1_vid first.")
    results = pd.concat(all_results, ignore_index=True)
    ads = pd.DataFrame(all_ads) if all_ads else pd.DataFrame(columns=[
        "enrol_number", "id", "start_frame", "end_frame", "ad_screenshot_paths", "frame_bounds_version"])
    attention = results.loc[results["label"].isin(["UNCERTAIN", "UNCLEAR"])].copy()
    attention["processing_status"] = attention["label"].map({"UNCERTAIN": "ERROR", "UNCLEAR": "UNCLEAR"})
    summary = attention.groupby(["enrol_number", "processing_status"]).size().reset_index(name="n_clips")
    os.makedirs(output_dir, exist_ok=True)
    # Always refresh reports, including when there are no failures left.
    for table, name in ((results, "all_results.xlsx"), (ads, "all_ads_round2.xlsx"),
                        (attention, "failed_clips.xlsx"), (summary, "failed_clips_summary.xlsx")):
        save_excel(table, os.path.join(output_dir, name))
    print(f"Aggregated {len(results)} clips and {len(ads)} ad groups.")
    print(results["label"].value_counts().to_string())
    print(f"Requiring attention: {len(attention)} clips (errors and unclear classifications reported separately).")
    return results, ads


if __name__ == "__main__":
    aggregate()
