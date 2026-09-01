import json, os, time
import pandas as pd
import argparse

from screenshot_filtering.questions import *
from screenshot_filtering.helpers import *

import torch
from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from screenshot_filtering.round1_vid import qwen_call_video, sec_to_frame, PLAYBACK_FPS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "data/results_videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "data/aggregated")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL = "Qwen/Qwen3-VL-32B-Instruct"

with open(os.path.join(BASE_DIR, '..', 'keys.txt')) as f:
    json_data = json.load(f)
hugg_key = json_data["huggingface"]


def rerun_failed_clips(model, processor, failed_clips_path):
    """
    Loads failed_clips.xlsx, finds the corresponding video paths from frames.xlsx,
    reruns inference, and updates results.xlsx and ads.xlsx for each participant.
    """
    failed = pd.read_excel(failed_clips_path)
    print(f"Loaded {len(failed)} failed clips across {failed['enrol_number'].nunique()} participants.")

    for enrol_nr, group in failed.groupby("enrol_number"):
        enrol_nr = str(enrol_nr)
        participant_dir = os.path.join(RESULTS_DIR, enrol_nr)
        results_path = os.path.join(participant_dir, "results.xlsx")
        ads_path = os.path.join(participant_dir, "ads.xlsx")
        frames_path = os.path.join(participant_dir, "frames.xlsx")

        if not os.path.exists(frames_path):
            print(f"WARNING: No frames.xlsx for participant {enrol_nr}, skipping.")
            continue
        if not os.path.exists(results_path):
            print(f"WARNING: No results.xlsx for participant {enrol_nr}, skipping.")
            continue

        frames_df = pd.read_excel(frames_path)
        results_df = pd.read_excel(results_path)

        # build lookup: clip_id -> clip metadata dict
        clip_lookup = {
            row["clip_id"]: {
                "clip_id": row["clip_id"],
                "video_path": row["video_path"],
                "n_frames": row["n_frames"],
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "screenshot_paths": row["screenshot_paths"] if isinstance(row["screenshot_paths"], list)
                    else eval(row["screenshot_paths"]) if isinstance(row["screenshot_paths"], str) else [],
            }
            for _, row in frames_df.iterrows()
        }

        clip_ids_to_rerun = group["id"].tolist()
        print(f"\n{'='*50}\nParticipant {enrol_nr}: rerunning {len(clip_ids_to_rerun)} clips\n{'='*50}")

        new_results = []
        new_ads = []

        for clip_id in clip_ids_to_rerun:
            if clip_id not in clip_lookup:
                print(f"WARNING: clip_id {clip_id} not found in frames.xlsx, skipping.")
                continue

            med = clip_lookup[clip_id]
            print(f"Rerunning clip: {clip_id}")

            try:
                response, response_time = qwen_call_video(model, processor, med)

                if isinstance(response, str):
                    raise ValueError(f"JSON parse failed: {response[:200]}")

                item = response["items"][0]
                signals = item.get("signals", [])
                label = item["label"]
                confidence = item["confidence"]

                result_entry = {
                    "id": med["clip_id"],
                    "label": label,
                    "confidence": confidence,
                    "signals": signals,
                    "response_time": round(response_time, 2),
                    "n_frames": med["n_frames"],
                    "start_time": med["start_time"],
                    "end_time": med["end_time"],
                    "n_screenshots": len(med["screenshot_paths"]),
                    "platform": item.get("platform", "UNKNOWN"),
                    "n_ads": len(item.get("ads", [])),
                    "ads": item.get("ads", []),
                }
                new_results.append(result_entry)

                # build ads rows if any ads found
                for ad in item.get("ads", []):
                    ad_row = result_entry.copy()
                    ad_row.pop("ads")
                    ad_row["start_sec"] = ad.get("start_sec", 0)
                    ad_row["end_sec"] = ad.get("end_sec", 0)
                    new_ads.append(ad_row)

            except Exception as e:
                import traceback
                print(f"Error rerunning clip {clip_id}: {e}")
                print(traceback.format_exc())
                new_results.append({
                    "id": clip_id,
                    "label": "UNCERTAIN",
                    "confidence": 0.0,
                    "signals": [f"Error: {str(e)}"],
                    "response_time": None,
                    "n_frames": med["n_frames"],
                    "start_time": med["start_time"],
                    "end_time": med["end_time"],
                    "n_screenshots": len(med["screenshot_paths"]),
                    "platform": None,
                    "n_ads": 0,
                    "ads": [],
                })

            time.sleep(1)

        if not new_results:
            print(f"No results to update for participant {enrol_nr}.")
            continue

        new_results_df = pd.DataFrame(new_results)

        # replace UNCERTAIN rows in results.xlsx with new results
        results_df = results_df[~results_df["id"].isin(clip_ids_to_rerun)]
        results_df = pd.concat([results_df, new_results_df], ignore_index=True)
        results_df = results_df.sort_values("id").reset_index(drop=True)

        # strip timezone before saving
        for col in ["start_time", "end_time"]:
            if col in results_df.columns:
                results_df[col] = pd.to_datetime(results_df[col]).dt.tz_localize(None)

        results_df.to_excel(results_path, index=False)
        print(f"Updated results.xlsx for participant {enrol_nr}")

        # update ads.xlsx
        if new_ads:
            new_ads_df = pd.DataFrame(new_ads)
            if os.path.exists(ads_path):
                existing_ads = pd.read_excel(ads_path)
                # remove old entries for rerun clips
                existing_ads = existing_ads[~existing_ads["id"].isin(clip_ids_to_rerun)]
                ads_df = pd.concat([existing_ads, new_ads_df], ignore_index=True)
            else:
                ads_df = new_ads_df

            for col in ["start_time", "end_time"]:
                if col in ads_df.columns:
                    ads_df[col] = pd.to_datetime(ads_df[col]).dt.tz_localize(None)

            ads_df.to_excel(ads_path, index=False)
            print(f"Updated ads.xlsx for participant {enrol_nr} ({len(new_ads)} new ad rows)")

    print(f"\n{'='*50}\nRerun complete!\n{'='*50}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--enrol", nargs="+", type=str, help="Only rerun failed clips for specific enrol numbers")
    args = parser.parse_args()

    failed_clips_path = os.path.join(OUTPUT_DIR, "failed_clips.xlsx")
    if not os.path.exists(failed_clips_path):
        print(f"ERROR: {failed_clips_path} not found. Run aggregate.py first.")
        exit(1)

    # optionally filter to specific participants
    failed = pd.read_excel(failed_clips_path)
    if args.enrol:
        failed = failed[failed["enrol_number"].astype(str).isin(args.enrol)]
        print(f"Filtered to participants: {args.enrol}")
    failed.to_excel(failed_clips_path, index=False)  # save filtered version for the function to read

    login(hugg_key)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL, device_map="cuda:0", dtype="auto", attn_implementation="flash_attention_2",
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL)
    print(f"Successfully loaded model and processor with id {MODEL}.")

    rerun_failed_clips(model, processor, failed_clips_path)