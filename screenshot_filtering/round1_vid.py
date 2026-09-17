import json, os, time, ast
import pandas as pd
import argparse
from functools import partial

from screenshot_filtering.questions import instructions_video, parse_qwen_json, fix_json_with_qwen
from PIL import Image
from screenshot_filtering.helpers import (save_excel, save_json, prepare_participant,
                                          BASE_DIR, DATA_DIR, RESULTS_DIR, PLAYBACK_FPS)

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login
from transformers import AutoModelForMultimodalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

# docs from here: https://deepwiki.com/QwenLM/Qwen3-VL/5.4-video-understanding
# and here: https://github.com/QwenLM/Qwen3-VL/blob/50068df2/cookbooks/video_understanding.ipynb
MODEL = "Qwen/Qwen3.8-27B" # https://huggingface.co/Qwen/Qwen3.8-27B


with open(os.path.join(BASE_DIR, 'keys.txt')) as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


def qwen_call_video(model, processor, clip_meta):
    """
    Uses actual .mp4 file + video_metadata for text-timestamp alignment and follows the official Qwen3-VL pattern from the docs.
    clip_meta: dict with keys clip_id, video_path, n_frames, screenshot_paths, start_time, end_time

    The flow is: messages -> process_vision_info(messages, ...) -> qwen-vl-utils -> processor -> model.generate()
    """
    video_id = clip_meta["clip_id"]
    video_path = clip_meta["video_path"]
    n_frames = clip_meta["n_frames"]
    total_duration_sec = n_frames / PLAYBACK_FPS

    print(f'======== Labeling clip: {video_id} ({n_frames} frames, {total_duration_sec:.1f}s). ========\n')

    # single-frame clips can't be decoded as video so pass as image instead
    if n_frames < 2:
        print(f"Single-frame clip, using image fallback.")
        frame = Image.open(clip_meta["screenshot_paths"][0]).convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instructions_video}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"ID: {video_id}"},
                {"type": "image", "image": frame},
            ]}
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False
        ).to(device)

        start_time = time.time()
        with torch.inference_mode():
            generation = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

        response_time = time.time() - start_time
        print(f"Time taken: {response_time:.2f} seconds")

        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
        raw_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(f"Raw response: {raw_response}")

        del inputs, generation, generated_ids_trimmed

        try:
            return parse_qwen_json(raw_response), response_time
        except Exception as e:
            print(f"JSON parse error: {e}, attempting fix...")
            try:
                return fix_json_with_qwen(model, processor, raw_response), response_time
            except Exception as e2:
                print(f"Fix also failed: {e2}")
                return raw_response, response_time
                
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions_video}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"ID: {video_id}"},
                {
                    "type": "video",
                    "video": video_path,
                    "fps": PLAYBACK_FPS,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 384 * 32 * 32,
                    "total_pixels": 24576 * 32 * 32,
                },
            ],
        }
    ]

    # step 1: get text input without tokenizing
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # step 2: extract frames and metadata using process_vision_info
    max_retries = 3
    for attempt in range(max_retries):
        try:
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"process_vision_info failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in 5 seconds...")
                time.sleep(5)
            else:
                raise  # re-raise on final attempt
                
    # step 3: split videos and metadata
    if videos is not None:
        videos, video_metadatas = zip(*videos)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    # step 4: call processor() directly with video_metadata for timestamp alignment
    inputs = processor(
        text=text,
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_resize=False,
        **video_kwargs
    )

    print(f"Input IDs shape: {inputs['input_ids'].shape}")
    inputs = inputs.to(device)

    start_time = time.time()
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Time taken to generate response: {response_time:.2f} seconds")
    print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    raw_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    del inputs, generation, generated_ids_trimmed
    print(f"Raw response: {raw_response}")

    try:
        response = parse_qwen_json(raw_response)
        return response, response_time
    except Exception as e:
        print(f"!!!!!!! JSON parse error: {e} !!!!!!!!")
        try:
            print(f"Failed at position {e.pos}, char: {repr(raw_response[e.pos-2:e.pos+2])}")
            print(f"Bytes: {raw_response[e.pos-2:e.pos+2].encode('utf-8')}")
        except Exception:
            pass
        print(f"\nAttempting fix with Qwen...")
        try:
            response = fix_json_with_qwen(model, processor, raw_response)
            print(f"Fixed response: {response}")
            return response, response_time
        except Exception as e2:
            print(f"Fix also failed: {e2}")
            return raw_response, response_time


def sec_to_frame(sec, playback_fps=PLAYBACK_FPS):
    """Convert a timestamp in seconds to a frame index."""
    return max(0, round(sec * playback_fps))


def filter_ads(media, model, processor):
    """
    media: list of dicts from screenshots_to_videos
    (has video_path, clip_id, n_frames, screenshot_paths, start_time, end_time)
    """
    results = []
    responses = []
    n = 1

    for med in media:
        print(f"Processing video {n}/{len(media)}")
        # TODO: skip clips with < 2 frames before calling qwen ???
        try:
            response, response_time = qwen_call_video(model, processor, med)
            responses.append(response)

            if isinstance(response, str):
                raise ValueError(f"JSON parse failed, raw response: {response[:200]}")

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
                "n_screenshots": len(med["screenshot_paths"])
            }

            ads = item.get("ads", [])
            result_entry["platform"] = item.get("platform", "UNKNOWN")
            result_entry["n_ads"] = len(ads)
            result_entry["ads"] = ads

            results.append(result_entry)

        except Exception as e:
            import traceback
            print(f"Error processing video {med['clip_id']}: {e}")
            print(traceback.format_exc())
            responses.append({"clip_id": med["clip_id"], "error": str(e)})
            results.append({
                "id": med["clip_id"],
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
                "ads": []
            })

        n += 1

    labeling_outputs = pd.DataFrame(results)
    return labeling_outputs, responses


def saved_round1(output_path, clips):
    """Keep successful results only when they still match the prepared clips."""
    if not os.path.exists(output_path):
        return pd.DataFrame()
    existing = pd.read_excel(output_path)
    expected = {clip["clip_id"]: clip for clip in clips}
    keep = []
    for row in existing.to_dict("records"):
        clip = expected.get(row["id"])
        if clip is None:
            raise ValueError("Saved results contain unknown clips; use a fresh results folder")
        if row["n_frames"] != clip["n_frames"] or any(
            pd.Timestamp(row[key]) != pd.Timestamp(clip[key]) for key in ("start_time", "end_time")
        ):
            raise ValueError(f"Prepared clip changed: {row['id']}; use a fresh results folder")
        if row["label"] not in {"AD", "NON_AD"}:
            continue
        try:
            row["ads"] = ast.literal_eval(row["ads"]) if isinstance(row["ads"], str) else row["ads"]
            if not isinstance(row["ads"], list) or len(row["ads"]) != row["n_ads"]:
                continue
            if row["label"] == "AD" and not row["ads"]:
                continue
            if row["label"] == "NON_AD" and row["ads"]:
                continue
            if any(not isinstance(ad, dict) or not {"start_sec", "end_sec"}.issubset(ad) for ad in row["ads"]):
                continue
        except (ValueError, SyntaxError, TypeError):
            continue
        keep.append(row)
    return pd.DataFrame(keep).drop_duplicates("id", keep="last") if keep else pd.DataFrame()


def save_round1(results, responses, participant_dir):
    ads = results.loc[results["n_ads"] > 0].copy()
    if not ads.empty:
        ads = ads.explode("ads").reset_index(drop=True)
        ads = pd.concat([ads.drop(columns="ads"), ads["ads"].apply(pd.Series)], axis=1)
    else:
        ads = pd.DataFrame(columns=[c for c in results.columns if c != "ads"] + ["start_sec", "end_sec"])
    save_json(responses, os.path.join(participant_dir, "responses.json"))
    save_excel(ads, os.path.join(participant_dir, "ads.xlsx"))
    save_excel(results, os.path.join(participant_dir, "results.xlsx"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--enrol", nargs="+", help="One or more enrol numbers to process")
    selection.add_argument("--all", action="store_true", help="Process all participant folders in DATA_DIR")
    args = parser.parse_args()
    enrol_numbers = sorted(folder for folder in os.listdir(DATA_DIR)
                           if folder.isdigit() and os.path.isdir(os.path.join(DATA_DIR, folder))) if args.all else args.enrol
    if not enrol_numbers:
        raise ValueError(f"No participant folders in {DATA_DIR}")
    model = processor = None
    failed = []
    for enrol_number in enrol_numbers:
        try:
            print(f"Processing participant {enrol_number}")
            clips = prepare_participant(os.path.join(DATA_DIR, enrol_number), enrol_number)
            participant_dir = os.path.join(RESULTS_DIR, enrol_number)
            os.makedirs(participant_dir, exist_ok=True)
            existing = saved_round1(os.path.join(participant_dir, "results.xlsx"), clips)
            completed = set(existing["id"]) if not existing.empty else set()
            pending = [clip for clip in clips if clip["clip_id"] not in completed]
            responses_path = os.path.join(participant_dir, "responses.json")
            responses = []
            if os.path.exists(responses_path):
                with open(responses_path, encoding="utf-8") as stream:
                    responses = json.load(stream)
                responses = [r for r in responses if isinstance(r, dict) and r.get("items")
                             and r["items"][0].get("id") in completed]
            if pending:
                if model is None:
                    login(hugg_key)
                    model = AutoModelForMultimodalLM.from_pretrained(
                        MODEL, device_map="cuda:0", dtype="auto", attn_implementation="sdpa").eval()
                    processor = AutoProcessor.from_pretrained(MODEL)
                    processor.apply_chat_template = partial(processor.apply_chat_template, enable_thinking=False)
                results, new_responses = filter_ads(pending, model, processor)
                results = pd.concat([existing, results], ignore_index=True)
                responses.extend(new_responses)
            else:
                print(f"All {len(clips)} clips already processed; skipping annotation.")
                results = existing
            for col in ("start_time", "end_time"):
                results[col] = pd.to_datetime(results[col]).dt.tz_localize(None)
            save_round1(results, responses, participant_dir)
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(enrol_number)
    if failed:
        raise SystemExit(f"Participants requiring attention: {', '.join(failed)}")
    print("Round 1 complete.")
