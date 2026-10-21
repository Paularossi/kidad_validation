import json, os, time, re
import pandas as pd
import argparse
from contextlib import contextmanager

from screenshot_filtering.questions import instructions_video, parse_model_response, parse_qwen_json
from PIL import Image, ImageOps
from screenshot_filtering.helpers import (save_excel, save_json, load_participant, DATA_DIR, RESULTS_DIR, PLAYBACK_FPS, validate_intervals, FRAME_BOUNDS_VERSION, load_annotation_model)


@contextmanager
def uniform_video_frames(paths, size=(553, 1200)):
    """Fit each original into the same RGB canvas without cropping or stretching."""
    frames = []
    try:
        for path in paths:
            with Image.open(path) as source:
                with source.convert("RGB") as rgb:
                    frames.append(ImageOps.pad(rgb, size, method=Image.Resampling.LANCZOS,
                                               color=(0, 0, 0)))
        yield frames
    finally:
        for frame in frames:
            frame.close()


def parse_round1_response(model, processor, raw_response):
    """Recover the first complete item when a later part of the response is invalid."""
    try:
        return parse_qwen_json(raw_response)
    except json.JSONDecodeError:
        text = re.sub(r"^```(?:json)?\s*", "", raw_response.strip())
        prefix = re.match(r'\{\s*"items"\s*:\s*\[\s*', text)
        if prefix:
            try:
                item, _ = json.JSONDecoder().raw_decode(text, prefix.end())
                if isinstance(item, dict):
                    print("Recovered first complete result; ignoring malformed/truncated remainder.", flush=True)
                    # Preserve the full output alongside the recovered item in responses.json.
                    return {"items": [item], "raw_response": raw_response,
                            "recovery": "first_complete_item"}
            except json.JSONDecodeError:
                pass
        return parse_model_response(model, processor, raw_response)


def qwen_call_video(model, processor, clip_meta):
    """
    Uses every ordered original screenshot as a video frame, avoiding decoder sampling.
    clip_meta: dict with keys clip_id, video_path, n_frames, screenshot_paths, start_time, end_time

    The flow is: messages -> process_vision_info(messages, ...) -> qwen-vl-utils -> processor -> model.generate()
    """
    import torch
    from qwen_vl_utils import process_vision_info
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    video_id = clip_meta["clip_id"]
    n_frames = clip_meta["n_frames"]
    paths = clip_meta["screenshot_paths"]
    if n_frames <= 0 or len(paths) != n_frames:
        raise ValueError(f"Invalid screenshot count for {video_id}")
    last_frame_sec = (n_frames - 1) / PLAYBACK_FPS
    timing_text = (
        f"ID: {video_id}. Original frame count: {n_frames}. FPS: {PLAYBACK_FPS}. "
        f"Frame spacing: {1 / PLAYBACK_FPS} seconds. First frame timestamp: 0.0. "
        f"Last original frame timestamp: {last_frame_sec}. "
        "Report the FIRST and LAST included frames; both endpoints are inclusive. "
        "Ignore any duplicated final frame used only for processing padding."
    )
    total_duration_sec = n_frames / PLAYBACK_FPS

    print(f'======== Labeling clip: {video_id} ({n_frames} frames, {total_duration_sec:.1f}s). ========\n')

    # single-frame clips can't be decoded as video so pass as image instead
    if n_frames < 2:
        print(f"Single-frame clip, using image fallback.")
        with Image.open(clip_meta["screenshot_paths"][0]) as source:
            frame = source.convert("RGB")
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instructions_video}]},
            {"role": "user", "content": [
                {"type": "text", "text": timing_text},
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

        return parse_round1_response(model, processor, raw_response), response_time

    with uniform_video_frames(paths) as frames:
        print(f"Prepared {len(frames)} frames on a 553 x 1200 canvas (aspect ratio preserved).", flush=True)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": instructions_video}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": timing_text},
                    {
                        "type": "video",
                        "video": frames,
                        "sample_fps": PLAYBACK_FPS,
                        "raw_fps": PLAYBACK_FPS,
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

    if videos is None or len(videos) != 1 or video_metadatas is None:
        raise ValueError(f"Expected exactly one prepared video for {video_id}")
    prepared_count = int(videos[0].shape[0])
    # qwen-vl-utils pads odd frame lists by duplicating their last image.
    if prepared_count not in {n_frames, n_frames + (n_frames % 2)}:
        raise ValueError(f"Frame coverage changed for {video_id}: {n_frames} -> {prepared_count}")
    indices = [int(i) for i in video_metadatas[0]["frames_indices"]]
    if indices[:n_frames] != list(range(n_frames)) or len(indices) != prepared_count:
        raise ValueError(f"Frames were sampled or reordered for {video_id}")
    video_metadatas[0] = {
        **video_metadatas[0],
        "fps": PLAYBACK_FPS,
        "total_num_frames": n_frames,
        "duration": n_frames / PLAYBACK_FPS,
        "frames_indices": list(range(n_frames)) + [n_frames - 1] * (prepared_count - n_frames),
    }
    video_kwargs["do_sample_frames"] = False
    print(f"Frame coverage: {n_frames}/{n_frames} originals; "
          f"{prepared_count - n_frames} padding frames; last timestamp {last_frame_sec}s.")

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

    return parse_round1_response(model, processor, raw_response), response_time


def extract_round1_item(response, clip_id):
    """Accept one result, identical copies, or a detached redundant empty ads object.

    Preserve the raw response in the caller's response log. Never discard a second
    distinct classification, nonempty intervals, unknown fields or conflicting content.
    """
    items = response.get("items") if isinstance(response, dict) else None
    if (isinstance(items, list) and len(items) > 1
            and isinstance(items[0], dict) and items[0].get("id") == clip_id
            and all(isinstance(item, dict) and item == items[0] for item in items[1:])):
        print(f"Normalized response for {clip_id}: collapsed {len(items)} identical result objects into one.", flush=True)
        items = items[:1]  # Do not mutate the response retained in the caller's log.
    if isinstance(items, list) and len(items) == 2:
        main = [item for item in items if isinstance(item, dict) and item.get("id") == clip_id]
        if (len(main) == 1 and main[0].get("label") in {"NON_AD", "UNCLEAR"}
                and main[0].get("ads", []) == []
                and sum(item == {"ads": []} for item in items) == 1):
            print(f"Normalized response for {clip_id}: removed a redundant separate empty ads object.", flush=True)
            items = main
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        raise ValueError("Expected exactly one result object for the clip")
    return items[0]


def filter_ads(media, model, processor):
    """
    media: list of dicts from load_participant
    (has video_path, clip_id, n_frames, screenshot_paths, start_time, end_time)
    """
    results = []
    responses = []
    n = 1

    for med in media:
        print(f"Processing video {n}/{len(media)}")
        try:
            response, response_time = qwen_call_video(model, processor, med)
            responses.append(response)

            if isinstance(response, str):
                raise ValueError(f"JSON parse failed, raw response: {response[:200]}")

            item = extract_round1_item(response, med["clip_id"])
            if item.get("id") != med["clip_id"]:
                raise ValueError("Response clip ID does not match the input")
            if item.get("label") not in {"AD", "NON_AD", "UNCLEAR"}:
                raise ValueError("Invalid clip classification")
            ads = validate_intervals(item.get("ads", []), item["label"], med["n_frames"])
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
            result_entry["frame_bounds_version"] = FRAME_BOUNDS_VERSION

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
                "frame_bounds_version": FRAME_BOUNDS_VERSION,
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
    if existing["id"].duplicated().any():
        raise ValueError("Duplicate saved clip IDs; review results before resuming")
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
        if row.get("frame_bounds_version") != FRAME_BOUNDS_VERSION:
            raise ValueError("Saved results use an older timing/prompt convention; use fresh results")
        try:
            row["ads"] = validate_intervals(row["ads"], row["label"], clip["n_frames"])
            if len(row["ads"]) != row["n_ads"]:
                continue
        except (ValueError, SyntaxError, TypeError):
            continue
        keep.append(row)
    return pd.DataFrame(keep)


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


def main():
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
    for participant_index, enrol_number in enumerate(enrol_numbers, 1):
        try:
            print(f"Processing participant {participant_index}/{len(enrol_numbers)}: {enrol_number}", flush=True)
            clips = load_participant(os.path.join(DATA_DIR, enrol_number))
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
                    model, processor = load_annotation_model()
                results, new_responses = filter_ads(pending, model, processor)
                results = pd.concat([existing, results], ignore_index=True)
                responses.extend(new_responses)
            else:
                print(f"All {len(clips)} clips already processed; skipping annotation.")
                results = existing
            for col in ("start_time", "end_time"):
                results[col] = pd.to_datetime(results[col]).dt.tz_localize(None)
            save_round1(results, responses, participant_dir)
            counts = results["label"].value_counts()
            errors, unclear = int(counts.get("UNCERTAIN", 0)), int(counts.get("UNCLEAR", 0))
            print(f"Participant {enrol_number}: {len(results) - errors - unclear} classified, "
                  f"{unclear} unclear, {errors} errors.", flush=True)
            if errors or unclear:
                failed.append(enrol_number)
        except Exception:
            import traceback
            traceback.print_exc()
            failed.append(enrol_number)
    if failed:
        raise SystemExit(f"Participants requiring attention: {', '.join(failed)}")
    print("Round 1 complete.")


if __name__ == "__main__":
    main()
