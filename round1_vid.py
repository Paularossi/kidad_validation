import json, os, time
import pandas as pd
import argparse

from screenshot_filtering.questions import *
from screenshot_filtering.helpers import *

import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# docs from here: https://deepwiki.com/QwenLM/Qwen3-VL/5.4-video-understanding
# and here: https://github.com/QwenLM/Qwen3-VL/blob/50068df2/cookbooks/video_understanding.ipynb
MODEL = "Qwen/Qwen3-VL-32B-Instruct"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYBACK_FPS = 2.0 # must match screenshots_to_videos playback_fps

with open(os.path.join(BASE_DIR, 'keys.txt')) as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


def qwen_call_video(model, processor, clip_meta):
    """
    Uses actual .mp4 file + video_metadata for text-timestamp alignment and follows the official Qwen3-VL pattern from the docs.
    clip_meta: dict with keys clip_id, video_path, n_frames, screenshot_paths, start_time, end_time
    """
    video_id = clip_meta["clip_id"]
    video_path = clip_meta["video_path"]
    n_frames = clip_meta["n_frames"]
    total_duration_sec = n_frames / PLAYBACK_FPS

    print(f'======== Labeling clip: {video_id} ({n_frames} frames, {total_duration_sec:.1f}s). ========\n')

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
                },
            ],
        }
    ]

    # step 1: get text input without tokenizing
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # step 2: extract frames and metadata using process_vision_info
    images, videos, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True
    )

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

    torch.cuda.empty_cache()

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    raw_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
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
        time.sleep(1)

    labeling_outputs = pd.DataFrame(results)
    return labeling_outputs, responses


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--enrol", nargs="+", type=int, help="One or more enrol numbers to process")
parser.add_argument("--all", action="store_true", help="Process all participants")
args = parser.parse_args()

metadata = pd.read_excel(os.path.join(BASE_DIR, "data/metadata.xlsx"))
metadata = metadata[~metadata['enrol_number'].astype(str).str.startswith("32")]

all_enrol_numbers = metadata['enrol_number'].unique().tolist()
if args.all:
    enrol_numbers = all_enrol_numbers
elif args.enrol:
    enrol_numbers = args.enrol
else:
    print("Please provide --enrol <number> or --all")
    exit(1)

login(hugg_key)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, device_map="cuda:0", dtype="auto", attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained(MODEL)
print(f"Successfully loaded model and processor with id {MODEL}.")

for enrol_test_nr in enrol_numbers:
    image_folder = os.path.join(BASE_DIR, f"data/participants/{enrol_test_nr}")
    images_set = set(os.listdir(image_folder)) if os.path.exists(image_folder) else set()

    participant_dir = os.path.join(BASE_DIR, f"data/results_videos/{str(enrol_test_nr)}")
    output_path = os.path.join(participant_dir, "results.xlsx")

    if os.path.exists(output_path):
        print(f"Skipping {enrol_test_nr} - already processed.")
        continue
    elif not images_set:
        print(f"Skipping {enrol_test_nr} - no images found.")
        continue

    os.makedirs(participant_dir, exist_ok=True)
    video_dir = os.path.join(participant_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    print(f"\n{'='*50}\nProcessing participant {enrol_test_nr}\n{'='*50}")
    try:
        meta_test = metadata[metadata['enrol_number'] == enrol_test_nr]
        print(f"Loaded metadata for participant {enrol_test_nr}: {len(meta_test)} screenshots.")
        meta_test['Time'] = pd.to_datetime(meta_test['Time'])
        meta_test = meta_test.sort_values(by='Time')
        meta_test['delta_time'] = meta_test['Time'].diff().dt.total_seconds().fillna(0)
        meta_test["image"] = meta_test["image"].astype(str)
        id_to_ts = dict(zip(meta_test["image"], meta_test["Time"]))

        images_list = meta_test['image'].tolist()
        images_list = [f"{image}.png" for image in images_list]
        meta_images = [os.path.join(image_folder, f) for f in images_list if f in images_set]
        print(f"Images found in folder: {len(meta_images)} out of {len(images_list)} in metadata")

        # create videos and use them primary input instead of frames
        videos_df = screenshots_to_videos(
            meta_images=meta_images,
            id_to_ts=id_to_ts,
            output_folder=video_dir,
            window_minutes=5.0,
            playback_fps=PLAYBACK_FPS
        )
        videos_df["start_time"] = videos_df["start_time"].dt.tz_localize(None)
        videos_df["end_time"] = videos_df["end_time"].dt.tz_localize(None)
        videos_df.to_excel(os.path.join(participant_dir, "frames.xlsx"), index=False)

        clips = videos_df.to_dict("records")

        results, responses = filter_ads(media=clips, model=model, processor=processor)

        results["start_time"] = results["start_time"].dt.tz_localize(None)
        results["end_time"] = results["end_time"].dt.tz_localize(None)
        results.to_excel(output_path, index=False)

        with open(os.path.join(participant_dir, "responses.json"), "w") as f:
            json.dump(responses, f, indent=4)

        ads_df = results[results["n_ads"] > 0].copy()
        if not ads_df.empty:
            ads_df = ads_df.explode("ads").reset_index(drop=True)
            ads_df = pd.concat([
                ads_df.drop(columns="ads"),
                ads_df["ads"].apply(pd.Series)
            ], axis=1)
            ads_df["start_time"] = ads_df["start_time"].dt.tz_localize(None) if ads_df["start_time"].dt.tz is not None else ads_df["start_time"]
            ads_df.to_excel(os.path.join(participant_dir, "ads.xlsx"), index=False)
            print(f"Saved {len(ads_df)} ads to ads.xlsx")
        else:
            print("No ads found for this participant.")

        print(f"Participant {enrol_test_nr} complete. Results saved to {output_path}")

    except Exception as e:
        import traceback
        print(f"Failed for participant {enrol_test_nr}: {e}")
        print(traceback.format_exc())
        continue

print(f"\n{'='*50}\nAll participants processed. Done!\n{'='*50}")
