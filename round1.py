import json, os, time
import pandas as pd
import argparse

from screenshot_filtering.questions import *
from screenshot_filtering.helpers import *

import torch # use the GPU if available

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from accelerate import infer_auto_device_map # to offload to gpu
# 'models.qwen3_vl.video_processing_qwen3_vl'

MODEL = "Qwen/Qwen3-VL-32B-Instruct" # apparently can process videos too
# could also try OpenGVLab/InternVL3-38B-hf https://huggingface.co/OpenGVLab/InternVL3-38B-hf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # always points to kidad/
TARGET_SIZE = (553, 1200)  # width, height for the screenshots

with open(os.path.join(BASE_DIR, 'keys.txt')) as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


def qwen_call_video(model, processor, video_frames):
    # sometimes this error "all input arrays must have the same shape" happens because the screenshots have different resolutions (width/height)
    # so make sure that all frames have the same resolution (553x1200)
    video_id = video_frames["clip_id"]
    frames = video_frames["frames"]

    frames = [f.resize(TARGET_SIZE, Image.LANCZOS) if f.size != TARGET_SIZE else f for f in frames]
    
    print(f'======== Labeling clip: {video_id} ({len(frames)} frames). ========\n')

    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions_video}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"ID: {video_id}"},
            {
                "type": "video",
                "video": frames,
                "sample_fps": 1,
                "raw_fps": 1,
            },
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
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
    raw_response  = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Raw response: {raw_response}")
    try:
        response = parse_qwen_json(raw_response)
        #print(response)
        return response, response_time
    except Exception as e:
        print(f"!!!!!!! JSON parse error: {e} !!!!!!!!")
        # debug: print exact bytes at failure point
        print(f"Failed at position {e.pos}, char: {repr(raw_response[e.pos-2:e.pos+2])}")
        print(f"Bytes: {raw_response[e.pos-2:e.pos+2].encode('utf-8')}")
        #print(f"Raw response: {raw_response}")

        print(f"\n Attempting fix with Qwen...")
        try:
            response = fix_json_with_qwen(model, processor, raw_response)
            print(f"Fixed response: {response}")
            return response, response_time
        except Exception as e2:
            print(f"Fix also failed: {e2}")
            #print(f"Raw response: {raw_response}")
            return raw_response, response_time


def filter_ads(media, model, processor):

    results = [] # for the labels
    responses = []
    n = 1

    for med in media:
        print(f"Processing video {n}/{len(media)}")
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
                "id": item["id"],
                "label": label,
                "confidence": confidence,
                "signals": signals,
                "response_time": round(response_time, 2),
                "n_frames": med["n_frames"],
                "start_time": med["start_time"],
                "end_time": med["end_time"],
                "n_screenshots": len(med["screenshot_paths"])
            }

            ads = item.get("ads", []) # get additional info in case an ad was found
            result_entry["platform"] = item.get("platform", "UNKNOWN")
            result_entry["n_ads"] = len(ads)
            result_entry["ads"] = ads  # full ads list as JSON

            # flatten first ad's details for easy reading
            if ads:
                result_entry["food_ad"] = ads[0].get("food_ad", "")
                result_entry["brands"] = ads[0].get("brands", [])
                result_entry["start_frame"] = ads[0].get("start_frame")
                result_entry["end_frame"] = ads[0].get("end_frame")
                # directly retrieve the ad screenshot paths
                start = ads[0].get("start_frame", 0)
                end = ads[0].get("end_frame", 0)
                result_entry["ad_screenshot_paths"] = med["screenshot_paths"][start:end+1]
            else:
                result_entry["food_ad"] = None
                result_entry["brands"] = None
                result_entry["start_frame"] = None
                result_entry["end_frame"] = None
                result_entry["ad_screenshot_paths"] = []

            results.append(result_entry)

        except Exception as e:
            print(f"Error processing video {med['clip_id']}: {e}")
            responses.append({"clip_id": med["clip_id"], "error": str(e)})  # always append something
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
                "n_ads": 0,
                "ads": [],
                "food_ad": None,
                "brands": None,
            })

        n += 1
        time.sleep(1) # to avoid rate limiting

    labeling_outputs = pd.DataFrame(results)

    return labeling_outputs, responses


# parse the arguments (if any)
parser = argparse.ArgumentParser()
parser.add_argument("--enrol", nargs="+", type=int, help="One or more enrol numbers to process")
parser.add_argument("--all", action="store_true", help="Process all participants")
args = parser.parse_args()

# ============================================================================
# start from here
metadata = pd.read_excel(os.path.join(BASE_DIR, "data/metadata.xlsx"))
metadata = metadata[~metadata['enrol_number'].astype(str).str.startswith("32")] # remove test accounts

# get list of participants to process
all_enrol_numbers = metadata['enrol_number'].unique().tolist()
if args.all:
    enrol_numbers = all_enrol_numbers
elif args.enrol:
    enrol_numbers = args.enrol
else:
    print("Please provide --enrol <number> or --all")
    exit(1)

# load model once before the participant loop
login(hugg_key)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL, device_map="cuda:0", dtype="auto", attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained(MODEL)
print(f"Successfully loaded model and processor with id {MODEL}.")

# loop through all participants and process their screenshots
for enrol_test_nr in enrol_numbers:
    image_folder = os.path.join(BASE_DIR, f"data/participants/{enrol_test_nr}") # one folder per participant
    images_set = set(os.listdir(image_folder)) if os.path.exists(image_folder) else set() # for faster lookup  
    
    # create participant folder
    participant_dir = os.path.join(BASE_DIR, f"data/results/{str(enrol_test_nr)}")
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

        meta_test['delta_time'] = meta_test['Time'].diff().dt.total_seconds().fillna(0) # find the time delta between screenshots
        meta_test["image"] = meta_test["image"].astype(str)
        id_to_ts = dict(zip(meta_test["image"], meta_test["Time"]))

        images_list = meta_test['image'].tolist()
        images_list = [f"{image}.png" for image in images_list]
        meta_images = [os.path.join(image_folder, f) for f in images_list if f in images_set]
        print(f"Images found in folder: {len(meta_images)} out of {len(images_list)} in metadata")

        # create the videos (for debugging or manual inspection, but actually pass the frames directly to the model)
        videos = screenshots_to_videos(meta_images=meta_images, id_to_ts=id_to_ts, output_folder=video_dir, window_minutes = 10.0)
        # just pass the screenshots as frames
        frames = group_screenshots(meta_images, id_to_ts, window_minutes=10.0)
        frames_df = pd.DataFrame(frames)
        frames_df["start_time"] = frames_df["start_time"].dt.tz_localize(None)
        frames_df["end_time"] = frames_df["end_time"].dt.tz_localize(None)
        frames_df.to_excel(os.path.join(participant_dir, "frames.xlsx"), index=False)

        # filter ads from non-ads
        results, responses = filter_ads(media=frames, model=model, processor=processor)

        results["start_time"] = results["start_time"].dt.tz_localize(None)
        results["end_time"] = results["end_time"].dt.tz_localize(None)
        results.to_excel(output_path, index=False)

        # save responses for debugging
        with open(os.path.join(participant_dir, "responses.json"), "w") as f:
            json.dump(responses, f, indent=4)

        # save exploded ads for round 2 annotation
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
        print(f"Failed for participant {enrol_test_nr}: {e}")
        continue  # skip to next participant instead of crashing

print(f"\n{'='*50}\nAll participants processed. Done!\n{'='*50}")
