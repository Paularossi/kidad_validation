import json, os, time, re, ast
import pandas as pd
from PIL import Image

from screenshot_filtering.questions import instructions_round2, parse_qwen_json, fix_json_with_qwen
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SIZE = (553, 1200)  # width, height

with open(os.path.join(BASE_DIR, 'keys.txt')) as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


def qwen_call_round2(model, processor, ad_row):
    """Run round 2 classification on a single ad."""

    ad_id = f"{ad_row['enrol_number']}_{ad_row['id']}_f{int(ad_row['start_frame'])}"
    print(f'======== Labeling ad: {ad_id} ========\n')

    # load frames from ad_screenshot_paths
    screenshot_paths = ast.literal_eval(ad_row["ad_screenshot_paths"]) if isinstance(ad_row["ad_screenshot_paths"], str) else ad_row["ad_screenshot_paths"]
    frames = []
    for p in screenshot_paths:
        if os.path.exists(p):
            img = Image.open(p).convert("RGB")
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.LANCZOS)
            frames.append(img)

    if not frames:
        raise ValueError(f"No valid frames found for ad {ad_id}")

    print(f"Loaded {len(frames)} frames for ad {ad_id}")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions_round2}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"ID: {ad_id}"},
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
    ).to(device)

    start_time = time.time()
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Time taken: {response_time:.2f} seconds")
    print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    torch.cuda.empty_cache()

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    raw_response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    try:
        response = parse_qwen_json(raw_response)
        print(response)
        return response, response_time
    except Exception as e:
        print(f"JSON parse error: {e}, attempting fix with Qwen...")
        try:
            response = fix_json_with_qwen(model, processor, raw_response)
            print(f"Fixed response: {response}")
            return response, response_time
        except Exception as e2:
            print(f"Fix also failed: {e2}")
            print(f"Raw response: {raw_response}")
            return raw_response, response_time


def classify_food_ads(food_ads_df, model, processor, output_path, responses_path):
    """Loop through all food ads and classify them."""

    # load existing results if any (to skip already processed ads)
    if os.path.exists(output_path):
        existing = pd.read_excel(output_path)
        processed_ids = set(existing["ad_id"].tolist())
        print(f"Found {len(processed_ids)} already processed ads, skipping.")
    else:
        existing = pd.DataFrame()
        processed_ids = set()

    # load existing responses if any
    if os.path.exists(responses_path):
        with open(responses_path, "r") as f:
            all_responses = json.load(f)
    else:
        all_responses = []

    results = []
    n = 1

    for _, ad_row in food_ads_df.iterrows():
        ad_id = f"{ad_row['enrol_number']}_{ad_row['id']}_f{int(ad_row['start_frame'])}"

        if ad_id in processed_ids:
            print(f"Skipping {ad_id} - already processed.")
            n += 1
            continue

        print(f"\nProcessing ad {n}/{len(food_ads_df)}: {ad_id}")

        try:
            response, response_time = qwen_call_round2(model, processor, ad_row)
            all_responses.append({"ad_id": ad_id, "response": response})

            if isinstance(response, str):
                raise ValueError(f"JSON parse failed: {response[:200]}")

            result_entry = {
                "ad_id": ad_id,
                "enrol_number": str(ad_row["enrol_number"]),
                "clip_id": ad_row["id"],
                "platform": ad_row.get("platform", ""),
                "start_frame": ad_row["start_frame"],
                "end_frame": ad_row["end_frame"],
                "food_ad_round1": ad_row.get("food_ad", ""),
                "brands_round1": ad_row.get("brands", ""),
                "response_time": round(response_time, 2),
                # round 2 fields
                "overall_category": response.get("overall_category", []),
                "product_categories": response.get("product_categories", []),
                "brand_business_categories": response.get("brand_business_categories", []),
                "brand_main": response.get("brand_main", ""),
                "brand_other": response.get("brand_other", []),
                "type_of_marketing": response.get("type_of_marketing", []),
                "marketing_strategies": response.get("marketing_strategies", []),
                "target_group": response.get("target_group", ""),
                "who_category": response.get("who_category", []),
                "nova_category": response.get("nova_category", ""),
                "confidence": response.get("confidence", 0.0),
            }

        except Exception as e:
            import traceback
            print(f"Error processing ad {ad_id}: {e}")
            print(traceback.format_exc())
            all_responses.append({"ad_id": ad_id, "error": str(e)})
            result_entry = {
                "ad_id": ad_id,
                "enrol_number": str(ad_row["enrol_number"]),
                "clip_id": ad_row["id"],
                "platform": ad_row.get("platform", ""),
                "start_frame": ad_row["start_frame"],
                "end_frame": ad_row["end_frame"],
                "food_ad_round1": ad_row.get("food_ad", ""),
                "brands_round1": ad_row.get("brands", ""),
                "response_time": None,
                "overall_category": None,
                "product_categories": None,
                "brand_business_categories": None,
                "brand_main": None,
                "brand_other": None,
                "type_of_marketing": None,
                "marketing_strategies": None,
                "target_group": None,
                "who_category": None,
                "nova_category": None,
                "confidence": 0.0,
                "error": str(e),
            }

        results.append(result_entry)
        n += 1
        time.sleep(1)

        # save incrementally every 10 ads to avoid losing progress
        if len(results) % 10 == 0:
            interim = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            interim.to_excel(output_path, index=False)
            with open(responses_path, "w") as f:
                json.dump(all_responses, f, indent=4)
            print(f"Progress saved: {len(results)} ads processed so far.")

    # final save
    final = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    final.to_excel(output_path, index=False)
    with open(responses_path, "w") as f:
        json.dump(all_responses, f, indent=4)

    print(f"\nRound 2 complete. {len(results)} ads classified.")
    print(f"Results saved to {output_path}")

    return final, all_responses


# ============================================================
# MAIN
# ============================================================

AGGREGATED_DIR = os.path.join(BASE_DIR, "data/aggregated")
output_path = os.path.join(AGGREGATED_DIR, "round2_results.xlsx")
responses_path = os.path.join(AGGREGATED_DIR, "round2_responses.json")

# load food ads
food_ads_path = os.path.join(AGGREGATED_DIR, "food_ads_round2.xlsx")
if not os.path.exists(food_ads_path):
    print(f"ERROR: {food_ads_path} not found. Run aggregate.py first.")
    exit(1)

food_ads = pd.read_excel(food_ads_path)
print(f"Loaded {len(food_ads)} food ads for round 2 classification.")

# load model once
login(hugg_key)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", # use 32B for the main run, 2B for testing
    device_map="cpu", # only for now
    dtype="auto",
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct") # change this to 32B
print("Model loaded successfully.")

# ===== TEST RUN - REMOVE LATER =====
food_ads = food_ads.iloc[263:265] # just two random ads 

# run round 2
results, responses = classify_food_ads(food_ads, model, processor, output_path, responses_path)

print(f"\n{'='*50}")
print("Round 2 classification complete!")
print(f"{'='*50}")