import json, os, time, ast
import pandas as pd
from PIL import Image
from screenshot_filtering.helpers import atomic_output, save_json, BASE_DIR, AGGREGATED_DIR, PLAYBACK_FPS

from screenshot_filtering.questions import instructions_round2, parse_qwen_json, fix_json_with_qwen
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from huggingface_hub import login
from transformers import AutoModelForMultimodalLM, AutoProcessor

TARGET_SIZE = (553, 1200)  # width, height
MODEL = "Qwen/Qwen3.8-27B"

with open(os.path.join(BASE_DIR, 'keys.txt')) as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


def qwen_call_round2(model, processor, ad_row):
    """Run round 2 classification on a single ad."""

    ad_id = f"{ad_row['enrol_number']}_{ad_row['id']}_f{int(ad_row['start_frame'])}"
    print(f'======== Labeling ad: {ad_id} ========\n')

    # load frames from ad_screenshot_paths
    screenshot_paths = ast.literal_eval(ad_row["ad_screenshot_paths"]) if isinstance(ad_row["ad_screenshot_paths"], str) else ad_row["ad_screenshot_paths"]
    if not isinstance(screenshot_paths, (list, tuple)) or not screenshot_paths:
        raise ValueError(f"No screenshot paths for ad {ad_id}")
    expected_count = int(ad_row["end_frame"]) - int(ad_row["start_frame"]) + 1
    if expected_count <= 0 or len(screenshot_paths) != expected_count:
        raise ValueError(f"Screenshot count does not match frame bounds for {ad_id}")
    missing = [p for p in screenshot_paths if not isinstance(p, str) or not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"Missing screenshots for {ad_id}: {missing[:3]}")
    frames = []
    for p in screenshot_paths:
        with Image.open(p) as source:
            img = source.convert("RGB")
        if img.size != TARGET_SIZE:
            img = img.resize(TARGET_SIZE, Image.LANCZOS)
        frames.append(img)

    print(f"Loaded {len(frames)} frames for ad {ad_id}")

    if len(frames) == 1: # a single screenshot is an image
        media_content = {
            "type": "image",
            "image": frames[0],
        }
        processor_kwargs = {}
    
    else:
        media_content = {
            "type": "video",
            "video": frames,
        }
    
        video_metadata = [{
            "total_num_frames": len(frames),
            "fps": PLAYBACK_FPS,
            "duration": len(frames) / PLAYBACK_FPS,
            "frames_indices": list(range(len(frames))),
        }]
    
        processor_kwargs = {
            "do_sample_frames": False,
            "video_metadata": video_metadata,
            "cap_pixels_per_frame": True,
            "return_metadata": True,
        }

    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions_round2}]},
        {"role": "user", "content": [
                {"type": "text", "text": f"ID: {ad_id}"},
                media_content,
            ],
        },
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
        processor_kwargs=processor_kwargs
    )
    # metadata has already been used to build timestamp tokens
    inputs.pop("video_metadata", None)
    inputs = inputs.to(device)
    print(f"Input IDs shape: {inputs.input_ids.shape}")

    start_time = time.time()
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Time taken: {response_time:.2f} seconds")
    print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


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


def validate_response(response, ad_id):
    """Reject incomplete annotations; retain empty fields only for skipped steps."""
    if not isinstance(response, dict):
        raise ValueError("Response must be a JSON object")

    # a missing ID is safe to fill because each call processes exactly one ad
    if "id" not in response:
        response["id"] = ad_id
    elif response["id"] != ad_id:
        raise ValueError(f"Response ID mismatch: expected {ad_id}, got {response['id']}")
    allowed = {
        "overall_category": {"PRODUCT_VISIBLE+BRAND_VISIBLE", "BRAND_VISIBLE_ONLY", "UNCLEAR", "NOT_APPLICABLE"},
        "product_categories": {"FOOD_PRODUCT", "ALCOHOL", "TOBACCO_NICOTINE", "INFANT_FORMULA", "UNCLEAR", "NOT_APPLICABLE"},
        "brand_business_categories": {"FOOD_COMPANY", "RETAILER", "RESTAURANT", "ALCOHOL_BRAND", "TOBACCO_NICOTINE_BRAND", "INFANT_FORMULA_BRAND", "DELIVERY_PLATFORM", "UNCLEAR", "NOT_APPLICABLE"},
    }
    for key, labels in allowed.items():
        value = response.get(key)
        if not isinstance(value, list) or any(not isinstance(v, str) or v not in labels for v in value):
            raise ValueError(f"Invalid or missing {key}")
        if not value and (key != "product_categories" or "PRODUCT_VISIBLE+BRAND_VISIBLE" in response["overall_category"]):
            raise ValueError(f"Missing classification for {key}")
    relevance = response.get("ad_relevance")
    if relevance not in {"YES", "NO", "UNSURE"}:
        raise ValueError("Invalid or missing ad_relevance")
    categories = response["product_categories"] + response["brand_business_categories"]
    if "NOT_APPLICABLE" in categories and relevance != "NO":
        raise ValueError("NOT_APPLICABLE requires ad_relevance NO")
    if "NOT_APPLICABLE" not in categories and "UNCLEAR" in categories and relevance != "UNSURE":
        raise ValueError("UNCLEAR requires ad_relevance UNSURE")
    confidence = response.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid or missing confidence")
    skipped = {"brand_main": "", "brand_other": [], "type_of_marketing": [],
               "marketing_strategies": [], "target_group": "", "who_category": [], "nova_category": ""}
    if relevance != "YES":
        # do not export detailed labels for ads excluded by the stopping rule
        response = {**response, **skipped}
    else:
        # use the prompt's output vocabulary to reject invented category labels
        schema_line = next(line.strip() for line in instructions_round2.splitlines() if line.strip().startswith('{"id":'))
        schema = json.loads(schema_line)
        for key in ("type_of_marketing", "marketing_strategies", "target_group", "who_category", "nova_category"):
            spec = schema[key]
            labels = set((spec[0] if isinstance(spec, list) else spec).split("|"))
            value = response.get(key)
            values = value if isinstance(value, list) else [value]
            if any(not isinstance(v, str) or v not in labels for v in values):
                raise ValueError(f"Invalid category in {key}")
        for key, default in skipped.items():
            value = response.get(key)
            if not isinstance(value, type(default)):
                raise ValueError(f"Invalid or missing {key}")
            if isinstance(value, list) and any(not isinstance(v, str) for v in value):
                raise ValueError(f"Invalid labels in {key}")
            if key not in {"brand_main", "brand_other"} and not value:
                raise ValueError(f"Missing classification for {key}")
    return response


def save_results(data, path):
    """Write participant IDs as actual Excel strings, including resumed rows."""
    data["enrol_number"] = data["enrol_number"].astype(str)
    with atomic_output(path) as temporary:
        with pd.ExcelWriter(temporary, engine="openpyxl") as writer:
            data.to_excel(writer, index=False)
            column = data.columns.get_loc("enrol_number") + 1
            for row in writer.sheets["Sheet1"].iter_rows(min_row=2, min_col=column, max_col=column):
                row[0].number_format = "@"



def classify_food_ads(food_ads_df, model, processor, output_path, responses_path):
    """Loop through all food ads and classify them."""

    # load existing results if any (to skip already processed ads)
    if os.path.exists(output_path):
        existing = pd.read_excel(output_path, dtype={"enrol_number": str})

        # keep only successful rows, since failed ads must be attempted again
        if "error" in existing.columns:
            failed_mask = (existing["error"].fillna("").astype(str).str.strip().ne(""))
            print(f"Found {failed_mask.sum()} failed ads to retry.")
            existing = existing.loc[~failed_mask].copy()

        existing = existing.drop_duplicates(subset="ad_id", keep="last")
        current = {f"{r['enrol_number']}_{r['id']}_f{int(r['start_frame'])}": r
                   for r in food_ads_df.to_dict("records")}
        existing = existing.loc[existing["ad_id"].isin(current)].copy()
        for row in existing.to_dict("records"):
            source = current[row["ad_id"]]
            old_paths = ast.literal_eval(row["ad_screenshot_paths"]) if isinstance(row["ad_screenshot_paths"], str) else row["ad_screenshot_paths"]
            new_paths = ast.literal_eval(source["ad_screenshot_paths"]) if isinstance(source["ad_screenshot_paths"], str) else source["ad_screenshot_paths"]
            if int(row["end_frame"]) != int(source["end_frame"]) or old_paths != new_paths:
                raise ValueError(f"Screenshots changed for {row['ad_id']}; use a fresh Round 2 output file")
        processed_ids = set(existing["ad_id"].astype(str))

        print(f"Found {len(processed_ids)} successfully processed ads, skipping.")
    else:
        existing = pd.DataFrame()
        processed_ids = set()

    # load existing responses if any
    if os.path.exists(responses_path):
        with open(responses_path, "r") as f:
            loaded_responses = json.load(f)

        # remove old errors and responses belonging to failed ads
        all_responses = [
            item
            for item in loaded_responses
            if item.get("ad_id") in processed_ids
            and "error" not in item
        ]
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
            if model is None:
                login(hugg_key)
                model = AutoModelForMultimodalLM.from_pretrained(
                    MODEL, device_map="cuda:0", dtype="auto", attn_implementation="sdpa").eval()
                processor = AutoProcessor.from_pretrained(MODEL)
                print("Model loaded successfully.")
            response, response_time = qwen_call_round2(model, processor, ad_row)

            if not isinstance(response, dict):
                raise ValueError(f"JSON parsing failed or returned the wrong type: {str(response)[:200]}")

            response = validate_response(response, ad_id)
            all_responses.append({"ad_id": ad_id, "response": response})

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
                "ad_relevance": response.get("ad_relevance", ""),
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
                "ad_relevance": None,
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

        result_entry["ad_screenshot_paths"] = ad_row["ad_screenshot_paths"]
        results.append(result_entry)
        if "error" not in result_entry:
            processed_ids.add(ad_id)
        n += 1

        # save incrementally every 10 ads to avoid losing progress
        if len(results) % 10 == 0:
            interim = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
            save_results(interim, output_path)
            save_json(all_responses, responses_path)
            print(f"Progress saved: {len(results)} ads processed so far.")

    # final save
    final = pd.concat([existing, pd.DataFrame(results)], ignore_index=True)
    save_results(final, output_path)
    save_json(all_responses, responses_path)

    print(f"\nRound 2 complete. {len(results)} ads classified.")
    print(f"Results saved to {output_path}")

    return final, all_responses


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    output_path = os.path.join(AGGREGATED_DIR, "round2_results.xlsx")
    responses_path = os.path.join(AGGREGATED_DIR, "round2_responses.json")

    # load food ads
    #food_ads_path = os.path.join(AGGREGATED_DIR, "food_ads_round2.xlsx")
    all_ads_path = os.path.join(AGGREGATED_DIR, "all_ads_round2.xlsx") # all ads
    if not os.path.exists(all_ads_path):
        print(f"ERROR: {all_ads_path} not found. Run aggregate.py first.")
        exit(1)

    all_ads = pd.read_excel(all_ads_path, dtype={"enrol_number": str})
    if "frame_bounds_version" not in all_ads.columns or not all_ads["frame_bounds_version"].eq("exclusive_seconds_v1").all():
        raise ValueError("Run aggregate.py again to rebuild the exclusive-end frame bounds")
    print(f"Loaded {len(all_ads)} ads for round 2 classification.")

    if all_ads.empty:
        print("No ads to annotate.")
        raise SystemExit(0)

    # run round 2
    results, responses = classify_food_ads(all_ads, None, None, output_path, responses_path)

    print(f"\n{'='*50}")
    print("Round 2 classification complete!")
    print(f"{'='*50}")