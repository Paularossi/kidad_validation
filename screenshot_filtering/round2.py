import json, os, time, ast
import pandas as pd
from PIL import Image
from screenshot_filtering.helpers import (atomic_output, save_json, AGGREGATED_DIR, PLAYBACK_FPS, FRAME_BOUNDS_VERSION, load_annotation_model)

from screenshot_filtering.questions import instructions_round2, parse_model_response
TARGET_SIZE = (553, 1200)  # Keep uniform frame dimensions for video processing.


def qwen_call_round2(model, processor, ad_row):
    """Run round 2 classification on a single ad."""
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

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
                {"type": "text", "text": f"ID: {ad_id}" + ad_row.get("validation_feedback", "")},
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

    print(f"Raw response: {raw_response}")
    return parse_model_response(model, processor, raw_response), response_time


def validate_response(response, ad_id):
    """Reject incomplete annotations; retain empty fields only for skipped steps."""
    if not isinstance(response, dict):
        raise ValueError("Response must be a JSON object")

    # a missing ID is safe to fill because each call processes exactly one ad
    if "id" not in response:
        response["id"] = ad_id
    elif response["id"] != ad_id:
        raise ValueError(f"Response ID mismatch: expected {ad_id}, got {response['id']}")
    schema_line = next(line.strip() for line in instructions_round2.splitlines() if line.strip().startswith('{"id":'))
    schema = json.loads(schema_line)
    allowed = {key: set(schema[key][0].split("|"))
               for key in ("overall_category", "product_categories", "brand_business_categories")}
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
    relevant = set(categories) - {"NOT_APPLICABLE", "UNCLEAR"}
    expected_relevance = "YES" if relevant else ("UNSURE" if "UNCLEAR" in categories else "NO")
    if relevance != expected_relevance:
        raise ValueError(f"Categories require ad_relevance {expected_relevance}")
    confidence = response.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("Invalid or missing confidence")
    skipped = {"type_of_marketing": [], "marketing_strategies": [], "target_group": "", "who_category": [], "nova_category": ""}
    for key, default in {"brand_main": "", "brand_other": []}.items(): # validate brands for ALL ads
        value = response.get(key)
        if not isinstance(value, type(default)) or (
                isinstance(value, list) and any(not isinstance(v, str) for v in value)):
            raise ValueError(f"Invalid or missing {key}")
    if relevance != "YES":
        # brands are kept and all other detailed labels are cleared by the stopping rule
        response = {**response, **skipped}
    else:
        # Enforce skipped classifications without inventing a food category.
        products = set(response["product_categories"])
        if "FOOD_SUPPLEMENTS" in products and not products.intersection({"FOOD_PRODUCT", "INFANT_FORMULA", "UNCLEAR"}):
            response = {**response, "who_category": ["NA"], "nova_category": "NA_PROCESSING"}
        elif response.get("who_category") == ["NA"]:
            response = {**response, "nova_category": "NA_PROCESSING"}
        # use the prompt's output vocabulary to reject invented category labels
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
            if not value:
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


def checked_round2(model, processor, ad_row, ad_id, logs):
    """Bind each synchronous response to its input; retry invalid annotations once."""
    elapsed = 0
    request = ad_row
    for attempt in range(2):
        raw, duration = qwen_call_round2(model, processor, request)
        elapsed += duration
        entry = {"ad_id": ad_id, "attempt": attempt + 1, "raw_response": raw}
        logs.append(entry)
        try:
            if not isinstance(raw, dict):
                raise ValueError("Response must be a JSON object")
            # this call contains exactly one known ad. IDs are bookkeeping, not predictions.
            response = {**raw, "id": ad_id}
            if raw.get("id") != ad_id:
                entry["warning"] = "Used input ad ID instead of missing/mismatched model ID"
                print(f"Warning for {ad_id}: {entry['warning']}", flush=True)
            response = validate_response(response, ad_id)
            entry["response"] = response
            return response, elapsed
        except ValueError as error:
            entry["validation_error"] = str(error)
            if attempt == 1:
                raise
            print(f"Invalid annotation for {ad_id}: {error}. Rechecking images once.", flush=True)
            request = ad_row.copy()
            request["validation_feedback"] = (
                f"\nThe previous response failed validation: {error}. "
                "Reinspect the supplied images and return one complete corrected annotation. "
                "Check product/business categories BEFORE deciding relevance. RETAILER is restricted "
                "to food/beverage retailers, not all shops. Do not change relevance to YES merely "
                "to resolve a contradiction; correct whichever classification the images do not support. "
                "If relevant, fill all required detail fields; otherwise follow the skip rules."
            )



def classify_food_ads(food_ads_df, model, processor, output_path, responses_path):
    """Loop through all food ads and classify them."""

    if food_ads_df.duplicated(["enrol_number", "id", "start_frame"]).any():
        raise ValueError("Duplicate ad-group IDs in Round 2 input; repair Round 1 intervals first")
    # load existing results if any (to skip already processed ads)
    if os.path.exists(output_path):
        existing = pd.read_excel(output_path, dtype={"enrol_number": str})

        # keep only successful rows, since failed ads must be attempted again
        if "error" in existing.columns:
            failed_mask = (existing["error"].fillna("").astype(str).str.strip().ne(""))
            print(f"Found {failed_mask.sum()} failed ads to retry.")
            existing = existing.loc[~failed_mask].copy()

        if existing["ad_id"].duplicated().any():
            raise ValueError("Duplicate saved ad-group IDs; review results before resuming")
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

        fields = ("overall_category", "product_categories", "brand_business_categories", "ad_relevance",
                  "brand_main", "brand_other", "type_of_marketing", "marketing_strategies", "target_group",
                  "who_category", "nova_category", "confidence")
        result_entry = dict.fromkeys(fields)
        result_entry.update(ad_id=ad_id, enrol_number=str(ad_row["enrol_number"]), clip_id=ad_row["id"],
                            platform=ad_row.get("platform", ""), start_frame=ad_row["start_frame"],
                            end_frame=ad_row["end_frame"], response_time=None, confidence=0.0,
                            ad_screenshot_paths=ad_row["ad_screenshot_paths"])
        try:
            if model is None:
                model, processor = load_annotation_model()
            response, response_time = checked_round2(model, processor, ad_row, ad_id, all_responses)

            result_entry.update({field: response[field] for field in fields})
            result_entry["response_time"] = round(response_time, 2)

        except Exception as e:
            import traceback
            print(f"Error processing ad {ad_id}: {e}")
            print(traceback.format_exc())
            all_responses.append({"ad_id": ad_id, "error": str(e)})
            result_entry["error"] = str(e)

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

    errors = int(final.get("error", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").sum())
    unclear = int(final.get("ad_relevance", pd.Series(dtype=str)).eq("UNSURE").sum())
    print(f"\nRound 2: {len(final) - errors} successful results ({unclear} unsure), {errors} errors.")
    print(f"Results saved to {output_path}")

    return final, all_responses


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    output_path = os.path.join(AGGREGATED_DIR, "round2_results.xlsx")
    responses_path = os.path.join(AGGREGATED_DIR, "round2_responses.json")

    # load food ads
    all_ads_path = os.path.join(AGGREGATED_DIR, "all_ads_round2.xlsx") # all ads
    if not os.path.exists(all_ads_path):
        print(f"ERROR: {all_ads_path} not found. Run aggregate.py first.")
        exit(1)

    all_ads = pd.read_excel(all_ads_path, dtype={"enrol_number": str})
    if "frame_bounds_version" not in all_ads.columns or not all_ads["frame_bounds_version"].eq(FRAME_BOUNDS_VERSION).all():
        raise ValueError("Rerun Round 1 with the inclusive-end prompt, then aggregate.py, before Round 2")
    print(f"Loaded {len(all_ads)} ads for round 2 classification.")

    if all_ads.empty:
        print("No ads to annotate.")
        raise SystemExit(0)

    # run round 2
    results, responses = classify_food_ads(all_ads, None, None, output_path, responses_path)

    if results.get("error", pd.Series(dtype=str)).fillna("").astype(str).str.strip().ne("").any():
        raise SystemExit("Round 2 has failed ads; rerun the same command to retry them.")
