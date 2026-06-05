# KidAd - Food Advertising Detection Pipeline

Automated pipeline for detecting and classifying food/beverage advertising in mobile social media screenshots. Built for the KidAd research project, studying food marketing exposure in children and adolescents in Belgium.
 
The pipeline uses [Qwen3-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct), a multimodal vision-language model, to classify social media screenshots into ads/non-ads and then annotate detected food ads across multiple dimensions (WHO food categories, NOVA processing level, marketing strategies, target age group, brand identification, etc.).

---



## Prerequisites
### Infrastructure
- **Python 3.8+** with packages listed in `requirements.txt`
- **Local model setup** (Qwen via `Transformers`) 
- **Platform**: DSRI (Data Science Research Infrastructure) from Maastricht University
- **Pod**: JupyterLab with PyTorch on GPU (NVIDIA A100 80GB)

### Setup (run once per GPU session)
 
```bash
source /workspace/persistent/.venv/bin/activate
cd /workspace/persistent/kidad
bash setup.sh
```

`setup.sh` installs system dependencies, creates the virtual environment if needed, installs PyTorch with CUDA 12.8, Flash Attention, and all Python requirements.

**Configure API keys:**
   - Create a `keys.txt` file
   - Add your HuggingFace key
   - Configure local model endpoints


## Pipeline
 
### Run order
 
```bash
# Round 1 - classify clips as AD or NON_AD
nohup python -m round1_vid --all > logs/all.log 2>&1 &
tail -f logs/all.log
 
# Aggregate - combine all participant results
python screenshot_filtering/aggregate.py
 
# Round 2 - classify food ads in detail
nohup python -m round2 > logs/round2.log 2>&1 &
tail -f logs/round2.log
```

---

### Round 1 - Ad Detection (`round1_vid.py`)
 
**Input**: participant screenshots from `data/participants/<enrol_number>/`
 
**What it does**:
1. Loads metadata from `metadata.xlsx` (filters out test accounts starting with "32")
2. Groups screenshots into 5-minute time windows using real capture timestamps
3. Converts each window into an `.mp4` video at 2fps using OpenCV
4. Passes each video to Qwen3-VL-32B with `process_vision_info` and `video_metadata` for text-timestamp alignment, enabling second-level temporal localization of ads
5. Parses JSON response, extracts label/confidence/signals/platform/ads with `start_sec`/`end_sec`
6. Skips clips with fewer than 2 frames (torchvision constraint)
7. Skips already-processed participants on rerun
**Output per participant** (`data/results_videos/<enrol_number>/`):
- `results.xlsx` - one row per clip: `id`, `label` (AD/NON_AD), `confidence`, `signals`, `platform`, `n_ads`, `ads` (JSON list with `start_sec`, `end_sec` per ad), `response_time`, `n_frames`, `start_time`, `end_time`
- `ads.xlsx` - exploded version: one row per detected ad, with `start_sec`, `end_sec` columns flattened
- `frames.xlsx` - clip metadata (video paths, frame counts, timestamps)
- `responses.json` - raw model responses for debugging
- `videos/` - `.mp4` files for manual inspection
**Key design decisions**:
- Uses `process_vision_info` with `return_video_metadata=True` and passes `video_metadata` to `processor()` directly - this enables Qwen3-VL's text-timestamp alignment feature for precise temporal localization
- `PLAYBACK_FPS = 2.0` must match between `screenshots_to_videos` and `qwen_call_video`
- Timestamps in seconds are converted to frame indices via `sec_to_frame(sec) = round(sec * PLAYBACK_FPS)`
---
 
### Aggregation (`screenshot_filtering/aggregate.py`)
 
**Input**: all `results.xlsx` and `ads.xlsx` files from `data/results_videos/`
 
**What it does**:
1. Combines results across all participants
2. Deduplicates and cleans up column naming
3. Filters food/beverage ads (`food_ad = YES or UNSURE`) for round 2
**Output** (`data/aggregated/`):
- `all_results.xlsx` - all clips across all participants
- `all_ads_round2.xlsx` - all detected ads
- `food_ads_round2.xlsx` - food/beverage ads only (input for round 2)
- `failed_clips.xlsx` + `failed_clips_summary.xlsx` - UNCERTAIN clips and failure rates per participant
---
 
### Round 2 - Food Ad Classification (`round2.py`)
 
**Input**: `data/aggregated/food_ads_round2.xlsx`
 
**What it does**:
1. Loads food ads, skips already-processed `ad_id` values (resume support)
2. For each ad, loads frames from `ad_screenshot_paths` (remapped from `participants/` to `food_ads/`)
3. Passes frames to Qwen3-VL-32B with `instructions_round2` prompt
4. Parses JSON response and saves results
5. Saves incrementally every 10 ads to avoid losing progress on crash
**Output** (`data/aggregated/`):
- `round2_results.xlsx` - one row per ad with all classification fields
- `round2_responses.json` - raw model responses
**Output columns**: `ad_id`, `enrol_number`, `clip_id`, `platform`, `start_frame`, `end_frame`, `food_ad_round1`, `brands_round1`, `response_time`, `overall_category`, `product_categories`, `brand_business_categories`, `brand_main`, `brand_other`, `type_of_marketing`, `marketing_strategies`, `target_group`, `who_category`, `nova_category`, `confidence`
 
---

## Model
 
- **Model**: `Qwen/Qwen3-VL-32B-Instruct`
- **VRAM**: ~67GB on A100 80GB
- **Loading**:
```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-32B-Instruct",
    device_map="cuda:0",
    dtype="auto",
    attn_implementation="flash_attention_2",
).eval()
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-32B-Instruct")
```
 
---
 
## Prompts (`screenshot_filtering/questions.py`)
 
### `instructions_video` (Round 1)
Classifies a clip as AD or NON_AD using a two-step approach:
- Step 1: platform detection (Instagram, Facebook, TikTok, Snapchat, etc.)
- Step 2: marketing detection - AD if any clearly identifiable brand/logo is present in any frame. Signals include "Sponsorisé", "Gesponsord", "Publicité", "Paid partnership", #ad, CTA buttons, product placement
- Step 3 (AD only): `start_sec` and `end_sec` for each detected ad in seconds from the video timeline
Output: `{"items":[{"id":"...","platform":"...","label":"AD|NON_AD","confidence":0.0,"signals":[...],"ads":[{"start_sec":0.0,"end_sec":0.0}]}]}`
 
### `instructions_round2` (Round 2)
Detailed classification of a known food/beverage ad across 7 dimensions:
- Step 1: product/entity classification (`overall_category`, `product_categories`, `brand_business_categories`)
- Step 2: brand identification (`brand_main`, `brand_other`) - Belgian and international brands
- Step 3: type of marketing (PAID_FOR_AD / OWNED_AD / INFLUENCER_AD / USER_GENERATED / UNCLEAR)
- Step 4: marketing strategies (SOCIAL_MEDIA_FEATURES, CELEBRITY_ENDORSEMENTS, HOLIDAY_THEMES, NUTRITION_HEALTH_CLAIMS, SPECIAL_OFFERS, PROMOTIONAL_CHARACTERS, IMAGES_CHILDREN_TEENS_ADULTS, TASTE_APPEAL, FUN_EMOTIONAL_APPEAL, CHILD_KID_REFERENCE, NONE, UNCLEAR)
- Step 5: target group (CHILD_TARGETED ≤15 / ADOLESCENT_TARGETED 16-18 / ADULT_TARGETED / UNKNOWN)
- Step 6: WHO food category (23 categories + NS/NA)
- Step 7: NOVA processing level (UNPROCESSED / PROCESSED / ULTRA_PROCESSED / INGREDIENTS / NA_PROCESSING / NS)
### JSON Parsing (`parse_qwen_json`, `fix_json_with_qwen`)
Handles common model output issues: markdown fence stripping, smart quote normalization, `}}` → `]}` fix. Falls back to a second Qwen call to fix malformed JSON.
 
---

## Data
 
- **Participants**: Belgian social media users, screenshots collected via passive mobile monitoring
- **Metadata columns**: `enrol_number`, `image` (filename without extension), `Time` (UTC), `AppId`
- **Test accounts**: filtered out (`enrol_number` starting with "32")
- **Relevant apps**: Instagram, Facebook, TikTok, Snapchat, and related app IDs
- **Screenshot naming**: numeric IDs (e.g. `101022.png`)
- **Capture interval**: irregular - always use real timestamps, never assume fixed interval
- **`enrol_number`**: stored as `str()` to prevent Excel from converting to scientific notation