# KidAd annotation pipeline

This project identifies ads in social media screenshots and describes the ads
relevant to the KidAd study: food and drinks, alcohol, tobacco/nicotine, and infant formula, studying food marketing exposure in children and adolescents in Belgium. The pipeline uses [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), a multimodal vision-language model, to classify social media screenshots into ads/non-ads and then annotate detected food ads across multiple dimensions (WHO food categories, NOVA processing level, marketing strategies, target age group, brand identification, etc.).

Round 1 creates video clips from screenshots when needed, reuses existing clips, and detects ads within them. Round 2 reads the screenshots belonging to each detected ad, checks relevance, and completes detailed annotation only when relevance is YES. Prompts and response definitions live in `screenshot_filtering/questions.py`.

## Environment

Run commands from the project root. On DSRI, activate your working environment:

```bash
cd /workspace/persistent/kidad_validation
source .venv/bin/activate
```

For a fresh installation, `bash setup.sh` currently creates the environment at `/workspace/persistent/.venv`; activate that path afterwards instead.

Place your Hugging Face token in a project-root `keys.txt` file:

```json
{"huggingface": "YOUR_TOKEN"}
```

Set these environment variables in the DSRI Deployment so model downloads use persistent storage, including after pod replacement:

```text
HF_HOME=/workspace/persistent/.cache/huggingface
HF_HUB_CACHE=/workspace/persistent/.cache/huggingface/hub
```

Rounds 1 and 2 need the GPU environment. Aggregation and export run locally or on DSRI.

## Input files

Set `DATA_DIR` in `screenshot_filtering/helpers.py` to `os.path.join(BASE_DIR, "data/participants")` for the full dataset, or `os.path.join(BASE_DIR, "data/validation 1")` for validation. This is the only dataset path to change for the three annotation stages. Output folders follow automatically: `data/results_participants/` and `data/aggregated_participants/` for the full dataset, or the existing `data/results_validation/` and `data/aggregated validation/` folders for validation. `--all` processes every numeric participant folder in the selected dataset.

Raw participant folders contain `<screenshot_id>.png` files, with `enrol_number`, `image` and `Time` supplied by `data/metadata.xlsx`. Round 1 creates videos in five-minute capture windows at 2 FPS and saves `clips.xlsx` and `frames.xlsx` beside the screenshots, without copying the images. Existing prepared folders are reused, including the validation sample's `frames/` layout; missing videos are recreated from their recorded screenshots. Missing images or incomplete or inconsistent metadata are reported as errors rather than silently skipped.

## Annotation

Run these commands from the project root, in order:

```bash
python -m screenshot_filtering.round1_vid --all
python -m screenshot_filtering.aggregate
python -m screenshot_filtering.round2
```

**`screenshot_filtering/round1_vid.py`** detects ads in each clip and records their start/end times. It saves clip classifications in `results.xlsx`, detected ad intervals in `ads.xlsx`, and model responses in `responses.json`, within each participant's results folder. Use `--enrol <enrol_number>` instead of `--all` to select participants. Results are saved after each participant. Rerunning reuses prepared clips, skips saved successful clips and retries failed or missing results; the model is loaded only when annotation is needed.

**`screenshot_filtering/aggregate.py`** combines participant results into `all_results.xlsx` and builds `all_ads_round2.xlsx`, which links each detected ad sequence to its original screenshots. It converts model timestamps into frame indices using the same 2 FPS playback rate: the start is included and the end is excluded, while equal start/end times select a single frame. Aggregation rebuilds these tables each time, so run it after updating Round 1 results. Run it on the same system as Round 2 so the generated screenshot paths remain valid.

**`screenshot_filtering/round2.py`** reads `all_ads_round2.xlsx` and assesses every detected ad for study relevance. Relevant ads receive detailed labels for brands, marketing, target group, WHO categories and NOVA; relevance `NO` or `UNSURE` stops detailed annotation. A single screenshot is processed as an image, and multiple screenshots as a video sequence. Results are saved to `round2_results.xlsx` and `round2_responses.json` every ten ads and at completion. Rerunning skips saved successes and retries failures, loading the model only if an ad needs processing. Resume assumes the same model and prompt; use fresh output folders for a deliberate new annotation run.

## Supporting files

**`screenshot_filtering/questions.py`** contains the annotation prompts, label definitions, response schemas and JSON parsing helpers. This is where the two phases' annotation rules are defined, including the relevance stopping rule.

**`screenshot_filtering/helpers.py`** contains screenshot grouping and video creation functions, plus shared Excel and JSON saving functions. It also holds the shared dataset path and preparation settings used by all three stages.

**`example.py`** prints example commands for each pipeline stage and for running a background job on DSRI. Run `python example.py` for all examples or `python example.py round2` for one stage; it does not start annotation.


## Prompts (`screenshot_filtering/questions.py`)
 
### `instructions_video` (Round 1)
Classifies a clip as AD or NON_AD using a two-step approach:
- Step 1: platform detection (Instagram, Facebook, TikTok, Snapchat, etc.)
- Step 2: marketing detection - AD if any clearly identifiable brand/logo is present in any frame. Signals include "Sponsorisé", "Gesponsord", "Publicité", "Paid partnership", #ad, CTA buttons, product placement
- Step 3 (AD only): `start_sec` and `end_sec` for each detected ad in seconds from the video timeline
Output: `{"items":[{"id":"...","platform":"...","label":"AD|NON_AD","confidence":0.0,"signals":[...],"ads":[{"start_sec":0.0,"end_sec":0.0}]}]}`
 
### `instructions_round2` (Round 2)
Detailed classification of a known food/beverage ad across 7 dimensions:
- Step 1: product/entity classification (`overall_category`, `product_categories`, `brand_business_categories`) + ad relevance to the study.
- Step 2: brand identification (`brand_main`, `brand_other`) - Belgian and international brands
- Step 3: type of marketing (PAID_FOR_AD / OWNED_AD / INFLUENCER_AD / USER_GENERATED / UNCLEAR)
- Step 4: marketing strategies (SOCIAL_MEDIA_FEATURES, CELEBRITY_ENDORSEMENTS, HOLIDAY_THEMES, NUTRITION_HEALTH_CLAIMS, SPECIAL_OFFERS, PROMOTIONAL_CHARACTERS, IMAGES_CHILDREN_TEENS_ADULTS, TASTE_APPEAL, FUN_EMOTIONAL_APPEAL, CHILD_KID_REFERENCE, NONE, UNCLEAR)
- Step 5: target group (CHILD_TARGETED ≤15 / ADOLESCENT_TARGETED 16-18 / ADULT_TARGETED / UNKNOWN)
- Step 6: WHO food category (23 categories + NS/NA)
- Step 7: NOVA processing level (UNPROCESSED / PROCESSED / ULTRA_PROCESSED / INGREDIENTS / NA_PROCESSING / NS)
### JSON Parsing (`parse_qwen_json`, `fix_json_with_qwen`)
Handles common model output issues: markdown fence stripping, smart quote normalization, `}}` → `]}` fix. Falls back to a second Qwen call to fix malformed JSON.


## Validation

**`screenshot_filtering/dieticians_video.py`** prepares the dieticians' validation sample from existing clips and original screenshots, using a fixed random seed. It copies the selected videos and frames and writes their metadata; it is a sample preparation script, not a required step for annotating the full dataset.

**`screenshot_filtering/export_dieticians.py`** matches AI results to the dieticians' workbook by participant and screenshot ID, adding AI labels and ad group IDs to a separate copy. It flags missing, failed or ambiguous matches and does not calculate agreement. By default it writes `data/aggregated validation/dieticians_with_ai.xlsx`; use `--output PATH` for a different destination.

```bash
python -m screenshot_filtering.export_dieticians
```

