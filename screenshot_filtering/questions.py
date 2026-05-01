import re
import json

instructions = """
    You are an expert in detecting marketing (including advertising) in mobile social media screenshots. Return ONLY compact JSON on one line.

    Classify each screenshot as AD, NON_AD, or UNCERTAIN.
    Platform UI may show small cues like “Sponsored”, “Promoted”, “Gesponsord”, “Reklame”, or ad disclosures, but you should also rely on visual layout (page/account name with subtle “Sponsored” tag, CTA buttons like “Shop now”, price/discount badges, product hero shots, etc.) and on any text captions.
    Labels:
    - AD: a paid promotion or sponsored/boosted post/ad unit
    - NON_AD: organic content, comments, chats, menus, camera, inbox, reels grid, etc.
    - UNCERTAIN: not enough evidence
    
    For every screenshot, determine which social media platform it belongs to (Instagram, Facebook, TikTok, Snapchat, Twitter/X, etc., or UNKNOWN if unclear).

    If the screenshot is labelled as AD, also determine:
    - Whether the ad promotes food, beverages, or alcohol (YES, NO, or UNSURE).
    - Any recognizable brands shown (list every distinct brand name, or an empty list if none).

    Output format (strict JSON):
    {"items":[{"id":"<id>","label":"AD|NON_AD|UNCERTAIN","platform":"Platform","confidence":0.0,"signals":["..."],"ad_followup":{"food_ad":"YES|NO|UNSURE","brands":["Brand"]}}]}

    Only include "ad_followup" when the label is AD; omit it otherwise. Classify the provided image. Be conservative with AD unless ad-specific signals are visible.
"""

# for first round of annotations (ads vs. non-ads)
instructions_video = """
    You are an expert system for detecting marketing (including advertising and brand presence) in mobile social media video clips. Return ONLY compact JSON on one line.

    You will be shown a short video clip recorded from a mobile social media feed.
    Analyze the FULL clip across ALL frames before deciding.

    --------------------------------------------------
    STEP 1 — PLATFORM DETECTION
    --------------------------------------------------
    Identify the platform from UI cues (layout, icons, caption structure, watermarks, navigation bars):
    - Instagram, Facebook, TikTok, Snapchat, Twitter/X, YouTube, Pinterest, UNCLEAR, NOT_APPLICABLE

    Use NOT_APPLICABLE when the clip is not from any listed platform.
    Use UNCLEAR when the platform cannot be confidently identified.

    --------------------------------------------------
    STEP 2 — MARKETING DETECTION
    --------------------------------------------------
    Classify the clip as AD, NON_AD, MIXED, or UNCERTAIN.

    AD: the entire clip (or dominant portion) is a paid promotion or sponsored/boosted post.
    NON_AD: entirely organic content — no company or brand name or logo is visible anywhere in the clip.
    MIXED: the clip contains both organic content and one or more ads.
    UNCERTAIN: not enough evidence to decide.

    A screenshot is classified as AD if a clearly identifiable company or brand name or logo is present. This includes:
    - Explicit signals: "Sponsored", "Promoted", "Gesponsord", "Reklame", "Paid partnership", #ad
    - Brand presence: logos, packaging, brand identity, product placement
    - Influencer content showing branded products
    - CTA buttons ("Shop now", "Order now"), price/discount badges, product hero shots
    Brand presence alone is SUFFICIENT to classify as AD.
    PIPELINE RULE: If NON_AD → stop, skip Step 3.

    --------------------------------------------------
    STEP 3 — AD CHARACTERISATION (only if AD or MIXED)
    --------------------------------------------------
    If the clip contains ANY ads (label is AD or MIXED), also determine for EACH detected ad:
    - "start_frame": the index of the frame (0-based) where the ad first becomes clearly visible
    - "end_frame": the index of the frame where the ad is no longer visible (or last frame if it continues)
    - Whether the ad promotes food, beverages, or alcohol (YES, NO, or UNSURE)
    - Any recognizable brands shown (list every distinct brand name, or empty list if none, do NOT guess)

    --------------------------------------------------
    OUTPUT FORMAT (strict JSON, one line):
    {"items":[{"id":"<id>","platform":"Instagram|Facebook|TikTok|Snapchat|Twitter/X|YouTube|Pinterest|UNCLEAR|NOT_APPLICABLE","label":"AD|NON_AD|MIXED|UNCERTAIN","confidence":0.0,"signals":["..."],"ads":[{"start_frame":0,"end_frame":0,"food_ad":"YES|NO|UNSURE","brands":["Brand"]}]}]}

    Confidence scale:
    0.90-1.00: very strong evidence
    0.70-0.89: strong evidence  
    0.50-0.69: moderate evidence
    0.30-0.49: weak evidence
    0.00-0.29: very unclear

    Only include "ads" when label is AD or MIXED; omit it otherwise.
    For AD, the "ads" list will typically have one entry spanning most or all of the clip.
    For MIXED, list each distinct ad as a separate entry with its own time range.
    Be conservative — only mark as AD if brand/ad signals are clearly visible in the clip.
    Output JSON ONLY.
    CRITICAL: Return exactly ONE object in the "items" array per clip, regardless of how many ads are found.
    Multiple ads within a clip go in the "ads" array inside that single item — never create multiple items for the same clip ID.
"""


def parse_qwen_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())

    # normalize curly double quotes only 
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    # sometimes qwen outputs a broken json with two }} at the end
    if text.endswith('}}') and not text.endswith('}]}'):
        text = text[:-1] + ']}'

    return json.loads(text)


def fix_json_with_qwen(model, processor, raw_response):
    """Use Qwen to fix malformed JSON."""
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Fix this malformed JSON and return ONLY the corrected valid JSON on one line, nothing else:\n\n{raw_response}"}
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(device)
    
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1500, do_sample=False)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    fixed = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Second attempt to fix JSON with Qwen:\n{fixed}\n")
    fixed = re.sub(r"^```(?:json)?\s*", "", fixed.strip())
    fixed = re.sub(r"\s*```$", "", fixed.strip())
    return json.loads(fixed)
