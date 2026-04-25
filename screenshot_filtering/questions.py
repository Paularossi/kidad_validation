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
    Classify the clip as AD, NON_AD, or UNCLEAR.

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
    - "start_sec": the second in the video where the ad first becomes clearly visible
    - "end_sec": the second in the video where the ad is no longer visible (or end of clip if it continues)
    - Whether the ad promotes food, beverages, or alcohol (YES, NO, or UNSURE)
    - Any recognizable brands shown (list every distinct brand name, or empty list if none, do NOT guess)

    --------------------------------------------------
    OUTPUT FORMAT (strict JSON, one line):
    {"items":[{"id":"<id>","platform":"Instagram|Facebook|TikTok|Snapchat|Twitter/X|YouTube|Pinterest|UNCLEAR|NOT_APPLICABLE","label":"AD|NON_AD|MIXED|UNCERTAIN","confidence":0.0,"signals":["..."],"ads":[{"start_sec":0,"end_sec":0,"food_ad":"YES|NO|UNSURE","brands":["Brand"]}]}]}

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
"""


def parse_qwen_json(text: str):
    # remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def process_first_output(response):

    AD_KEYWORDS = re.compile(r"(sponsored|promoted|gesponsord|advertentie|publicité)", re.I)

    if isinstance(response, list):
        response = response[0]
    
    item = response["items"][0]
    signals = item.get("signals", [])
    label = item["label"]
    platform = item["platform"]
    confidence = item["confidence"]
    ad_followup = item.get("ad_followup")

    result_entry = {
        "id": item["id"],
        "label": label,
        "platform": platform,
        "confidence": confidence,
        "signals": signals
    }

    if isinstance(ad_followup, dict):
        for key, value in ad_followup.items():
            if key not in result_entry:
                result_entry[key] = value

    # check for ad keywords in signals
    if any(AD_KEYWORDS.search(signal) for signal in signals):
        label_new = "AD"
        confidence_new = max(confidence, 0.8)  # Boost confidence if ad keywords are found
        
        # add the new label and confidence to the results
        result_entry["label"] = label_new
        result_entry["confidence"] = confidence_new

    return result_entry