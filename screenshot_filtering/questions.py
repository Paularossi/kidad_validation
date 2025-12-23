import re
import json

instructions = """
    You are an expert in detecting advertisements in mobile social media screenshots. Return ONLY compact JSON on one line.

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