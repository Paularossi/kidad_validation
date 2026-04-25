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

instructions_video = """
    You are an expert in detecting marketing (including advertising) in mobile social media video clips. Return ONLY compact JSON on one line.

    You will be shown a short video clip recorded from a mobile social media feed.
    Analyze the FULL clip across all frames before deciding.

    Classify the clip as AD, NON_AD, UNCERTAIN, or MIXED.
    Labels:
    - AD: the entire clip (or dominant portion) is a paid promotion or sponsored/boosted post
    - NON_AD: entirely organic content (comments, chats, menus, reels grid, etc.)
    - MIXED: the clip contains both organic content and one or more ads
    - UNCERTAIN: not enough evidence to decide

    Platform UI may show cues like "Sponsored", "Promoted", "Gesponsord", "Reklame", or ad disclosures — but also use:
    - Visual layout: account/page name with subtle "Sponsored" tag, CTA buttons (e.g. "Shop now"), price/discount badges, product hero shots
    - Text overlays and captions appearing at any point in the clip

    For every clip, determine which social media platform it belongs to (Instagram, Facebook, TikTok, Snapchat, Twitter/X, etc., or UNKNOWN if unclear).

    If the clip contains ANY ads (label is AD or MIXED), also determine for EACH detected ad:
    - "start_sec": the second in the video where the ad first becomes clearly visible
    - "end_sec": the second in the video where the ad is no longer visible (or end of clip if it continues)
    - Whether the ad promotes food, beverages, or alcohol (YES, NO, or UNSURE)
    - Any recognizable brands shown (list every distinct brand name, or empty list if none)

    Output format (strict JSON):
    {"items":[{"id":"<id>","label":"AD|NON_AD|MIXED|UNCERTAIN","platform":"Platform","confidence":0.0,"signals":["..."],"ads":[{"start_sec":0,"end_sec":0,"food_ad":"YES|NO|UNSURE","brands":["Brand"]}]}]}

    Only include "ads" when the label is AD or MIXED; omit it otherwise.
    For AD, the "ads" list will typically have one entry spanning most or all of the clip.
    For MIXED, list each distinct ad as a separate entry with its own time range.
    Be conservative — only mark as AD if ad-specific signals are clearly visible in the clip.
"""


def parse_qwen_json(text: str):
    # remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    text = text.strip()

    # try a direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # fall back: find the outermost JSON object or array using a stack-based bracket matcher
    for open_char, close_char in [('{', '}'), ('[', ']')]:
        start = text.find(open_char)
        if start == -1:
            continue
        depth = 0
        end = -1
        in_string = False
        escape_next = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escape_next:
                escape_next = False
                continue
            if ch == '\\' and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Could not extract valid JSON from model response: {text[:200]!r}")


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