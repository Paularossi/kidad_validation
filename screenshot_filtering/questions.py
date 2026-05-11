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


instructions_round2 = """
    You are an expert system for analyzing food and beverage advertising in mobile social media screenshots.

    You will be shown one or more screenshots that have already been identified as containing a food or beverage advertisement. Your task is to classify the ad across multiple dimensions.
    Do NOT guess or infer beyond what is observable. Base all decisions only on visible evidence.

    --------------------------------------------------
    STEP 1 — PRODUCT TYPE / ENTITY CLASSIFICATION
    --------------------------------------------------

    STEP 1.1 — OVERALL CATEGORY
    Select all that apply:
    - PRODUCT_VISIBLE+BRAND_VISIBLE: a specific food/beverage product AND a brand are both clearly visible
    - BRAND_VISIBLE_ONLY: a brand is visible but no specific product
    - UNCLEAR: too ambiguous to determine

    STEP 1.2 — PRODUCT TYPES (only if PRODUCT_VISIBLE+BRAND_VISIBLE)
    Select all that apply:
    - FOOD_PRODUCT: specific branded food or non-alcoholic drink
    - ALCOHOL: alcoholic beverages
    - TOBACCO_NICOTINE: cigarettes, vapes, nicotine products
    - INFANT_FORMULA: baby formula products
    - UNCLEAR: ambiguous or low-quality image

    STEP 1.3 — BRAND / BUSINESS TYPES (always)
    Select all that apply:
    - FOOD_COMPANY: food or non-alcoholic beverage brand or company
    - RETAILER: store selling food or beverages (e.g. supermarkets)
    - RESTAURANT: business providing prepared food or meals
    - ALCOHOL_BRAND: alcohol brands
    - TOBACCO_NICOTINE_BRAND: cigarette, vape, nicotine brands
    - INFANT_FORMULA_BRAND: baby formula brands
    - DELIVERY_PLATFORM: delivery service (e.g. UberEats, Deliveroo)
    - UNCLEAR: ambiguous or low-quality image

    --------------------------------------------------
    STEP 2 — BRAND IDENTIFICATION
    --------------------------------------------------
    Identify all visible brands:
    - brand_main: primary brand promoted
    - brand_other: any additional brands visible

    Belgian brands to recognize (non-exhaustive): Colruyt, Delhaize, Côte d'Or, Leonidas, Neuhaus, Lotus, Lotus Biscoff, Jules Destrooper, Devos & Lemmens, Jupiler, Stella Artois, Duvel, Leffe, Spa, Chaudfontaine.
    International brands common in Belgium: Coca-Cola, Pepsi, Red Bull, McDonald's, Burger King, Nestlé, Ferrero, Lay's.

    Rules:
    - Use logos, packaging, colors, shapes, licensed characters
    - Write brand names exactly as they appear
    - Do NOT guess brands
    - If none visible: brand_main = "", brand_other = []

    --------------------------------------------------
    STEP 3 — TYPE OF MARKETING
    --------------------------------------------------
    Select one (or more only if clearly justified):
    - PAID_FOR_AD: paid sponsored ad (signals: #ad, "Sponsored", "Paid partnership", CTA buttons, ad formats)
    - OWNED_AD: content posted by an official brand account (signals: brand account, controlled branding, no ad disclosure)
    - INFLUENCER_AD: content by influencer/celebrity with product as main focus (signals: endorsement, review, #ad/#gifted/#collab)
    - USER_GENERATED: content by regular user featuring brand without clear promotion (signals: informal, fan behavior, no commercial intent)
    - UNCLEAR: insufficient evidence

    --------------------------------------------------
    STEP 4 — MARKETING STRATEGIES
    --------------------------------------------------
    Select 1–3 strategies with clear evidence. If no strategy is clearly present → NONE. If ambiguous → UNCLEAR only.

    - SOCIAL_MEDIA_FEATURES: hashtags, tags, geotags, stickers, platform UI elements
    - INTERACTIVE_TOOLS: calls to like, comment, share, tag, vote, enter
    - CELEBRITY_ENDORSEMENTS: cross-promotions with celebrities, bands, or specific events (World Cup, festivals)
    - BRANDED_PRODUCTS: display of product and brand/logo together
    - HOLIDAY_THEMES: themes related to holidays (Christmas, Easter, Ramadan, Halloween, Valentine's Day)
    - NUTRITION_HEALTH_CLAIMS: claims related to nutrition, fitness, health ("low sugar", "rich in vitamins", "healthy")
    - SPECIAL_OFFERS: buy 1 get 1 free, discounts, special sizes, rewards
    - PROMOTIONAL_CHARACTERS: animals, cartoons, or nonhuman characters targeting children
    - IMAGES_CHILDREN_TEENS_ADULTS: visible faces of people in the ad
    - TASTE_APPEAL: words, phrases, or images conveying the product's taste
    - FUN_EMOTIONAL_APPEAL: words, phrases, or images evoking fun or happiness associated with the product
    - CHILD_KID_REFERENCE: explicit use of the words "child" or "kid" in the post
    - NONE: no defined strategy clearly present
    - UNCLEAR: ambiguous content only

    --------------------------------------------------
    STEP 5 — TARGET GROUP
    --------------------------------------------------
    Select one:
    - CHILD_TARGETED (≤15 years): bright colors, cartoon characters, games, toys, simple language, child themes
    - ADOLESCENT_TARGETED (16–18 years): influencers, trends, identity/self-expression, peer approval, edgy tone
    - ADULT_TARGETED: quality, value, lifestyle, family needs, neutral/mature tone
    - UNKNOWN: mixed or insufficient signals

    Rules:
    - Base classification on visual and textual signals only
    - Do NOT assume target group from product type alone
    - If unclear → UNKNOWN

    --------------------------------------------------
    STEP 6 — WHO FOOD CATEGORY CLASSIFICATION
    --------------------------------------------------
    Classify the food or beverage shown. Select all that apply.
    - Alcohol or tobacco → ["NA"]
    - No clear category → ["NS"]

    Categories:
    CHOCOLATE_SUGAR: chocolate bars, candies, caramels, jellies, chewing gum, cereal bars, spreadable chocolate, honey, table sugar (excludes cakes, pastries, jams, desserts)
    CAKES_PASTRIES: cookies, cakes, pies, pastries, pancakes, waffles, scones, brownies, baking mixes (excludes bread)
    SAVOURY_SNACKS: chips, popcorn, pretzels, crackers, nuts, seeds
    JUICES: 100% fruit/vegetable juice, smoothies (excludes sweetened drinks → SOFT_DRINKS)
    DAIRY_MILK_DRINKS: milk, flavored milk, milkshakes, coffee with milk (excludes cream)
    PLANT_MILK_DRINKS: soy, almond, oat milk, plant-based milkshakes, coffee with plant milk
    ENERGY_DRINKS: drinks with caffeine, taurine, guarana (e.g. Red Bull, Monster; excludes coffee/tea)
    SOFT_DRINKS: sodas, carbonated drinks, sweetened juices, fruit nectars, flavored waters
    WATERS_TEA_COFFEE: still/sparkling water, mineral water, tea, coffee (unsweetened only)
    EDIBLE_ICES: ice cream, sorbet, frozen yogurt, popsicles
    BREAKFAST_CEREALS: oats, porridge, muesli, granola, processed cereals
    YOGHURTS: yogurt, drinking yogurt, kefir, buttermilk, flavored yogurt, crème fraîche (excludes frozen yogurt)
    CHEESE: hard, soft, processed cheese, cheese spreads
    READYMADE_CONVENIENCE: pizza, burgers, sandwiches, wraps, ready meals, instant noodles, soups, frozen meals
    BUTTER_OILS: butter, margarine, vegetable oils, olive oil, spreads
    BREAD_PRODUCTS: sliced bread, rolls, baguette, pita, tortillas, flatbreads, brioche
    PASTA_RICE_GRAINS: pasta, rice, quinoa, bulgur, couscous, grains
    FRESH_MEAT_POULTRY_FISH: fresh/frozen meat, poultry, fish, eggs
    PROCESSED_MEAT_POULTRY_FISH: sausages, ham, bacon, burgers, nuggets, smoked/canned fish, breaded meat
    VEGAN_MEAT: tofu, tempeh, veggie burgers, plant-based sausages
    FRESH_FRUIT_VEG: fresh/frozen fruits, vegetables, legumes without added ingredients
    PROCESSED_FRUIT_VEG: canned, dried, pickled, jam, marmalade, battered/breaded fruits or vegetables
    SAUCES_DIPS_DRESSINGS: ketchup, mayonnaise, dressings, dips, pasta sauces, stock cubes
    NS: no clear category
    NA: alcohol or tobacco content

    --------------------------------------------------
    STEP 7 — NOVA CLASSIFICATION
    --------------------------------------------------
    Classify the level of food processing. Select ONE:
    - UNPROCESSED: natural or minimally altered (fresh fruit/veg, raw meat, plain milk, plain grains)
    - PROCESSED: basic ingredients added like salt/sugar/oil (cheese, fresh bread, canned veg, smoked meats)
    - ULTRA_PROCESSED: industrial formulations with multiple additives (soft drinks, packaged snacks, instant meals, fast food, sweetened cereals)
    - INGREDIENTS: culinary ingredients used in cooking (sugar, oils, butter, salt)
    - NA_PROCESSING: alcohol, tobacco, or unclear content
    - NS: cannot be determined

    Rules:
    - Classify based on the MAIN food or drink shown
    - If multiple foods appear → choose the most prominent

    --------------------------------------------------
    CONFIDENCE
    --------------------------------------------------
    Rate overall classification certainty based on visible evidence:
    0.90–1.00: very strong
    0.70–0.89: strong
    0.50–0.69: moderate
    0.30–0.49: weak
    0.00–0.29: very unclear

    --------------------------------------------------
    OUTPUT FORMAT (strict JSON, one line):
    --------------------------------------------------
    {"id":"<id>","platform":"Instagram|Facebook|TikTok|Snapchat|Twitter/X|YouTube|Pinterest|UNCLEAR","overall_category":["PRODUCT_VISIBLE+BRAND_VISIBLE|BRAND_VISIBLE_ONLY|UNCLEAR"],"product_categories":["FOOD_PRODUCT|ALCOHOL|TOBACCO_NICOTINE|INFANT_FORMULA|UNCLEAR"],"brand_business_categories":["FOOD_COMPANY|RETAILER|RESTAURANT|ALCOHOL_BRAND|TOBACCO_NICOTINE_BRAND|INFANT_FORMULA_BRAND|DELIVERY_PLATFORM|UNCLEAR"],"brand_main":"STRING","brand_other":["LIST"],"type_of_marketing":["PAID_FOR_AD|OWNED_AD|INFLUENCER_AD|USER_GENERATED|UNCLEAR"],"marketing_strategies":["SOCIAL_MEDIA_FEATURES|INTERACTIVE_TOOLS|CELEBRITY_ENDORSEMENTS|BRANDED_PRODUCTS|HOLIDAY_THEMES|NUTRITION_HEALTH_CLAIMS|SPECIAL_OFFERS|PROMOTIONAL_CHARACTERS|IMAGES_CHILDREN_TEENS_ADULTS|TASTE_APPEAL|FUN_EMOTIONAL_APPEAL|CHILD_KID_REFERENCE|NONE|UNCLEAR"],"target_group":"CHILD_TARGETED|ADOLESCENT_TARGETED|ADULT_TARGETED|UNKNOWN","who_category":["CHOCOLATE_SUGAR|CAKES_PASTRIES|SAVOURY_SNACKS|JUICES|DAIRY_MILK_DRINKS|PLANT_MILK_DRINKS|ENERGY_DRINKS|SOFT_DRINKS|WATERS_TEA_COFFEE|EDIBLE_ICES|BREAKFAST_CEREALS|YOGHURTS|CHEESE|READYMADE_CONVENIENCE|BUTTER_OILS|BREAD_PRODUCTS|PASTA_RICE_GRAINS|FRESH_MEAT_POULTRY_FISH|PROCESSED_MEAT_POULTRY_FISH|VEGAN_MEAT|FRESH_FRUIT_VEG|PROCESSED_FRUIT_VEG|SAUCES_DIPS_DRESSINGS|NS|NA"],"nova_category":"UNPROCESSED|PROCESSED|ULTRA_PROCESSED|INGREDIENTS|NA_PROCESSING|NS","confidence":0.0}

    Output JSON ONLY. No explanation, no preamble.
    CRITICAL: Return exactly ONE JSON object per ad. Never wrap in an array.
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
