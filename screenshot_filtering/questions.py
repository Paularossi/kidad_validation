import re
import json

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
    Classify the clip as AD, NON_AD or UNCLEAR.

    AD: the clip contains at least one paid promotion, sponsored/boosted post, brand advertisement or brand exposure — even if only in some frames or in ordinary user-generated content.
    NON_AD: no advertising or identifiable company/brand evidence is visible or explicitly mentioned in readable text anywhere in the content.
    UNCLEAR: not enough evidence to decide (poor image quality, ambiguous brand or logo visibility).

    A clip is classified as AD if an identifiable company or brand name, logo or packaging is visible, or the brand is explicitly mentioned in readable on-screen text, captions or hashtags, in ANY frame. This includes:
    - Explicit signals: "Sponsored", "Promoted", "Gesponsord", "Reklame", "Sponsorisé", "Paid partnership", #ad
    - Brand presence: logos, packaging, brand identity, product placement
    - Influencer content showing branded products
    - Ordinary user-generated content containing branded products, even without promotion or sponsorship disclosure
    - Sponsor logos on football shirts and other sports clothing
    - National, local and international brands; unfamiliarity alone is not evidence of NON_AD
    - CTA buttons ("Shop now", "Order now"), price/discount badges, product hero shots
    Brand presence alone is SUFFICIENT to classify as AD.
    Payment, promotional intent and a sponsorship label are NOT required.
    Small, blurry or partially visible branding counts when identifiable. Do not guess a brand from ambiguous colors or shapes alone. Readable brand mentions count even when packaging cannot be identified.
    Inspect the content and readable captions throughout the clip, including its first and last frames. Platform navigation logos alone do not count.
    PIPELINE RULE: If NON_AD or UNCLEAR → stop, skip Step 3.

    --------------------------------------------------
    STEP 3 — AD CHARACTERISATION (only if AD)
    --------------------------------------------------
    If the clip contains ANY ads, determine for EACH detected ad:
    - "start_sec": the playback timestamp of the FIRST frame containing the ad/brand exposure.
    - "end_sec": the playback timestamp of the LAST frame still containing that same exposure.
    BOTH endpoints are INCLUDED. Do not report the first frame after the ad disappears.
    A single-frame exposure has start_sec = end_sec. For a one-image clip, both are 0.0.
    The supplied timing information gives the number of original frames, FPS and last valid frame timestamp. If the exposure continues to the end, use that last frame timestamp, NOT the video duration.
    Example at 2 FPS: an exposure in frames at 41.0, 41.5, 42.0 and 42.5 seconds must return start_sec=41.0 and end_sec=42.5.
    Continue the interval while the same exposure remains visible or mentioned in readable text, including during scrolling or partial occlusion when its identity is supported by the content. Do not end it solely because a Sponsored label scrolls away.
    Separate different posts/advertisements into different intervals, even when adjacent. Do not combine a fitness ad and a food ad into one interval. Do not bridge frames with no evidence that the exposure continues.

    CRITICAL: Use the supplied video playback timeline and frame spacing, never the phone clock or original capture times. Preserve fractional seconds; do not round to whole seconds. Each endpoint identifies an included frame.

    --------------------------------------------------
    OUTPUT FORMAT (strict JSON, one line):
    {"items":[{"id":"<id>","platform":"Instagram|Facebook|TikTok|Snapchat|Twitter/X|YouTube|Pinterest|UNCLEAR|NOT_APPLICABLE","label":"AD|NON_AD|UNCLEAR","confidence":0.0,"ads":[{"start_sec":0.0,"end_sec":0.0}],"signals":["..."]}]}

    Only include "ads" when label is AD; omit it otherwise.
    Write all ad intervals BEFORE "signals". Put "signals" last and include at most THREE short pieces of observed evidence, each at most 12 words.
    Do not produce exhaustive brand/product lists or enumerate possible brand/product combinations. The signal limit does not limit the number of ad intervals.
    If the user requests signals = [], return an empty signals array without explanations; still inspect all frames and return all detected ad intervals.
    For AD with multiple distinct ads, list each as a separate entry in the "ads" array with its own start_sec and end_sec. AD requires at least one interval.
    Brand/ad evidence visible or explicitly mentioned in readable text in at least one frame is sufficient. Keep uncertainty about the exact brand identity separate from evidence that advertising is present.
    Output JSON ONLY.
    CRITICAL: Return exactly ONE object in the "items" array per clip, regardless of how many ads are found.
    Multiple ads within a clip go in the "ads" array inside that single item — never create multiple items for the same clip ID.
"""


instructions_round2 = """
    You are an expert system for analyzing advertising in mobile social media screenshots.

    You will be shown one or more screenshots that have already been identified as an advertisement. Your task is to classify the ad across multiple dimensions.
    Do NOT guess or infer beyond what is observable. Base decisions on the supplied images and readable on-screen text, captions and hashtags. Review ALL supplied frames, including the last one.
    AD includes brand exposure in ordinary user-generated content and sports sponsorship. Do not exclude an exposure because it is organic or lacks a paid-ad disclosure.

    --------------------------------------------------
    STEP 1 — PRODUCT TYPE / ENTITY CLASSIFICATION
    --------------------------------------------------

    STEP 1.1 — OVERALL CATEGORY
    Select all that apply:
    - PRODUCT_VISIBLE+BRAND_VISIBLE: a specific product is visible AND its brand is identifiable visually or explicitly mentioned in readable text
    - BRAND_VISIBLE_ONLY: a brand is identifiable visually or explicitly mentioned in readable text, but no specific product is visible
    - UNCLEAR: too ambiguous to determine
    - NOT_APPLICABLE: ad not including neither a product nor a brand

    STEP 1.2 — PRODUCT TYPES (only if PRODUCT_VISIBLE+BRAND_VISIBLE)
    Select all that apply:
    - FOOD_PRODUCT: specific branded food or non-alcoholic drink
    - FOOD_SUPPLEMENTS: specific branded food supplements or vitamins (e.g., protein powders/drinks such as whey and casein; recovery carbohydrate/protein mixtures; amino acids such as BCAA and glutamine; minerals such as calcium; caffeine capsules, tablets, gels and shots; creatine; L-carnitine; beta-alanine; nitrate/beetroot juice; and other supplements)
    - ALCOHOL: alcoholic beverages
    - TOBACCO_NICOTINE: cigarettes, vapes, nicotine products
    - INFANT_FORMULA: baby formula products
    - UNCLEAR: ambiguous or low-quality image
    - NOT_APPLICABLE: product visible but unrelated to food, beverages, food supplements/vitamins, alcohol, tobacco/nicotine, or infant formula (e.g., iPhone, jeans, facial cream)

    STEP 1.3 — BRAND / BUSINESS TYPES (always)
    Select all that apply:
    - FOOD_COMPANY: food or non-alcoholic beverage brand or company
    - FOOD_SUPPLEMENTS_BRAND: food supplements or vitamin brand or company
    - RETAILER: store selling food or beverages (e.g. supermarkets)
    - RESTAURANT: business providing prepared food or meals
    - ALCOHOL_BRAND: alcohol brands
    - TOBACCO_NICOTINE_BRAND: cigarette, vape, nicotine brands
    - INFANT_FORMULA_BRAND: baby formula brands
    - DELIVERY_PLATFORM: delivery service (e.g. UberEats, Deliveroo)
    - UNCLEAR: ambiguous or low-quality image
    - NOT_APPLICABLE: business/logo visible but unrelated to food, beverages, food supplements/vitamins, alcohol, tobacco/nicotine, or infant formula (e.g., Zara, Sephora)

    STEP 1.4 — AD RELEVANCE (always)
    Is this a relevant ad for this study?
    - YES: at least one relevant product or business category was selected in 1.2 or 1.3 (any category other than UNCLEAR or NOT_APPLICABLE).
    - UNSURE: no relevant category was identified and at least one category is UNCLEAR.
    - NO: no relevant category was identified and all assessed categories are NOT_APPLICABLE.

    When question 1.2 is skipped because no product is visible, return product_categories = []; a skipped question is not NOT_APPLICABLE.
    PIPELINE RULE: If AD RELEVANCE = NO or UNSURE → stop, skip Steps 3-7.
    For skipped Steps 3-7, return empty strings for target_group, nova_category and empty lists for brand_other, type_of_marketing, marketing_strategies, who_category. Still return brand_main, brand_other and confidence.

    --------------------------------------------------
    STEP 2 — BRAND IDENTIFICATION (always)
    --------------------------------------------------
    Identify all brands visible or explicitly mentioned in readable text (international, national and local):
    - brand_main: primary brand promoted
    - brand_other: any additional brands visible or explicitly mentioned

    Rules:
    - Use readable names, logos, packaging and text mentions. Colors, shapes and licensed characters are supporting evidence, not sufficient grounds for guessing.
    - Write brand names exactly as they appear
    - Do NOT guess brands
    - If no brand can be identified visually or from readable text: brand_main = "", brand_other = []

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
    - Alcohol, tobacco or FOOD_SUPPLEMENTS alone → ["NA"]. Do not force supplements into a WHO food category.
    - If ordinary food/drinks also appear, classify those separately; omit supplements from this field.
    - No clear category → ["NS"]

    Categories:
    CHOCOLATE_SUGAR: chocolate bars, candies, caramels, jellies, chewing gum, cereal bars, spreadable chocolate, honey, table sugar (excludes cakes, pastries, jams, desserts)
    CAKES_PASTRIES: cookies, cakes, pies, pastries, pancakes, waffles, scones, brownies, baking mixes (excludes bread)
    SAVOURY_SNACKS: chips, popcorn, pretzels, crackers, nuts, seeds
    JUICES: 100% fruit/vegetable juice, smoothies (excludes sweetened drinks → SOFT_DRINKS)
    DAIRY_MILK_DRINKS: milk, flavored milk, milkshakes, coffee with milk (excludes cream)
    PLANT_MILK_DRINKS: soy, almond, oat milk, plant-based milkshakes, coffee with plant milk
    ENERGY_DRINKS: drinks with caffeine, taurine, guarana (e.g. Red Bull, Monster; excludes coffee/tea and sports drinks)
    SOFT_DRINKS: sodas, carbonated drinks, sweetened juices, fruit nectars, flavored waters
    SPORTS_DRINKS: isotonic drinks, electrolyte drinks, recovery drinks (excludes energy drinks)
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
    NA: only alcohol, tobacco or food supplements are shown; no ordinary food/beverage to classify

    --------------------------------------------------
    STEP 7 — NOVA CLASSIFICATION
    --------------------------------------------------
    If WHO category is ["NA"], skip processing classification and return nova_category = "NA_PROCESSING".
    For supplement-only ads, return who_category = ["NA"] and nova_category = "NA_PROCESSING".
    If ordinary food/drinks also appear, classify those; do not classify the supplements.
    Otherwise classify the level of food processing. Select ONE:
    - UNPROCESSED: natural or minimally altered (fresh fruit/veg, raw meat, plain milk, plain grains)
    - PROCESSED: basic ingredients added like salt/sugar/oil (cheese, fresh bread, canned veg, smoked meats)
    - ULTRA_PROCESSED: industrial formulations with multiple additives (soft drinks, packaged snacks, instant meals, fast food, sweetened cereals)
    - INGREDIENTS: culinary ingredients used in cooking (sugar, oils, butter, salt)
    - NA_PROCESSING: alcohol, tobacco, food supplements, or unclear content
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
    {"id":"<id>","platform":"Instagram|Facebook|TikTok|Snapchat|Twitter/X|YouTube|Pinterest|UNCLEAR","overall_category":["PRODUCT_VISIBLE+BRAND_VISIBLE|BRAND_VISIBLE_ONLY|UNCLEAR|NOT_APPLICABLE"],"product_categories":["FOOD_PRODUCT|FOOD_SUPPLEMENTS|ALCOHOL|TOBACCO_NICOTINE|INFANT_FORMULA|UNCLEAR|NOT_APPLICABLE"],"brand_business_categories":["FOOD_COMPANY|FOOD_SUPPLEMENTS_BRAND|RETAILER|RESTAURANT|ALCOHOL_BRAND|TOBACCO_NICOTINE_BRAND|INFANT_FORMULA_BRAND|DELIVERY_PLATFORM|UNCLEAR|NOT_APPLICABLE"],"ad_relevance": "YES|NO|UNSURE","brand_main":"STRING","brand_other":["LIST"],"type_of_marketing":["PAID_FOR_AD|OWNED_AD|INFLUENCER_AD|USER_GENERATED|UNCLEAR"],"marketing_strategies":["SOCIAL_MEDIA_FEATURES|INTERACTIVE_TOOLS|CELEBRITY_ENDORSEMENTS|HOLIDAY_THEMES|NUTRITION_HEALTH_CLAIMS|SPECIAL_OFFERS|PROMOTIONAL_CHARACTERS|IMAGES_CHILDREN_TEENS_ADULTS|TASTE_APPEAL|FUN_EMOTIONAL_APPEAL|CHILD_KID_REFERENCE|NONE|UNCLEAR"],"target_group":"CHILD_TARGETED|ADOLESCENT_TARGETED|ADULT_TARGETED|UNKNOWN","who_category":["CHOCOLATE_SUGAR|CAKES_PASTRIES|SAVOURY_SNACKS|JUICES|DAIRY_MILK_DRINKS|PLANT_MILK_DRINKS|ENERGY_DRINKS|SOFT_DRINKS|SPORTS_DRINKS|WATERS_TEA_COFFEE|EDIBLE_ICES|BREAKFAST_CEREALS|YOGHURTS|CHEESE|READYMADE_CONVENIENCE|BUTTER_OILS|BREAD_PRODUCTS|PASTA_RICE_GRAINS|FRESH_MEAT_POULTRY_FISH|PROCESSED_MEAT_POULTRY_FISH|VEGAN_MEAT|FRESH_FRUIT_VEG|PROCESSED_FRUIT_VEG|SAUCES_DIPS_DRESSINGS|NS|NA"],"nova_category":"UNPROCESSED|PROCESSED|ULTRA_PROCESSED|INGREDIENTS|NA_PROCESSING|NS","confidence":0.0}

    Output JSON ONLY. No explanation, no preamble.
    CRITICAL: Return exactly ONE JSON object per ad. Never wrap in an array.
"""


def parse_qwen_json(text):
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)



def fix_json_with_qwen(model, processor, raw_response):
    """Use Qwen to fix malformed JSON."""
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": f"Repair only the JSON syntax; preserve all IDs, classifications and intervals. Return ONLY the corrected valid JSON on one line:\n\n{raw_response}"}
        ]}
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(device)

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=1500, do_sample=False)

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    fixed = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(f"Second attempt to fix JSON with Qwen:\n{fixed}\n")
    return parse_qwen_json(fixed)


def parse_model_response(model, processor, raw_response):
    """Parse once; use model repair only for invalid JSON and retain failed output."""
    try:
        return parse_qwen_json(raw_response)
    except json.JSONDecodeError as error:
        print(f"JSON parse error: {error}; attempting syntax repair...")
        try:
            return fix_json_with_qwen(model, processor, raw_response)
        except Exception as repair_error:
            print(f"JSON repair failed: {repair_error}")
            return raw_response
