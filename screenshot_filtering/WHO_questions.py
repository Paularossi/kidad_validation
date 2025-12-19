import re

PATTERN_AYA = re.compile(r"\*{1,2}(.*?)\*{1,2}:\s*([^\n–-]+?)\s*[–-]\s*(.*?)(?=\n\*|$)", re.DOTALL) 

instructions = (
    "You will be provided with a picture of an online advertisement delivered to Belgium/Netherlands, its corresponding text (which may be in English, French, or Dutch), and the name of the company running the ad. "
    "You will be given sets of questions about various aspects of the advertisement along with definitions and examples. "
    "Please answer each question using the exact format: *QUESTION_LABEL*: Yes/No - Brief explanation. Do not include any extra text, greetings, or commentary. "
    "For example, your answers should look like: *CARTOON*: Yes/No - explanation; *CELEBRITY*: Yes/No - explanation; and so on. Ensure that the question label is between a set of stars. "
    "Ensure that each answer includes a brief explanation of the features in the image/text that led to your choice. Ensure that you answer all questions. "
    "You will also be provided with a set of definitions which you should refer to when answering: "
    "1. Food/drink manufacturing company or brand – a company or brand involved in producing and processing foods or beverages. Manufacturers focus on creating and packaging of consumable goods rather than selling directly to consumers. This category excludes restaurant/takeaway/delivery companies or brands and food retailers. "
    "2. Food/drink retailer – a company or brand that sells food and drink products directly to consumers for home consumption (e.g., supermarkets, grocery stores, convenience stores and specialty food shops). These retailers primarily serve as intermediaries, providing a variety of products from different manufacturers to the end consumer. This category excludes manufacturing companies and restaurant/takeaway/delivery companies or brands. "
    "3. Restaurant/takeaway/delivery outlet – a food service establishment that prepares and sells ready-to-eat meals and beverages for immediate consumption (either on the premises, through takeaway, or via delivery). These outlets focus on providing prepared food directly to customers for immediate or near-immediate consumption and do not primarily sell food or drink items in raw or packaged form for home cooking. This category excludes food retailers and manufacturing companies or brands. "
    "List all distinct food items/dishes (not ingredients) observable in the ad based on the overall presentation. "
    "Important: When the ad features a composite food item (e.g., a burger, sandwich, or pizza), select only the single food category that best represents the whole item rather than listing separate categories for its individual ingredients (such as bread, meat, or sauces). "
)

type_ad = [ # make it clear that this question is about the company running the ad
    ("FOOD_PRODUCT_COMPANY", "Is the ad promoting a specific food or drink product from a food/drink manufacturing company or brand, was directly evidenced by the image or text? (e.g. a Coca-cola bottle in someone's hand) "),
    ("FOOD_PRODUCT_NONFOOD_COMPANY", "Is the ad promoting a specific food or drink product but created by a non-food brand/company/retailer/service/event? (e.g. a bank sponsoring free coffee at an event) "),
    ("FOOD_COMPANY_NO_PRODUCT", "Is the ad promoting a food or drink manufacturing company or brand, without showing a specific food or drink product? (e.g. an ad for Nestlé as a brand but not for a specific food or drink) "),
    ("RETAILER_FOOD_PRODUCT", "Is the ad promoting a specific food or drink product from a food or drink retailer? (e.g., a supermarket ad showcasing discounts on fresh produce) "),
    ("RETAILER_NO_FOOD_PRODUCT", "Is the ad promoting a food or drink retailer without featuring any specific food or drink product? (e.g., a store ad focusing on clothes or electronics products) "),
    ("RESTAURANT_FOOD_PRODUCT", "Is the ad promoting a specific food or meal or drink product from a restaurant/takeaway/delivery outlet? (e.g., a McDonald’s ad promoting a new burger) "),
    ("RESTAURANT_NO_FOOD_PRODUCT", "Is the ad promoting a restaurant/takeaway/delivery outlet without showcasing any specific food or meal or drink product? (e.g., a restaurant ad focusing on restaurant ambiance or service without showing specific foods or meals) "),
    ("NONFOOD_PRODUCT", "Is the ad promoting a non-food or drink product or service? (e.g. a mobile phone or car ad) "),
    ("INFANT_FORMULA", "Is the ad promoting infant formula, follow-up, or growing-up milks? (e.g. an ad for Enfamil or Aptamil) ")
]

marketing_str = [
    ("OWNED_CARTOON", "Are there characters specifically created/owned by a company to represent its brand, often used in advertising and product packaging? These characters help establish brand identity and engage with consumers, especially children (e.g., M&Ms, Dino from the Dino brand) "),
    ("LICENSED_CHARACTER", "Are there well-known characters from TV shows or books that a brand pays to use in its promotions? These characters attract fans and enhance the appeal of products through their existing popularity (e.g., Miffy (Nijntje), Mickey Mouse) "),
    ("OTHER_CHARACTER", "Are there other character types (not owned/licensed) that represent cartoons and are not associated with a company/brand (e.g. cartoon-style animals on a lunchbox) "),
    ("MOVIE_TIE_IN", "Are there promotional strategies that align products or brands with popular films, utilizing themes, characters, or storylines from the movies to attract audiences and enhance product appeal? (e.g., Shrek, Frozen) "),
    ("FAMOUS_SPORTS", "Are there renowned athletes or sports teams that are recognized on a national or international level? Their endorsements can lend authority and desirability to products, particularly in sports and fitness markets (e.g., Belgian Red Devils, Kevin de Bruyne) "),
    ("AMATEUR_SPORTS", "Are there individuals who participate in sports at a non-professional level, often representing community teams or clubs? Their relatability can foster a sense of connection with local consumers and promote healthy lifestyles (e.g., a person playing a sport) "),
    ("CELEBRITY", "Are there well-known public figures from various fields such as entertainment, music, or literature who endorse products or brands? Their fame can significantly boost brand visibility and credibility (e.g., Jeroen Meus, Jamie Oliver) "),
    ("AWARDS", "Are there recognitions or accolades received by a product or brand that signify quality or excellence? Featuring awards in marketing materials can enhance credibility and consumer trust (e.g., “Best Beer of 2024”) "),
    ("EVENTS", "Are there promotions tied to significant cultural events, historical anniversaries, or festivals that resonate with the target audience? These events provide a thematic backdrop for marketing campaigns and engage consumers on an emotional level (e.g., Christmas, Halloween) "),
    ("SPORT_EVENTS", "Are there promotions that coincide with major sporting events, leveraging the excitement and audience engagement of the event? This strategy often includes sponsorships, themed products, and advertising tied to the event’s timeline (e.g., Tour de France, Olympic Games)")
]

target_age_group = [
    ("CHILD_TARGETED", "Is the ad targeted at children up to 15 years old? These ads often use simplified messaging, bright colors, and fun or animated characters that resonate with a younger audience. They may feature elements like toys, games, or collectibles and focus on excitement, fun, and social inclusion. The ad may also indirectly appeal to parents by highlighting educational or health benefits. "),
    ("ADOLESCENT_TARGETED", "Is the ad targeted at adolescents between 16 and 18 years old? These ads often highlight themes of individuality, self-expression, and peer influence. They frequently use influencers, social media trends, or aspirational messaging to appeal to teenagers' desire for autonomy, status, or connection to their peers. "),
    ("ADULT_TARGETED", "Is the ad targeted at adults with no specific focus on children or adolescents? These ads emphasize practical, emotional, or lifestyle benefits relevant to a broad adult audience. Messaging may focus on value, quality, sophistication, or relatable everyday needs. ")
]

premium_offer = [
    ("GAMES", "Are there offers that incentivize consumers to download mobile games or apps, often tied to promotions or rewards (e.g. 'download the McDonald's app to receive a free hamburger') "),
    ("CONTESTS", "Are there promotions/contests where consumers enter to win prizes by completing specific actions, often requiring purchase or engagement (e.g., social media photo contests encouraging participants to share picture with a specific product to win prizes). "),
    ("2FOR3", "Are there offers that encourage bulk purchasing by providing extra products for a reduced price? e.g. 1+1 free, 3 for the price of 2 "),
    ("EXTRA", "Are there promotions offering additional product quantity for the same price? e.g. 20%% extra, 50 extra grams free "),
    ("LIMITED", "Are there special products offered for a short/limited time? e.g., limited Halloween edition on candies, seasonal flavor of Dutch stroopwafels available only during Christmas holidays "),
    ("CHARITY", "Are there offers where a portion of proceeds goes to a charitable cause, appealing to socially conscious consumers? e.g., 'Every Purchase Helps Local Farmers' or 'Plant a tree with every purchase' "),
    ("GIFTS", "Are there promotions that include free gifts or collectible items with purchases, encouraging repeat buys? me.g., collectible cups offered with every purchase of a specific soft drink "),
    ("DISCOUNT", "Are there direct reductions in the selling price of products, making them more attractive to consumers? e.g., '€1 Off Any Product', temporary price discount during Black Friday "),
    ("LOYALTY", "Are there programs that reward loyal customers for repeat purchases, often through points or discounts? e.g. coffee stamp cards, supermarket loyalty cards offering points that can be redeemed for discounts or free products ")
]

who_cat = [
    ("CHOCOLATE_SUGAR", "Does the ad feature chocolate or sugar confectionery? (caramels, jellies, cereal bars, spreadable chocolate, honey, table sugar, excluding cakes, jams and sweet desserts) "),
    ("CAKES_PASTRIES", "Does the ad feature cakes or pastries? (cookies, cakes, pies, pastries, pancakes, waffles, scones, baked desserts, dry mixes for making cakes and other sweet baked goods, excluding bread and bread products) "),
    ("SAVOURY_SNACKS", "Does the ad feature savoury snacks? (crackers, nuts, seeds, popcorn, chips, pretzels) "),
    ("JUICES", "Does the ad feature juices? (100%% fruit and vegetable juices, smoothies, excluding sweetened fruit nectars) "),
    ("DAIRY_MILK_DRINKS", "Does the ad feature dairy milk drinks? (dairy milks both sweetened and unsweetened, milkshakes, coffees with milk, excluding cream) "),
    ("PLANT_MILK_DRINKS", "Does the ad feature plant-based milk drinks? (plant-based milks both sweetened and unsweetened, milkshakes, coffees with plant-based milk) "),
    ("ENERGY_DRINKS", "Does the ad feature energy drinks? (beverages containing caffeine or other stimulants such as guarana, taurine, glucuronolactone and vitamins, excluding coffee and tea) "),
    ("SOFT_DRINKS", "Does the ad feature other beverages such as soft drinks or sweetened beverages? (water-based flavored drinks (carbonated and still), fruit and vegetable nectars) "),
    ("WATERS_TEA_COFFEE", "Does the ad feature unsweetened waters, tea, or coffee? (waters (including mineral waters), coffee, tea, other hot beverages) "),
    ("EDIBLE_ICES", "Does the ad feature edible ices? (dairy and plant-based ice creams (including sorbets), frozen yogurts) "),
    ("BREAKFAST_CEREALS", "Does the ad feature breakfast cereals? (minimally processed breakfast cereals (instant oats, porridge mix) and highly processed breakfast cereals (muesli, granola)) "),
    ("YOGHURTS", "Does the ad feature yoghurts, sour milk or similar? (yogurt and sour milks (kefir, buttermilk, drinking yogurt), fruit yogurt, crème fraiche, whipped cream, excluding frozen yogurt) "),
    ("CHEESE", "Does the ad feature cheese? (hard, medium and soft cheeses, processed cheeses, cheese spreads, Gouda, Cheddar, Gruyere, Parmesan, Brie) "),
    ("READYMADE_CONVENIENCE", "Does the ad feature ready-made or convenience foods and composite dishes? (tinned composite dishes (e.g., meatballs in sauce and baked beans), pasta, noodles and rice with sauce, pizza, sandwiches and wraps (e.g., hot dogs, hamburgers), ready-to-eat meals composed of a combination of carbohydrate and either vegetable or meat, soups (ready-to-eat, tinned), frozen and refrigerated dishes (frozen pizza, frozen fish sticks)) "),
    ("BUTTER_OILS", "Does the ad feature butter or other fats and oils? (e.g. butter, margarine, olive oil, oil-based spreads) "),
    ("BREAD_PRODUCTS", "Does the ad feature bread, bread products, or crispbreads? (sweet and raisin breads (brioche), flatbreads, tortillas, pita, leavened bread (all types of cereal flours, e.g., white, or whole-grain wheat, spelt and rye)) "),
    ("PASTA_RICE_GRAINS", "Does the ad feature fresh or dried pasta, rice, or grains? (e.g. spaghetti, rice, quinoa, bulgur, buckwheat) "),
    ("FRESH_MEAT_POULTRY_FISH", "Does the ad feature fresh or frozen meat, poultry, fish, or eggs? (chicken breasts, salmon fillets, eggs) "),
    ("PROCESSED_MEAT_POULTRY_FISH", "Does the ad feature processed meat, poultry, fish, or similar? (tinned tuna, smoked fish, ham, burgers, sausages, fish fingers, breaded meat products) "),
    ("VEGAN_MEAT", "Does the ad feature savoury plant-based foods/meat analogues? (tofu and tempeh, veggie burgers, veggie sausages) "),
    ("FRESH_FRUIT_VEG", "Does the ad feature fresh or frozen fruit, vegetables, or legumes? (e.g. apples, broccoli, lentils, fruit and vegetables without additional ingredients) "),
    ("PROCESSED_FRUIT_VEG", "Does the ad feature processed fruit, vegetables, or legumes? (tinned, pickled, dried, battered (e.g., deep fried onion rings) and breaded vegetables and fruits, pouches, jams and marmalades) "),
    ("SAUCES_DIPS_DRESSINGS", "Does the ad feature sauces, dips, or dressings? (stock cubes, cooking sauces (pasta sauces), dips, salad dressings, condiments (ketchup)) "),
    ("NA", "Is the ad for a non-applicable company or brand that does not feature any foods or drinks? "),
    ("NS", "Is the product category in the ad non-specified or unclear? ")
]

alcohol = [
    ("ALCOHOL", "Can you see the presence of alcoholic drinks or alcoholic brands in the ad? (e.g. items like beer, wine, etc., or brands like Jupiler, Heineken, etc.)")
]

# change this
speculation = [
    ("SPECULATION_LEVEL", "Final question. Based on the ad's image and text, rate on a scale from 0 to 10 how much you had to speculate or guess your answers rather than rely on directly observable information. (0 means every answer is fully supported by the ad; 10 means answers required extensive inference.) Provide a brief explanation of your rating.")
]


def create_user_content():
    user_content = []

    # add each category question block as a grouped selection
    type_ad_text = "### Ad Type: Select exactly ONE category from the list below. Answer 'Yes' for one and 'No' for all others.\n"
    type_ad_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in type_ad])  # Concatenate all type_ad questions
    user_content.append({"type": "text", "text": type_ad_text})
    
    alcohol_text = "### Alcohol presence: answer with Yes/No.\n"
    alcohol_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in alcohol])
    user_content.append({"type": "text", "text": alcohol_text})
    
    marketing_text = "### Marketing Strategies: Select all that apply. Answer 'Yes' for any that are present, and 'No' otherwise.\n"
    marketing_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in marketing_str])
    user_content.append({"type": "text", "text": marketing_text})
    
    premium_text = "### Premium Offers: Select all that apply. Answer 'Yes' for any that are present, and 'No' otherwise.\n"
    premium_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in premium_offer])
    user_content.append({"type": "text", "text": premium_text})
    
    age_text = "### Target Age Group: Select exactly ONE category. Answer 'Yes' for one, and 'No' for all others.\n"
    age_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in target_age_group])
    user_content.append({"type": "text", "text": age_text})
    
    who_cat_text = "### WHO Food Categories: Select all that apply. Answer 'Yes' for any that are present, and explicitly 'No' for all others.\n"
    who_cat_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in who_cat])
    user_content.append({"type": "text", "text": who_cat_text})

    speculation_text = "### Speculation Level: Rate from 0-10.\n"
    speculation_text += "\n".join([f"*{q[0]}*: {q[1]}" for q in speculation])
    user_content.append({"type": "text", "text": speculation_text})

    
    return user_content


# ============ Answer Logic: ============

def get_alcohol(answer_dict):
    if answer_dict["ALCOHOL"][0] == "Yes":
        return "1", answer_dict["ALCOHOL"][1] 
    
    return "0", "The ad does not feature alcoholic drinks or brands."
    

def get_ad_type(answer_dict):
    
    # check each ad type in order of priority and return the first "Yes" answer
    if answer_dict["FOOD_PRODUCT_COMPANY"][0] == "Yes":
        return "1", answer_dict["FOOD_PRODUCT_COMPANY"][1]
    
    elif answer_dict["FOOD_PRODUCT_NONFOOD_COMPANY"][0] == "Yes":
        return "2", answer_dict["FOOD_PRODUCT_NONFOOD_COMPANY"][1]
    
    elif answer_dict["FOOD_COMPANY_NO_PRODUCT"][0] == "Yes":
        return "3", answer_dict["FOOD_COMPANY_NO_PRODUCT"][1]
    
    elif answer_dict["RETAILER_FOOD_PRODUCT"][0] == "Yes":
        return "4", answer_dict["RETAILER_FOOD_PRODUCT"][1]
    
    elif answer_dict["RETAILER_NO_FOOD_PRODUCT"][0] == "Yes":
        return "5", answer_dict["RETAILER_NO_FOOD_PRODUCT"][1]
    
    elif answer_dict["RESTAURANT_FOOD_PRODUCT"][0] == "Yes":
        return "6", answer_dict["RESTAURANT_FOOD_PRODUCT"][1]
    
    elif answer_dict["RESTAURANT_NO_FOOD_PRODUCT"][0] == "Yes":
        return "7", answer_dict["RESTAURANT_NO_FOOD_PRODUCT"][1]
    
    elif answer_dict["NONFOOD_PRODUCT"][0] == "Yes":
        return "8", answer_dict["NONFOOD_PRODUCT"][1]
    
    elif answer_dict["INFANT_FORMULA"][0] == "Yes":
        return "10", answer_dict["INFANT_FORMULA"][1]
    
    # if no matches are found, return "NA" with a default explanation
    return "-1", "No applicable ad type was found."


def get_marketing_strategy(answer_dict):
    strategies = []
    explanations = []

    # check each strategy and append to the list if it's present in the ad (Yes)
    if answer_dict["OWNED_CARTOON"][0] == "Yes": 
        strategies.append("1")  # Cartoon character
        explanations.append(answer_dict["OWNED_CARTOON"][1])

    if answer_dict["LICENSED_CHARACTER"][0] == "Yes":
        strategies.append("2")  # Licensed character
        explanations.append(answer_dict["LICENSED_CHARACTER"][1])
    if answer_dict["OTHER_CHARACTER"][0] == "Yes":
        strategies.append("3")  # Licensed character
        explanations.append(answer_dict["OTHER_CHARACTER"][1])
    try:
        if answer_dict["MOVIE_TIE_IN"][0] == "Yes":
            strategies.append("4")  # Movie tie-in
            explanations.append(answer_dict["MOVIE_TIE_IN"][1])
    except:
        if answer_dict["MOVIE_TIE-IN"][0] == "Yes":
            strategies.append("4")  # Movie tie-in
            explanations.append(answer_dict["MOVIE_TIE-IN"][1])

    if answer_dict["FAMOUS_SPORTS"][0] == "Yes":
        strategies.append("5")  # Famous athletes or sports teams
        explanations.append(answer_dict["FAMOUS_SPORTS"][1])

    if answer_dict["AMATEUR_SPORTS"][0] == "Yes":
        strategies.append("6")  # Amateur sports
        explanations.append(answer_dict["AMATEUR_SPORTS"][1])

    if answer_dict["CELEBRITY"][0] == "Yes":
        strategies.append("7")  # Non-sports celebrities
        explanations.append(answer_dict["CELEBRITY"][1])
        
    if answer_dict["AWARDS"][0] == "Yes":
        strategies.append("8")  # Awards-related marketing
        explanations.append(answer_dict["AWARDS"][1])

    if answer_dict["EVENTS"][0] == "Yes":
        strategies.append("9")  # Cultural or historical events
        explanations.append(answer_dict["EVENTS"][1])

    if answer_dict["SPORT_EVENTS"][0] == "Yes":
        strategies.append("10")  # Sporting events
        explanations.append(answer_dict["SPORT_EVENTS"][1])

    # if no marketing strategies are found, return 0 with a default explanation
    if not strategies:
        return "0", "No marketing strategies found in the ad."

    strategies_string = ", ".join(strategies)
    explanations_string = " ".join(explanations)

    return strategies_string, explanations_string


def get_premium_offer(answer_dict):
    premium_offers = []
    explanations = []

    # check each offer type and append to the list if it's present in the ad (Yes).
    if answer_dict["GAMES"][0] == "Yes":
        premium_offers.append("1")  # Games/App offers
        explanations.append(answer_dict["GAMES"][1])

    if answer_dict["CONTESTS"][0] == "Yes":
        premium_offers.append("2")  # Contests/Promotions
        explanations.append(answer_dict["CONTESTS"][1])

    if answer_dict["2FOR3"][0] == "Yes":
        premium_offers.append("3")  # Bulk purchase offers
        explanations.append(answer_dict["2FOR3"][1])

    if answer_dict["EXTRA"][0] == "Yes":
        premium_offers.append("4")  # Extra product quantity offers
        explanations.append(answer_dict["EXTRA"][1])

    if answer_dict["LIMITED"][0] == "Yes":
        premium_offers.append("5")  # Limited edition or seasonal products
        explanations.append(answer_dict["LIMITED"][1])

    if answer_dict["CHARITY"][0] == "Yes":
        premium_offers.append("6")  # Charity offers
        explanations.append(answer_dict["CHARITY"][1])

    if answer_dict["GIFTS"][0] == "Yes":
        premium_offers.append("7")  # Free gifts or collectibles
        explanations.append(answer_dict["GIFTS"][1])

    if answer_dict["DISCOUNT"][0] == "Yes":
        premium_offers.append("8")  # Direct price discounts
        explanations.append(answer_dict["DISCOUNT"][1])

    if answer_dict["LOYALTY"][0] == "Yes":
        premium_offers.append("9")  # Loyalty programs
        explanations.append(answer_dict["LOYALTY"][1])

    # if no premium offers are found, return 0 with a default explanation.
    if not premium_offers:
        return "0", "No premium offers found in the ad."

    premium_offers_string = ", ".join(premium_offers)
    explanations_string = " ".join(explanations)

    return premium_offers_string, explanations_string


def get_target_group(answer_dict):
    if answer_dict["CHILD_TARGETED"][0] == "Yes":
        return "1", answer_dict["CHILD_TARGETED"][1]
    
    elif answer_dict["ADOLESCENT_TARGETED"][0] == "Yes":
        return "2", answer_dict["ADOLESCENT_TARGETED"][1]
    
    elif answer_dict["ADULT_TARGETED"][0] == "Yes":
        return "3", answer_dict["ADULT_TARGETED"][1]
    
    else:
        return "-1", "Missing answer."


def get_who_cat(answer_dict):
    who_categories = []
    explanations = []

    if answer_dict["NA"][0] == "Yes":
        return "NA", answer_dict["NA"][1] # Non-applicable
        
    # check each WHO category and append to the list if it's present in the ad (Yes).
    if answer_dict["CHOCOLATE_SUGAR"][0] == "Yes":
        who_categories.append("1")  # Chocolate and sugar confectionery
        explanations.append(answer_dict["CHOCOLATE_SUGAR"][1])

    if answer_dict["CAKES_PASTRIES"][0] == "Yes":
        who_categories.append("2")  # Cakes and pastries
        explanations.append(answer_dict["CAKES_PASTRIES"][1])

    if answer_dict["SAVOURY_SNACKS"][0] == "Yes":
        who_categories.append("3")  # Savoury snacks
        explanations.append(answer_dict["SAVOURY_SNACKS"][1])

    if answer_dict["JUICES"][0] == "Yes":
        who_categories.append("4a")  # Juices
        explanations.append(answer_dict["JUICES"][1])

    if answer_dict["DAIRY_MILK_DRINKS"][0] == "Yes":
        who_categories.append("4b")  # Milk drinks
        explanations.append(answer_dict["DAIRY_MILK_DRINKS"][1])
        
    if answer_dict["PLANT_MILK_DRINKS"][0] == "Yes":
        who_categories.append("4c")  # Milk drinks
        explanations.append(answer_dict["PLANT_MILK_DRINKS"][1])

    if answer_dict["ENERGY_DRINKS"][0] == "Yes":
        who_categories.append("4d")  # Energy drinks
        explanations.append(answer_dict["ENERGY_DRINKS"][1])

    if answer_dict["SOFT_DRINKS"][0] == "Yes":
        who_categories.append("4e")  # Other beverages (Soft drinks, sweetened beverages)
        explanations.append(answer_dict["SOFT_DRINKS"][1])

    if answer_dict["WATERS_TEA_COFFEE"][0] == "Yes":
        who_categories.append("4f")  # Waters, tea, and coffee (unsweetened)
        explanations.append(answer_dict["WATERS_TEA_COFFEE"][1])

    if answer_dict["EDIBLE_ICES"][0] == "Yes":
        who_categories.append("5")  # Edible ices
        explanations.append(answer_dict["EDIBLE_ICES"][1])

    if answer_dict["BREAKFAST_CEREALS"][0] == "Yes":
        who_categories.append("6")  # Breakfast cereals
        explanations.append(answer_dict["BREAKFAST_CEREALS"][1])

    if answer_dict["YOGHURTS"][0] == "Yes":
        who_categories.append("7")  # Yoghurts, sour milk, cream, etc.
        explanations.append(answer_dict["YOGHURTS"][1])

    if answer_dict["CHEESE"][0] == "Yes":
        who_categories.append("8")  # Cheese
        explanations.append(answer_dict["CHEESE"][1])

    if answer_dict["READYMADE_CONVENIENCE"][0] == "Yes":
        who_categories.append("9")  # Ready-made and convenience foods
        explanations.append(answer_dict["READYMADE_CONVENIENCE"][1])

    if answer_dict["BUTTER_OILS"][0] == "Yes":
        who_categories.append("10")  # Butter and other fats and oils
        explanations.append(answer_dict["BUTTER_OILS"][1])

    if answer_dict["BREAD_PRODUCTS"][0] == "Yes":
        who_categories.append("11")  # Bread, bread products, and crisp breads
        explanations.append(answer_dict["BREAD_PRODUCTS"][1])

    if answer_dict["PASTA_RICE_GRAINS"][0] == "Yes":
        who_categories.append("12")  # Fresh or dried pasta, rice, and grains
        explanations.append(answer_dict["PASTA_RICE_GRAINS"][1])

    if answer_dict["FRESH_MEAT_POULTRY_FISH"][0] == "Yes":
        who_categories.append("13")  # Fresh and frozen meat, poultry, fish, and eggs
        explanations.append(answer_dict["FRESH_MEAT_POULTRY_FISH"][1])

    if answer_dict["PROCESSED_MEAT_POULTRY_FISH"][0] == "Yes":
        who_categories.append("14")  # Processed meat, poultry, fish, and similar
        explanations.append(answer_dict["PROCESSED_MEAT_POULTRY_FISH"][1])

    if answer_dict["FRESH_FRUIT_VEG"][0] == "Yes":
        who_categories.append("15")  # Fresh and frozen fruit, vegetables, and legumes
        explanations.append(answer_dict["FRESH_FRUIT_VEG"][1])

    if answer_dict["PROCESSED_FRUIT_VEG"][0] == "Yes":
        who_categories.append("16")  # Processed fruit, vegetables, and legumes
        explanations.append(answer_dict["PROCESSED_FRUIT_VEG"][1])
    
    if answer_dict["VEGAN_MEAT"][0] == "Yes":
        who_categories.append("17")  # vegan meat
        explanations.append(answer_dict["VEGAN_MEAT"][1])

    if answer_dict["SAUCES_DIPS_DRESSINGS"][0] == "Yes":
        who_categories.append("18")  # Sauces, dips, and dressings
        explanations.append(answer_dict["SAUCES_DIPS_DRESSINGS"][1])

    if answer_dict["NS"][0] == "Yes":
        who_categories.append("NS")  # Non-specified
        explanations.append(answer_dict["NS"][1])

    # if no WHO categories are found, return "NA" with a default explanation
    if not who_categories:
        return "NA", "No WHO categories found in the ad."

    who_category_string = ", ".join(who_categories)
    explanations_string = " ".join(explanations)

    return who_category_string, explanations_string


# in case some questions were not answered
def process_missing_output(response, expected_labels):
    if isinstance(response, list):
        response = response[0]

    extracted = {label.strip(): {"answer": answer, "explanation": explanation.strip() if explanation else ""}
        for label, answer, explanation in PATTERN_AYA.findall(response)}

    full_output = {}
    for label in expected_labels:
        if label in extracted:
            full_output[label] = extracted[label]
        else:
            full_output[label] = {"answer": "MISSING", "explanation": ""}

    # add any *_PROCESSING labels that were found in the output
    for label in extracted:
        if label not in full_output:
            full_output[label] = extracted[label]

    answer_dict = {label: [data["answer"], data["explanation"]] for label, data in full_output.items()}

    return answer_dict


# should technically work for all models, if their output has been processed properly
def get_final_dict_entry(answer_dict, ad_id):

    ad_type, ad_type_explanation = get_ad_type(answer_dict)
    marketing_str, marketing_str_explanation = get_marketing_strategy(answer_dict)
    prem_offer, prem_offer_explanation = get_premium_offer(answer_dict)
    who_cat, who_cat_explanation = get_who_cat(answer_dict)
    target_group, target_group_expl = get_target_group(answer_dict)
    is_alcohol, is_alcohol_expl = get_alcohol(answer_dict)
                                
    dict_entry = {
        "img_id": ad_id,
        "type_ad": ad_type,
        "type_ad_expl": ad_type_explanation,
        "marketing_str": marketing_str,
        "marketing_str_expl": marketing_str_explanation,
        "prem_offer": prem_offer,
        "prem_offer_expl": prem_offer_explanation,
        "target_group": target_group,
        "target_group_expl": target_group_expl,
        "who_cat": who_cat,
        "who_cat_expl": who_cat_explanation,
        "is_alcohol": is_alcohol,
        "is_alcohol_expl": is_alcohol_expl,
        "speculation": answer_dict["SPECULATION_LEVEL"][0], # added speculation
        "speculation_expl": answer_dict["SPECULATION_LEVEL"][1]
    }

    print(dict_entry) 

    return dict_entry
