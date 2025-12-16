import base64, json, os, time, requests, re
import pandas as pd

from screenshot_filtering.questions import instructions

image_folder = "data/screenshots 1"
images = [file for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

with open('keys.txt') as f:
    json_data = json.load(f)

CONF_THRESHOLD = 0.7
MISTRAL_API_KEY = json_data["mistralai"]
API_URL = "https://api.mistral.ai/v1/chat/completions"
MODEL = "pixtral-12b-2409"

AD_KEYWORDS = re.compile(r"(sponsored|promoted|gesponsord|advertentie|publicité)", re.I)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def apply_ocr_fallback(image_path, result):
    # crude OCR: check top 15% of screenshot
    try:
        import pytesseract
        from PIL import Image
        im = Image.open(image_path)
        w,h = im.size
        crop = im.crop((0,0,w,int(h*0.15)))
        text = pytesseract.image_to_string(crop)
        if AD_KEYWORDS.search(text):
            result["label"] = "AD"
            result["confidence"] = max(result["confidence"], CONF_THRESHOLD)
    except Exception:
        pass
    return result

"image" = images[4]

def mistral_call(model_id, api_key, api_url, image):

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # strip the image path to just the filename without .png/.jpg
    image_id = os.path.splitext(os.path.basename(image))[0]

    image_path = os.path.join(image_folder, image)
    base64_image = encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    user_content = []
    messages = [{"role": "system", "content": instructions}] # get the instructions

    user_content.append({"type": "text", "text": f"ID: {image_id}"})
    user_content.append({"type": "image_url", "image_url": image_url})
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.01,
        "response_format": {"type":"json_object"}
    }

    response = requests.post(api_url, headers = headers, json = payload)
    response.raise_for_status()
    #print(response.json())
    return json.loads(response.json()["choices"][0]["message"]["content"])


def filter_ads(images, model, api_key, api_url):

    results = [] # for the labels
    responses = []
    n = 1

    for image in images:
        print(f"Processing image {n}/{len(images)}: {image}")
        n += 1
        try:
            response = mistral_call(model, api_key, api_url, image)
            responses.append(response)
            item = response["items"][0]
            signals = item.get("signals", [])
            label = item["label"]
            confidence = item["confidence"]
            ad_followup = item.get("ad_followup")

            result_entry = {
                "id": item["id"],
                "label": label,
                "confidence": confidence,
                "signals": signals
            }

            if isinstance(ad_followup, dict):
                for key, value in ad_followup.items():
                    if key not in result_entry:
                        result_entry[key] = value

            results.append(result_entry)

            # Check for ad keywords in signals
            if any(AD_KEYWORDS.search(signal) for signal in signals):
                label_new = "AD"
                confidence_new = max(confidence, 0.8)  # Boost confidence if ad keywords are found
                
                # add the new label and confidence to the results
                results[-1]["label"] = label_new
                results[-1]["confidence"] = confidence_new

        except Exception as e:
            print(f"Error processing image {image}: {e}")
            results.append({
                "id": os.path.splitext(os.path.basename(image))[0],
                "label": "UNCERTAIN",
                "confidence": 0.0,
                "signals": [f"Error: {str(e)}"]
            })

        time.sleep(1)  # to avoid rate limiting
    
    labeling_outputs = pd.DataFrame(results)

    return labeling_outputs, responses

# example for one image
#result = mistral_call(MODEL, MISTRAL_API_KEY, API_URL, images[3])
#print(json.dumps(result, indent=2))

# for all images
results, responses = filter_ads(images[0:10], MODEL, MISTRAL_API_KEY, API_URL)


