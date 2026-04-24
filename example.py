import base64, json, os, time, requests, re
import pandas as pd

from kidad.screenshot_filtering.questions import * # when running from the dsri use kidad.
from kidad.screenshot_filtering.helpers import *

# use the GPU if available
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

from huggingface_hub import login
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from accelerate import infer_auto_device_map # to offload to gpu
# 'models.qwen3_vl.video_processing_qwen3_vl'

CONF_THRESHOLD = 0.7
#MISTRAL_API_KEY = json_data["mistralai"]
#API_URL = "https://api.mistral.ai/v1/chat/completions"
#MODEL = "pixtral-12b-2409"
MODEL = "Qwen/Qwen3-VL-32B-Instruct" # apparently can process videos too
# could also try OpenGVLab/InternVL3-38B-hf https://huggingface.co/OpenGVLab/InternVL3-38B-hf

AD_KEYWORDS = re.compile(r"(sponsored|promoted|gesponsord|advertentie|publicité)", re.I)

with open('kidad/keys.txt') as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


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


def qwen_call(model, processor, model_id, image):

    image_id = os.path.splitext(os.path.basename(image))[0]
    print(f'======== Labeling image: {image_id}. ========\n')

    image_path = os.path.join(image_folder, image)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"ID: {image_id}"},
            {"type": "image", "image": image_path},
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(device)

    print(model.device)
    print(inputs["input_ids"].device)

    start_time = time.time()
    # generate the response
    with torch.inference_mode(): # optimize inference by disabling gradient calculations to save memory and speed up processing
        generation = model.generate(**inputs, max_new_tokens=800, do_sample=False) # deterministic generation (not random)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Time taken to generate response: {response_time:.2f} seconds") 
    print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    torch.cuda.empty_cache() # free unused memory

    # decode the response based on the ml being used, Qwen requires trimming the input tokens before decoding
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    response = parse_qwen_json(response)
    print(response)


def qwen_call_video(model, processor, model_id, video, model_fps = 0.7):

    video_id = os.path.splitext(os.path.basename(video))[0]
    print(f'======== Labeling clip: {video_id}. ========\n')

    messages = [
        {"role": "system", "content": [{"type": "text", "text": instructions_video}]},
        {"role": "user", "content": [
            {"type": "text", "text": f"ID: {video_id}"},
            {
                "type": "video",
                "video": video
            },
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages, 
        tokenize=True, 
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(device)

    start_time = time.time()
    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=800, do_sample=False)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Time taken to generate response: {response_time:.2f} seconds") 
    print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    torch.cuda.empty_cache()

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
    response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    response = parse_qwen_json(response)
    print(response)
    return response


def filter_ads(media, model_id, api_key = None, api_url = None, images = True):

    login(hugg_key)

    # initiate the right model
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, device_map="cuda:0", dtype="auto", attn_implementation="flash_attention_2",
    ).eval() # use .eval() to switch to evaluation (inference) mode
    processor = AutoProcessor.from_pretrained(model_id)
    print(f"Successfully loaded model and processor with id {model_id}.")

    results = [] # for the labels
    responses = []
    n = 1

    for med in media:
        print(f"Processing {"image" if images else "video"} {n}/{len(media)}: {med}")
        try:
            if images:
                response = qwen_call(model, processor, model_id, med)
            else: 
                response = qwen_call_video(model, processor, model_id, med)
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

            # check for ad keywords in signals
            if any(AD_KEYWORDS.search(signal) for signal in signals):
                label_new = "AD"
                confidence_new = max(confidence, 0.8) # boost confidence if ad keywords are found
                
                # add the new label and confidence to the results
                results[-1]["label"] = label_new
                results[-1]["confidence"] = confidence_new

        except Exception as e:
            print(f"Error processing {"image" if images else "video"} {med}: {e}")
            results.append({
                "id": os.path.splitext(os.path.basename(med))[0],
                "label": "UNCERTAIN",
                "confidence": 0.0,
                "signals": [f"Error: {str(e)}"]
            })

        n += 1
        time.sleep(1) # to avoid rate limiting
    
    labeling_outputs = pd.DataFrame(results)

    return labeling_outputs, responses


# start from here
image_folder = "kidad/data/test_images"
video_folder = "kidad/data/videos"
images = [file for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
images = sorted(images)


# ============================================================================
# TEST STEP - CONVERT INTO VIDEOS
# read metadata
metadata = pd.read_excel("kidad/data/metadata.xlsx")
metadata = metadata[~metadata['enrol_number'].astype(str).str.startswith("32")] # remove test accounts

metadata['enrol_number'].value_counts() # check the counts, idk why they are all different

# take a random participant for testing
meta_test = metadata[metadata['enrol_number'] == 23841529678]
meta_test = meta_test.sort_values(by='Time')

meta_test['AppId'].value_counts()
meta_test = meta_test[meta_test['AppId'].str.contains("musically", case=False, na=False)]
meta_test = meta_test.head(300) # just a test sample
meta_test['Time'] = pd.to_datetime(meta_test['Time'])
meta_test['delta_time'] = meta_test['Time'].diff().dt.total_seconds().fillna(0) # find the time delta between screenshots
meta_test["image"] = meta_test["image"].astype(str)
meta_test = meta_test.sort_values(by='Time')
id_to_ts = dict(zip(meta_test["image"], meta_test["Time"]))

images_list = meta_test['image'].tolist()
images_list = [f"{image}.png" for image in images_list]
meta_images = [image for image in images if image in images_list]
meta_images = [os.path.join(image_folder, f) for f in meta_images]

videos = screenshots_to_videos(meta_images=meta_images, id_to_ts=id_to_ts, output_folder=video_folder, window_minutes = 10.0)


# ============================================================================
# the steps for doing the analysis:
# 1. filter ads from non-ads
#results, responses = filter_ads(images[0:10], MODEL) # for all images

# 2. build ad groups (based on screenshot similarity)
#group_metadata = pd.read_excel(f"data/{screenshot_set}_ad_groups_{MODEL}.xlsx")

#image_groups, images = build_ad_groups(group_metadata, images, image_folder)

# 3. annotate them using qwen for the same questions as AI validation

# === USE VIDEOS ===
results, responses = filter_ads(media=videos["video_path"].tolist(), model_id=MODEL, images = False)

