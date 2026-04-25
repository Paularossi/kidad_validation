import json
import base64
import os
import time
import pandas as pd
import torch
import re

from huggingface_hub import login
from transformers import Gemma3ForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from screenshot_filtering.questions import *

# run this file in the terminal with `nohup python3 -m screenshot_filtering.classification`

# use the GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

TEXT_MODELS = ["google/gemma-3-12b-it"] # or the bigger one google/gemma-3-27b-it
MULTIMODAL_MODELS = ["Qwen/Qwen2.5-VL-32B-Instruct"]

# ad disclosure keywords: English, Dutch (gesponsord/advertentie), French (publicité)
AD_KEYWORDS = re.compile(r"(sponsored|promoted|gesponsord|advertentie|publicité)", re.I)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

def initiate_transformers_model(model_id):
    """Load the right model and processor based on model_id."""
    # available models
    MODEL_MAP = { # try bigger
        "google/gemma-3-12b-it": Gemma3ForConditionalGeneration, # Gemma3 - https://huggingface.co/google/gemma-3-12b-it
        "Qwen/Qwen2.5-VL-32B-Instruct": Qwen2_5_VLForConditionalGeneration, # Qwen - https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct
    }

    if model_id not in MODEL_MAP:
        print(f"Model {model_id} not available.")
        return None, None

    # initiate the right model
    model = MODEL_MAP[model_id].from_pretrained(model_id, device_map="auto", trust_remote_code=True).eval() # use .eval() to switch to evaluation (inference) mode
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code = True)
    
    print(f"Successfully loaded model and processor with id {model_id}.")
    return model, processor


def apply_ocr_fallback(image_path, result_entry, conf_threshold=0.8):
    """Boost label to AD if ad-related keywords are found via OCR in the top banner area."""
    try:
        import pytesseract
        from PIL import Image
        im = Image.open(image_path)
        w, h = im.size
        crop = im.crop((0, 0, w, int(h * 0.15)))
        text = pytesseract.image_to_string(crop)
        if AD_KEYWORDS.search(text):
            result_entry["label"] = "AD"
            result_entry["confidence"] = max(result_entry.get("confidence", 0.0), conf_threshold)
    except Exception:
        pass
    return result_entry


def _call_with_retry(fn, *args, max_retries=3, base_delay=5.0, **kwargs):
    """Call fn(*args, **kwargs) with exponential-backoff retry on failure."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.0f}s...")
            time.sleep(delay)


#image = images[4]
def start_classification_trns(model, processor, model_id, image, image_folder):

    image_id = os.path.splitext(os.path.basename(image))[0]
    print(f'======== Labeling image: {image_id}. ========\n')

    image_path = os.path.join(image_folder, image)
    base64_image = encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    user_content = []
    messages = [{"role": "system", "content": [{"type": "text", "text": instructions}]}] # get the instructions

    user_content.append({"type": "text", "text": f"ID: {image_id}"})
    user_content.append({"type": "image", "url": f"data:image/png;base64,{base64_image}"} # Gemma
            if model_id.startswith("google") else {"type": "image", "image": image_path}) # qwen?
    messages.append({"role": "user", "content": user_content})

    # prepare the input based on the model being used
    if model_id in TEXT_MODELS: # use the chat template
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt" # pytorch tensor format output for gpu acceleration
        ).to(model.device, dtype=torch.bfloat16) # bfloat16 instead of float16 for less memory consumption (best for inference)

        input_len = inputs["input_ids"].shape[-1] # length of input prompt (to remove)
    
    elif model_id in MULTIMODAL_MODELS: # need separate processing for images
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages) # qwen requires separate image processing
        inputs = processor(
            text=[text_input], images=image_inputs, padding=True, return_tensors="pt"
        ).to(model.device)
        # use padding to make sure all inputs are the same length (pytorch can't create a tensor otherwise)

    start_time = time.time()
    # generate the response (regardless the model)
    with torch.inference_mode(): # optimize inference by disabling gradient calculations to save memory and speed up processing
        generation = model.generate(**inputs, max_new_tokens=1300, do_sample=False) # deterministic generation (not random)

    end_time = time.time()
    response_time = end_time - start_time
    print(f"Time taken to generate response: {response_time:.2f} seconds") 
    print(f"CUDA memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    torch.cuda.empty_cache() # free unused memory

    # decode the response based on the ml being used
    if model_id in MULTIMODAL_MODELS:
        # Qwen requires trimming the input tokens before decoding
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generation)]
        response = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if isinstance(response, list):
            response = response[0]

        response = parse_qwen_json(response)
        print(response) 
    else:
        response = processor.decode(generation[0][input_len:], skip_special_tokens=True) # for gemma
        print(response) 

    return response, response_time


def label_images(images, model_id, image_folder, checkpoint_path=None):
    """Classify a list of images using the specified model.

    Args:
        images: List of image filenames (without directory prefix).
        model_id: HuggingFace model ID.
        image_folder: Directory containing the image files.
        checkpoint_path: Optional path to an Excel file for checkpoint saving/resuming.
                         If the file exists, already-processed IDs are skipped.
    """
    model, processor = initiate_transformers_model(model_id)
    if model is None:
        print(f"Error loading the model {model_id}. Quitting...")
        return None
    print(f"Starting classifying {len(images)} images with model {model_id}...")

    # load checkpoint if available
    results = []
    responses = []
    processed_ids = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint_df = pd.read_excel(checkpoint_path)
        results = checkpoint_df.to_dict(orient="records")
        processed_ids = set(checkpoint_df["id"].astype(str).tolist())
        print(f"Resumed from checkpoint: {len(processed_ids)} items already processed.")

    n = 1 # just to count the images
    
    for image in images:
        image_id = os.path.splitext(os.path.basename(image))[0]
        if image_id in processed_ids:
            print(f"Skipping already-processed image: {image_id}")
            n += 1
            continue

        try:
            response, response_time = _call_with_retry(
                start_classification_trns, model, processor, model_id, image, image_folder
            )
            responses.append(response)

            dict_entry = process_first_output(json.loads(response) if model_id in TEXT_MODELS else response)
            dict_entry.update({"response_time": round(response_time, 2)})

            # apply OCR fallback to boost AD confidence when ad keywords appear in the banner
            image_path = os.path.join(image_folder, image)
            dict_entry = apply_ocr_fallback(image_path, dict_entry)
        except Exception as e:
            print(f"Error processing image {image} due to: {e}.")
            dict_entry = {"id": image_id, "label": "UNCERTAIN", "confidence": 0.0,
                          "signals": [f"Error: {str(e)}"]}
    
        results.append(dict_entry)

        # save checkpoint after every image
        if checkpoint_path:
            try:
                pd.DataFrame(results).to_excel(checkpoint_path, index=False)
            except Exception as ce:
                print(f"Warning: could not save checkpoint: {ce}")

        print(f"===== Image {n} out of {len(images)} classified! =====")
        n += 1

    try:
        labeling_outputs = pd.DataFrame(results)
        labeling_outputs['id'] = labeling_outputs['id'].astype(str)
    except Exception as e:
        print(results)
        print(f"Unable to convert the output to a dataframe. Returning the data as it is.")
        return results, responses

    print(f"DONEEEE classifying {len(images)} images using model {model_id} !!!")
    return labeling_outputs, responses


if __name__ == "__main__":
    screenshot_set = "screenshots 1"
    image_folder = "data/screenshots 1"
    images = [file for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images = sorted(images)

    with open('keys.txt') as f:
        json_data = json.load(f)

    hugg_key = json_data["huggingface"]

    model_id = MULTIMODAL_MODELS[0]
    login(hugg_key) # log into hugging face (gated models like gemma)

    labeling_outputs, responses = label_images(
        images, model_id, image_folder,
        checkpoint_path=f"data/{screenshot_set}_first_filtering_qwen_checkpoint.xlsx"
    )
    print(labeling_outputs)

    labeling_outputs.to_excel(f"data/{screenshot_set}_first_filtering_qwen.xlsx", index=False)
