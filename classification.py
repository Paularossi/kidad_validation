import json
import base64
import os
import time
import pandas as pd
import torch

from huggingface_hub import login
from transformers import AutoModelForImageTextToText, Gemma3ForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from questions import instructions, process_first_output

# use the GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

TEXT_MODELS = ["google/gemma-3-12b-it"] # or the bigger one google/gemma-3-27b-it
MULTIMODAL_MODELS = ["Qwen/Qwen2.5-VL-32B-Instruct"]

image_folder = "data/screenshots 1"
images = [file for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

with open('keys.txt') as f:
    json_data = json.load(f)

hugg_key = json_data["huggingface"]


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



def start_classification_trns(model, processor, model_id, image):

    image_id = os.path.splitext(os.path.basename(image))[0]
    print(f'======== Labeling image: {image_id}. ========\n')

    image_path = os.path.join(image_folder, image)
    base64_image = encode_image(image_path)
    image_url = f"data:image/jpeg;base64,{base64_image}"
    user_content = []
    messages = [{"role": "system", "content": instructions}] # get the instructions

    user_content.append({"type": "text", "text": f"ID: {image_id}"})
    user_content.append({"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"} # Gemma
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
        generation = model.generate(**inputs, max_new_tokens=1300, do_sample=False, # deterministic generation (not random)
            temperature = 0.1)

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
        print(response) 
    else:
        response = processor.decode(generation[0][input_len:], skip_special_tokens=True) # for gemma and aya
        print(response) 

    return response, response_time


def label_images(images, model_id):
    
    model, processor = initiate_transformers_model(model_id)
    if model is None:
        print(f"Error loading the model {model_id}. Quitting...")
        return None
    print(f"Starting classifying {len(images)} images with model {model_id}...")
    
    results = [] # for the labels
    responses = []
    n = 1 # just to count the images
    
    for image in images:
        try:
            response, response_time = start_classification_trns(model, processor, model_id, image)
            responses.append(response)

            dict_entry = process_first_output(json.loads(response))
            dict_entry.update({"response_time": round(response_time, 2)})
        except Exception as e:
            print(f"Error processing image {image} due to: {e}.")
            dict_entry = {"img_id": image}
    
        results.append(dict_entry)
        print(f"===== Image {n} out of {len(images)} classified! =====")
        n += 1

    try:
        labeling_outputs = pd.DataFrame(results)
        labeling_outputs['img_id'] = labeling_outputs['img_id'].astype(str)
    except Exception as e:
        print(results)
        print(f"Unable to convert the output to a dataframe. Returning the data as it is.")
        return results, responses

    print(f"DONEEEE classifying {len(images)} images using model {model_id} !!!")
    return labeling_outputs, responses


model_id = TEXT_MODELS[0]
login(hugg_key) # log into hugging face (gated models like gemma)

labeling_outputs, responses = label_images(images[0:10], model_id)
print(labeling_outputs)