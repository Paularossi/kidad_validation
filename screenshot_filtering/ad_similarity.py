import os
import numpy as np
import pandas as pd
import torch
import imagehash
from datetime import datetime, timedelta
from PIL import Image
import pytesseract
from transformers import CLIPProcessor, CLIPModel

from screenshot_filtering.helpers import *

INTERVAL_SECONDS = 2
image_folder = "data/screenshots 1"
images = [file for file in os.listdir(image_folder) if file.lower().endswith(('.jpg', '.jpeg', '.png'))]

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


# load CLIP model + processor for image embeddings
def load_clip(device = None, model_name = "openai/clip-vit-base-patch32"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor, device


@torch.inference_mode()
def compute_clip_embeddings(df, image_col = "screenshot_path", batch_size = 32, model_name = "openai/clip-vit-base-patch32", device = None):
    """Compute normalized CLIP embeddings for each image in df and return a copy with the new ``clip_emb`` column."""
    model, processor, device = load_clip(device=device, model_name=model_name)

    embs = []
    paths = df[image_col].tolist()

    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i:i+batch_size]
        images = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            images.append(img)

        # encode the current batch into the CLIP embedding space
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)
        feats = model.get_image_features(**inputs)  # (B, D)

        # L2 normalize so cosine similarity = dot product
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats = feats.detach().cpu().numpy()

        embs.extend(list(feats))

    out = df.copy()
    out["clip_emb"] = embs
    return out


def add_ocr_tokens_tesseract(df, path_col="screenshot_path", ocr_col="ocr_tokens", lang="eng", psm=6):
    """
    Extract OCR text with Tesseract for each image and store as a token set.
    """
    df = df.copy()
    tokens = []

    config = f"--psm {psm}"
    for p in df[path_col].tolist():
        img = Image.open(p).convert("RGB")
        text = pytesseract.image_to_string(img, lang=lang, config=config)
        tokens.append(normalize_ocr_text(text))

    df[ocr_col] = tokens
    return df



def group_screenshots_by_similarity(df, time_col="timestamp", emb_col="clip_emb", path_col="screenshot_path", time_threshold_s=6.0, 
                                    sim_threshold=0.75, phash_dist_threshold=12, hash_size=16, ocr_col="ocr_tokens", ocr_threshold=0.25, use_ocr=True):
    """Traverse the timeline and build exposure groups whenever a screenshot stays close in time and visually similar.
    The grouping blends CLIP cosine similarity, pHash distance, and optional OCR overlap so ad variants are grouped together.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # compute pHash once
    phash = [imagehash.phash(Image.open(p).convert("RGB"), hash_size=hash_size) for p in df[path_col]]
    df["_phash"] = phash

    group_id = [1]
    prev_dt = [np.nan]
    prev_sim = [np.nan]
    prev_hash = [np.nan]
    prev_ocr = [np.nan]

    gid = 1
    anchor_idx = 0  # track the first frame of the current group for anchor comparisons

    for i in range(1, len(df)):
        dt = (df.iloc[i][time_col] - df.iloc[i-1][time_col]).total_seconds()
        sim = float(np.dot(df.iloc[i-1][emb_col], df.iloc[i][emb_col])) # cosine similarity
        hdist = int(df.iloc[i-1]["_phash"] - df.iloc[i]["_phash"]) # phash hamming distance
        if use_ocr:
            ocr = jaccard(df.iloc[i-1][ocr_col], df.iloc[i][ocr_col]) # OCR Jaccard similarity
        else:
            ocr = np.nan

        sim_anchor = float(np.dot(df.iloc[anchor_idx][emb_col], df.iloc[i][emb_col]))
        hdist_anchor = int(df.iloc[anchor_idx]["_phash"] - df.iloc[i]["_phash"])
        ocr_anchor = jaccard(df.iloc[anchor_idx][ocr_col], df.iloc[i][ocr_col]) if use_ocr else 0.0

        # treat frames as the same exposure if the gap is short and any similarity cue clears its threshold
        same_group = (dt <= time_threshold_s) and (
            (sim >= sim_threshold) or
            (hdist <= phash_dist_threshold) or
            (use_ocr and (ocr >= ocr_threshold)) or
            (sim_anchor >= sim_threshold) or
            (hdist_anchor <= phash_dist_threshold) or
            (use_ocr and (ocr_anchor >= ocr_threshold))
        )

        if not same_group:
            gid += 1
            anchor_idx = i  # new group anchor

        group_id.append(gid)
        prev_dt.append(dt)
        prev_sim.append(sim)
        prev_hash.append(hdist)
        prev_ocr.append(ocr)

    df["group_id"] = group_id
    df["prev_dt_seconds"] = prev_dt
    df["prev_cosine_sim"] = prev_sim
    df["prev_phash_dist"] = prev_hash
    if use_ocr:
        df["prev_ocr_jaccard"] = prev_ocr

    df.drop(columns=["_phash"], inplace=True)  # cleanup helper column

    return df


# generate dummy timestamps for the first 10 images
files = sorted(images, key=extract_num)

start_time = datetime(2025, 1, 1, 0, 0, 0)  # arbitrary
df = pd.DataFrame({
    "screenshot_id": [os.path.splitext(f)[0] for f in files],
    "screenshot_path": [os.path.join(image_folder, f) for f in files],
    "timestamp": [start_time + timedelta(seconds=i * INTERVAL_SECONDS) for i in range(len(files))]
})

# 1) compute CLIP embeddings and previous-frame similarities
df_ads_emb = compute_clip_embeddings(df, image_col="screenshot_path", batch_size=32, model_name="openai/clip-vit-base-patch32")

prev_sims = [np.nan]
# compute cosine similarity to previous frame
for i in range(1, len(df_ads_emb)):
    prev_sims.append(float(np.dot(df_ads_emb.loc[i-1, "clip_emb"], df_ads_emb.loc[i, "clip_emb"])))

df_ads_emb["prev_cosine_sim"] = prev_sims
df_ads_emb[["screenshot_id", "timestamp", "prev_cosine_sim"]]


# 2) group by similarity using CLIP + pHash + OCR
df_ads_emb = df_ads_emb.sort_values("timestamp").reset_index(drop=True)
df_ads_emb = add_ocr_tokens_tesseract(df_ads_emb, lang="eng+fra+nld", psm=6)
df_ads_emb_ocr = group_screenshots_by_similarity(df_ads_emb, time_col="timestamp", emb_col="clip_emb", path_col="screenshot_path",
                                                 time_threshold_s=6.0, sim_threshold=0.75, phash_dist_threshold=12, hash_size=16,
                                                 ocr_col="ocr_tokens", ocr_threshold=0.25, use_ocr=True)

df_ads_emb_ocr.to_excel("data/screenshots1_ad_groups_ocr.xlsx", index=False)
