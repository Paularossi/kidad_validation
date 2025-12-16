import re

# normalize OCR text into a set of tokens
def normalize_ocr_text(text: str) -> set:
    # lowercase, keep words/numbers, drop very short tokens
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in tokens if len(t) >= 3}

# compute Jaccard similarity between two token sets
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

# extract first number found in a string
def extract_num(fn: str) -> int:
    m = re.search(r"(\d+)", fn)
    return int(m.group(1)) if m else 10**18


