# KidAd Validation Pipeline

This project streamlines large-scale ad validation on social media feeds (Instagram, TikTok, etc.). The workflow combines computer vision filtering with two multimodal transformers (Gemma 3 and Qwen 2.5) to filter and annotate food or alcohol advertisements, reusing the previously developed [AI-validation](https://github.com/Paularossi/AI-validation) pipeline.

## Prerequisites
- **Python 3.8+** with packages listed in `requirements.txt`
- **Local model setup** (for Gemma, Pixtral, Qwen via `Transformers`)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Paularossi/kidad-validation.git
   cd kidad-validation
   ```

2. **Set up a virtual environment (venv)**

    Create the virtual environment with:
    ```bash
   python -m venv venv
   ```

   And then activate it with:
   ```bash
   venv\Scripts\activate.bat
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys:**
   - Create a `keys.txt` file
   - Add your HuggingFace key
   - Configure local model endpoints

## Pipeline Overview 
- **Screening:** Load screenshots and filter to images likely containing food/alcohol ads using Gemma3 and Qwen2.5 models plus OCR fallbacks.
- **Ad grouping:** Detect consecutive screenshots that belong to the same ad so downstream annotation works on coherent ad units.
- **Ad annotation:** Reuse the [AI-validation](https://github.com/Paularossi/AI-validation) pipeline and run Gemma 3 and Qwen 2.5 over each grouped ad, producing structured labels for policy checks.



**Usage**
- Place credential files (e.g., `keys.txt`) and screenshots under `data/`.
- Run `classification.py` to batch-filter screenshots and capture raw model responses from the first round of filtering.
- Outputs are stored as pandas dataframes.
- **TODO: add screenshot similarity calculation and screenshot grouping into ads.**
