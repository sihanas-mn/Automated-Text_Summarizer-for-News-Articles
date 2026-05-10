# Automated Text Summarizer for News Articles

This project is a **Streamlit web app** that summarizes news articles from a URL using a fine-tuned **T5-Base** model.

## Features

- Fetches article content from a news URL using `newspaper3k`
- Extracts and displays:
  - Title
  - Authors
  - Publish date
  - Publisher
- Generates an abstractive summary with a fine-tuned T5 model
- Lets you expand and view the original article text

## Project Files

- `app.py` – Streamlit application
- `abstractive-model-sihanas.pth` – fine-tuned model weights
- `requirements.txt` – Python dependencies
- `T5 finetuning.ipynb` – notebook used for model training/fine-tuning

## Prerequisites

- Python 3.9+ recommended
- `pip`
- Internet connection (required to fetch article content and Hugging Face model files on first run)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sihanas-mn/Automated-Text_Summarizer-for-News-Articles.git
   cd Automated-Text_Summarizer-for-News-Articles
   ```

2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows (PowerShell)
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

From the project root, start Streamlit:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal (usually `http://localhost:8501`).

## How to Use

1. Paste a valid news article URL into **News Article URL**.
2. Click **Summarize**.
3. Wait for processing:
   - the app downloads/parses the article
   - metadata is displayed
   - the summary is generated
4. Expand **Show original article** if you want to compare with the source text.

## Notes

- The app automatically uses **GPU (CUDA)** if available, otherwise CPU.
- First run can take longer because model/tokenizer resources are loaded and cached.
- If an article URL is unsupported or blocked, extraction may fail.

## Troubleshooting

- **`No module named ...`**  
  Reinstall dependencies with `pip install -r requirements.txt`.

- **Model loading errors**  
  Ensure `abstractive-model-sihanas.pth` exists in the project root and is not corrupted.

- **Article fetch failures**  
  Try another public news URL or check your network connection.
