# Tesseract OCR + Streamlit App for PDF Documents

A local, offline Streamlit web application that accepts PDF files, extracts text from every page using Tesseract OCR, and presents the structured output in a clean, interactive UI.

![Screenshot](docs/screenshot.png)

## Installation

### 1. Install System Prerequisites

This application runs entirely offline and requires the system binaries for **Tesseract OCR** and **Poppler** (for PDF rasterization).

#### Windows:
1. Download and run the Tesseract installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) (and add to your `PATH` or specify custom path in sidebar).
2. **Poppler (PDF rasterization)**: We have downloaded and prepackaged the Windows binaries of Poppler inside `src/poppler/` automatically, so you do **not** need to install or configure Poppler manually on Windows!
3. Alternatively, you can install Tesseract via `winget`:
   ```bash
   winget install --id tesseract-ocr.tesseract
   ```

#### Ubuntu / Debian:
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

#### macOS (Homebrew):
```bash
brew install tesseract poppler
```

### 2. Set Up Python Environment & Dependencies
Create a virtual environment and install the required dependencies:
```bash
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Running the App

Start the Streamlit application using:
```bash
streamlit run app.py
```

## Installing Tesseract Language Packs

To process bilingual documents, download and install additional Tesseract language packs:

### Ubuntu / Debian:
```bash
sudo apt-get install tesseract-ocr-hin      # Hindi (hin)
sudo apt-get install tesseract-ocr-ara      # Arabic (ara)
sudo apt-get install tesseract-ocr-chi-sim  # Chinese Simplified (chi_sim)
sudo apt-get install tesseract-ocr-chi-tra  # Chinese Traditional (chi_tra)
sudo apt-get install tesseract-ocr-fra      # French (fra)
sudo apt-get install tesseract-ocr-deu      # German (deu)
sudo apt-get install tesseract-ocr-spa      # Spanish (spa)
sudo apt-get install tesseract-ocr-jpn      # Japanese (jpn)
```

### macOS (Homebrew):
Installs all available languages:
```bash
brew install tesseract-lang
```

### Windows:
Re-run the UB Mannheim installer and select the desired language packs in the components selection window, or download language data files (`.traineddata`) directly from the [Tesseract OCR tessdata repository](https://github.com/tesseract-ocr/tessdata) and copy them into your `tessdata` directory (usually `C:\Program Files\Tesseract-OCR\tessdata`).

Verify installed languages by running:
```bash
tesseract --list-langs
```

## Configuration Reference

| Sidebar Setting | Description | Impact on OCR Quality |
|---|---|---|
| **Render DPI** | Image resolution for PDF rasterization (default: 300). | Higher DPI improves OCR accuracy for small/blurry text but increases processing time and memory usage. |
| **Page Range** | Selects target pages (All, Single Page, Page Range). | Limits processing to requested sections to save computation. |
| **Language Pack** | Tesseract language model combination (e.g., `eng+hin`). | Crucial for bilingual pages. Specifying incorrect language packs yields garbled text. |
| **PSM Mode** | Page Segmentation Mode (Tesseract `--psm`). | Determines layout analysis logic. `3` is default. Use `11` or `13` for sparse or handwritten documents. |
| **OEM Mode** | OCR Engine Mode (Tesseract `--oem`). | Configures engine backend. LSTM + Legacy (recommended) is most accurate. |
| **Denoise Image** | OpenCV fastNlMeansDenoising filter. | Reduces background artifacts, especially on scanned or degraded paper. |
| **Deskew Text** | Rotates image based on detected text angle. | Fixes misaligned scans. Correcting tilt (up to ±45°) is vital for line-by-line parsing. |
| **Binarization** | Converts image to black and white (Otsu, Adaptive, Sauvola). | Otsu works best for clean printed pages. Adaptive and Sauvola are preferred for uneven lighting and handwriting. |
| **Contrast Enhancement** | CLAHE (Contrast Limited Adaptive Histogram Equalization). | Boosts faint characters and handles varying brightness across the page. |
| **Upscale Small Images** | Upscales images with width < 1500px using bicubic interpolation. | Prevents loss of OCR accuracy on low-resolution scans. |

## Limitations

* **Handwriting Accuracy:** Tesseract handwriting recognition accuracy is significantly lower than modern cloud-based VLMs or specialized handwriting engines. Use the `Handwritten / Mixed` preset and ensure high DPI scans.
* **Signature Handling:** Tesseract cannot natively distinguish signature images from handwritten text. Signatures are often interpreted as garbled noise characters.
* **Low-Resolution Scans:** PDF pages scanned at `< 150 DPI` yield poor accuracy. While upscaling helps, it cannot reconstruct missing details.
* **RTL Rendering in PDF:** Right-to-Left (RTL) languages like Arabic may suffer from character ordering issues in the exported PDF due to limitations in `fpdf2`'s standard layout engine.
* **Language Pack Prerequisite:** Tesseract language data packs must be installed on the local system prior to execution; otherwise, execution on those languages will fail.

## Self-Evaluation Results

| Check | Status | Notes |
|-------|--------|-------|
| C-1   | PASS   | `validate_installation()` correctly raises `RuntimeError` with installation instructions. |
| C-2   | PASS   | `validate_languages()` splits language string and checks against `pytesseract.get_languages()`. |
| C-3   | PASS   | Non-PDF files uploaded raise `ValueError` before rendering. |
| C-4   | PASS   | Validates `%PDF` magic bytes in `PDFHandler` initialization. |
| C-5   | PASS   | Empty or wordless pages return `0.0` average confidence without crashing. |
| C-6   | PASS   | Preprocessing steps are idempotent and skippable. |
| C-7   | PASS   | TXT, JSON, and PDF exporters write output for all target pages. |
| C-8   | PASS   | App validates page ranges and displays `st.warning()` if out of bounds. |
| C-9   | PASS   | Files are processed directly in-memory, requiring no disk temp file footprint. |
| C-10  | PASS   | Correctly exports UTF-8 BOM (`utf-8-sig`) for TXT and preserves unicode characters in JSON. |
| P-1   | PASS   | Standard single page executes quickly. |
| P-2   | PASS   | `st.progress()` and `st.status()` are updated page-by-page. |
| P-3   | PASS   | Page previews use `min(dpi, 150)` to prevent browser memory bloating. |
| P-4   | PASS   | Warns users before processing large documents (> 50 pages). |
| U-1   | PASS   | Shows a clean guidance block on initial launch with no uploaded file. |
| U-2   | PASS   | Uploading a new PDF clears session state, resetting previous results. |
| U-3   | PASS   | Download buttons are only rendered when OCR results are available. |
| U-4   | PASS   | Confidence colors map correctly (🟢 >= 80%, 🟡 60-79%, 🔴 < 60%). |
| U-5   | PASS   | PSM dropdown provides one-line descriptive labels. |
| U-6   | PASS   | Presets update individual preprocessing checkboxes instantly. |
| Q-1   | PASS   | Zero absolute or user home paths hardcoded in source code. |
| Q-2   | PASS   | Defaults are managed via `dataclass` instances. |
| Q-3   | PASS   | Code is strictly modularized with clean imports. |
| Q-4   | PASS   | Full type hints are used across all public APIs. |
| Q-5   | PASS   | Comprehensive docstrings document each module and class. |
