# PROMPT.md — Tesseract OCR + Streamlit App for PDF Documents

> **Purpose:** A self-contained execution prompt for an AI coding agent to build,
> test, and deliver a production-ready OCR pipeline. Follow every section in order.
> Do not skip sections. Do not ask the user for clarification unless an **[ASK]**
> marker explicitly appears.

---

## 1. CONTEXT — Background & Boundaries

### 1.1 What You Are Building

A **local, offline Streamlit web application** that accepts a PDF file (possibly
scanned or photographed), extracts text from every page using **Tesseract OCR**,
and presents the structured output in a clean, interactive UI. The app must handle:

- Digitally generated PDFs with embedded raster images
- Fully scanned PDFs (every page is a rasterised image)
- Pages containing **handwritten** text (forms, notes, annotations, signatures)
- **Bilingual** pages — two languages may appear on the same page (e.g., English +
  Hindi, English + Arabic, English + Chinese Simplified)

### 1.2 Canonical Tech Stack

| Layer | Library / Tool | Pinned Version |
|---|---|---|
| Language | Python | ≥ 3.10 |
| OCR Engine | Tesseract (system binary) | ≥ 5.3 |
| OCR Python Binding | `pytesseract` | ≥ 0.3.10 |
| PDF → Image | `pdf2image` + `poppler-utils` | ≥ 1.17 |
| Image Processing | `Pillow` | ≥ 10.0 |
| Advanced Preprocessing | `opencv-python-headless` | ≥ 4.9 |
| Numeric Ops | `numpy` | ≥ 1.26 |
| Frontend | `streamlit` | ≥ 1.35 |
| PDF Metadata | `pypdf` | ≥ 4.0 |
| Export | `fpdf2` | ≥ 2.7 |

**No cloud OCR APIs. No proprietary engines. Everything runs locally.**

### 1.3 Hard Boundaries (Never Violate)

- **B-1:** Do NOT use any OCR engine other than Tesseract (no EasyOCR, no
  PaddleOCR, no cloud Vision APIs).
- **B-2:** Do NOT store uploaded PDF files permanently on disk. Write only to a
  `tempfile.TemporaryDirectory` that is cleaned up after the session ends.
- **B-3:** The Streamlit app must run with a single command: `streamlit run app.py`.
  No Docker, no server setup required from the user.
- **B-4:** All configuration (DPI, PSM mode, language pairs) must be adjustable
  through the Streamlit sidebar — no hardcoded magic numbers in business logic.
- **B-5:** The codebase must be split into modules (see §2.3). No monolithic
  single-file solutions.

### 1.4 Supported Bilingual Pairs (Out of the Box)

The UI must offer at minimum these language combos from a dropdown:

```
English only          → eng
English + Hindi       → eng+hin
```

> The user may also type a custom Tesseract language string (e.g., `eng+hin+ben`)
> if they have additional language packs installed.

---

## 2. EXECUTION PROTOCOL — How to Do the Work

Execute the following phases **strictly in order**. Mark each phase complete before
proceeding to the next.

---

### PHASE 0 — Project Scaffold

**0.1** Create the following directory tree exactly:

```
ocr_app/
├── app.py                  ← Streamlit entry point
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml         ← Theme + server settings
└── src/
    ├── __init__.py
    ├── pdf_handler.py      ← PDF ingestion & page rendering
    ├── preprocessor.py     ← Image preprocessing pipeline
    ├── ocr_engine.py       ← Tesseract wrapper & config
    ├── postprocessor.py    ← Text cleaning & structuring
    └── exporter.py         ← TXT / JSON / PDF export
```

**0.2** Create `requirements.txt` with all packages from §1.2, each pinned to the
minimum version shown. Add a comment block at the top:

```
# System prerequisite (install separately):
# sudo apt-get install tesseract-ocr poppler-utils
# macOS: brew install tesseract poppler
# Windows: install Tesseract installer from UB Mannheim; add to PATH
```

**0.3** Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor      = "#4F8EF7"
backgroundColor   = "#0F1117"
secondaryBackgroundColor = "#1E2130"
textColor         = "#FAFAFA"
font              = "sans serif"

[server]
maxUploadSize = 100   # MB
```

---

### PHASE 1 — PDF Handler (`src/pdf_handler.py`)

**Responsibilities:** Accept a file-like object (or bytes), validate it is a PDF,
extract metadata, and convert each page to a PIL Image at the requested DPI.

**1.1** Implement `class PDFHandler`:

```python
# Public API that must exist:
PDFHandler(file_bytes: bytes)
    .page_count: int
    .metadata: dict          # title, author, creator, page_count
    .render_page(
        page_number: int,    # 0-indexed
        dpi: int = 300
    ) -> PIL.Image.Image
    .render_all_pages(
        dpi: int = 300,
        progress_callback: Callable[[int, int], None] | None = None
    ) -> list[PIL.Image.Image]
```

**1.2** Use `pdf2image.convert_from_bytes()` for rasterisation. Do not use
`pypdf` for rasterisation — only use it for metadata extraction.

**1.3** If `pdf2image` raises `PDFInfoNotInstalledError`, catch it and raise a
clear `RuntimeError` with the message:
> `"poppler-utils is not installed. Run: sudo apt-get install poppler-utils"`

**1.4** Validate that the file starts with `%PDF` magic bytes before attempting
conversion. Raise `ValueError("Uploaded file is not a valid PDF.")` otherwise.

---

### PHASE 2 — Image Preprocessor (`src/preprocessor.py`)

**Responsibilities:** Apply a configurable preprocessing pipeline to a PIL Image
before passing it to Tesseract. Preprocessing dramatically improves accuracy on
scanned and handwritten documents.

**2.1** Implement `class ImagePreprocessor`:

```python
# Public API:
ImagePreprocessor(config: PreprocessorConfig)
    .process(image: PIL.Image.Image) -> PIL.Image.Image
    .process_batch(images: list[PIL.Image.Image]) -> list[PIL.Image.Image]
```

**2.2** Implement `PreprocessorConfig` as a dataclass:

```python
@dataclass
class PreprocessorConfig:
    grayscale:          bool = True
    denoise:            bool = True
    denoise_strength:   int  = 10        # OpenCV h parameter for fastNlMeansDenoising
    deskew:             bool = True
    deskew_threshold:   float = 0.5      # degrees; skip if rotation < threshold
    binarize:           bool = True
    binarize_method:    str  = "otsu"    # "otsu" | "adaptive" | "sauvola"
    contrast_enhance:   bool = True
    contrast_clip:      float = 2.0      # CLAHE clip limit
    upscale_if_small:   bool = True
    upscale_min_width:  int  = 1500      # px; upscale if image width < this
    remove_borders:     bool = False     # crop black/white scan borders
```

**2.3** Implement each step as a **private method** on `ImagePreprocessor`. Each
step must:
- Accept and return a `numpy.ndarray` (BGR for OpenCV)
- Be idempotent: calling it twice should not degrade quality
- Be skippable via its corresponding `bool` flag

**2.4** Deskew implementation: Use OpenCV's `minAreaRect` on the binarised image
to detect the dominant text angle, then rotate using `cv2.warpAffine`. Clamp
rotation to ±45°.

**2.5** Upscaling: Use `cv2.INTER_CUBIC` for upscaling. Never use
`cv2.INTER_AREA` for upscaling (it is only appropriate for downscaling).

**2.6** Expose a `HANDWRITING_PRESET` and `PRINTED_PRESET` as module-level
constants:

```python
HANDWRITING_PRESET = PreprocessorConfig(
    denoise=True,
    denoise_strength=15,
    deskew=True,
    binarize=True,
    binarize_method="adaptive",   # better for uneven lighting in handwriting
    contrast_enhance=True,
    contrast_clip=3.0,
)

PRINTED_PRESET = PreprocessorConfig(
    denoise=True,
    denoise_strength=10,
    binarize=True,
    binarize_method="otsu",
    contrast_enhance=False,
)
```

---

### PHASE 3 — OCR Engine (`src/ocr_engine.py`)

**Responsibilities:** Wrap pytesseract, manage Tesseract configuration, and return
structured per-page results.

**3.1** Implement `class OCREngine`:

```python
# Public API:
OCREngine(config: OCRConfig)
    .run_page(
        image: PIL.Image.Image,
        page_number: int
    ) -> PageResult
    .run_batch(
        images: list[PIL.Image.Image],
        progress_callback: Callable[[int, int], None] | None = None
    ) -> list[PageResult]
    .validate_installation() -> None    # raises if tesseract binary not found
    .validate_languages(lang_str: str) -> list[str]  # returns missing packs
```

**3.2** Implement `OCRConfig` as a dataclass:

```python
@dataclass
class OCRConfig:
    language:        str = "eng"
    psm:             int = 3      # Page Segmentation Mode (Tesseract --psm)
    oem:             int = 3      # OCR Engine Mode (LSTM + legacy)
    dpi:             int = 300
    extra_config:    str = ""     # any extra --tessdata-dir or custom flags
```

**3.3** Implement `PageResult` as a dataclass:

```python
@dataclass
class PageResult:
    page_number:   int
    raw_text:      str
    word_data:     pd.DataFrame   # from pytesseract.image_to_data()
    confidence:    float          # mean confidence of words with conf > 0
    language:      str
    psm_used:      int
    processing_ms: float
```

**3.4** PSM Mode Reference — document this table in a module docstring:

| PSM | Best For |
|-----|----------|
| 3   | Fully automatic (default for mixed pages) |
| 4   | Single column of text |
| 6   | Uniform block of text |
| 7   | Single text line |
| 8   | Single word |
| 11  | Sparse text — good for forms with scattered fields |
| 13  | Raw line — useful for handwriting one line at a time |

**3.5** Confidence Calculation: Filter `word_data` to rows where `conf > -1`
(Tesseract outputs `-1` for non-word elements). Compute `mean(conf)` over the
remaining rows. Store in `PageResult.confidence`.

**3.6** If `pytesseract.get_tesseract_version()` raises `TesseractNotFoundError`,
catch it in `validate_installation()` and raise:
> `RuntimeError("Tesseract binary not found. Install it and add to PATH.")`

---

### PHASE 4 — Postprocessor (`src/postprocessor.py`)

**Responsibilities:** Clean raw OCR text and optionally structure it.

**4.1** Implement `class TextPostprocessor`:

```python
# Public API:
TextPostprocessor(config: PostprocessorConfig)
    .clean(text: str) -> str
    .clean_batch(results: list[PageResult]) -> list[PageResult]
    .merge_to_document(results: list[PageResult]) -> str
```

**4.2** Implement `PostprocessorConfig`:

```python
@dataclass
class PostprocessorConfig:
    remove_extra_whitespace:  bool = True
    remove_empty_lines:       bool = False   # keep paragraph spacing
    fix_hyphenation:          bool = True    # rejoin words split across lines
    normalize_unicode:        bool = True    # NFC normalization
    strip_control_chars:      bool = True
```

**4.3** `fix_hyphenation`: Match pattern `r'(\w+)-\n(\w+)'` and rejoin as
`r'\1\2\n'` — only apply when `remove_extra_whitespace` is also True.

**4.4** `merge_to_document`: Join all page texts with a page separator:

```
[--- Page 1 ---]
<text>

[--- Page 2 ---]
<text>
...
```

---

### PHASE 5 — Exporter (`src/exporter.py`)

**5.1** Implement three export functions:

```python
def export_txt(results: list[PageResult], postprocessor: TextPostprocessor) -> bytes:
    """Returns UTF-8 encoded plain text."""

def export_json(results: list[PageResult]) -> bytes:
    """Returns structured JSON with per-page metadata and text."""

def export_pdf(results: list[PageResult], original_filename: str) -> bytes:
    """Returns a searchable text-layer PDF using fpdf2."""
```

**5.2** JSON schema for `export_json`:

```json
{
  "source_file": "<original filename>",
  "total_pages": 5,
  "language": "eng+hin",
  "pages": [
    {
      "page_number": 1,
      "text": "...",
      "confidence": 84.3,
      "psm_used": 6,
      "processing_ms": 1203.4
    }
  ]
}
```

**5.3** For `export_pdf` use `fpdf2`. Each page: set font to a Unicode-capable
font (include DejaVu via `fpdf2`'s bundled fonts), add page number footer, write
the cleaned text using `multi_cell`.

---

### PHASE 6 — Streamlit App (`app.py`)

Build the UI in this exact layout:

#### 6.1 Page Config

```python
st.set_page_config(
    page_title="Tesseract OCR — PDF Reader",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

#### 6.2 Sidebar — Configuration Panel

Group controls under expanders:

**📄 Document Settings**
- `dpi` slider: 150 / 200 / 300 / 400 / 600 (default 300)
- `page_range` selectbox: "All Pages", "Single Page", "Page Range"
  - If "Single Page": show `page_number` number_input
  - If "Page Range": show `from_page` and `to_page` number_inputs

**🔬 OCR Settings**
- `language` selectbox (the list from §1.4) + text_input for custom string
- `psm` selectbox (modes from §3.4, shown as `3 — Fully automatic (default)`)
- `oem` radio: `3 — LSTM + Legacy (recommended)` / `1 — LSTM only`

**🛠 Preprocessing**
- `document_type` radio: `Printed Text` / `Handwritten / Mixed` (sets preset)
- Advanced toggle (expander):
  - `denoise` checkbox
  - `denoise_strength` slider 5–30
  - `deskew` checkbox
  - `binarize_method` selectbox: otsu / adaptive / sauvola
  - `contrast_enhance` checkbox
  - `upscale_if_small` checkbox

**📝 Postprocessing**
- `fix_hyphenation` checkbox
- `remove_extra_whitespace` checkbox
- `normalize_unicode` checkbox

#### 6.3 Main Area Layout (two columns, ratio 1:1)

```
[LEFT COLUMN]          [RIGHT COLUMN]
PDF Page Preview       Extracted Text
(st.image)             (st.text_area, height=600)

                       Confidence Badge
                       Processing Time Badge
```

**Left column:** Render the current page as an image using `PDFHandler.render_page()`
at `min(dpi, 150)` DPI for display only (not OCR). Add a page navigation widget
(`Previous | Page N of M | Next`) below the preview.

**Right column:** Display the `PageResult.raw_text` in an `st.text_area`. Show a
color-coded confidence badge:
- 🟢 ≥ 80% confidence
- 🟡 60–79%
- 🔴 < 60%

Show processing time as `⏱ {ms:.0f} ms`.

#### 6.4 Progress & Status

- Use `st.progress()` + `st.status()` during batch OCR runs.
- Show a spinner during single-page runs.
- Display a summary table after batch completion:

```
| Page | Confidence | Words | Time (ms) |
|------|------------|-------|-----------|
```

Use `st.dataframe()` with `use_container_width=True`.

#### 6.5 Export Panel (below the summary table)

Three `st.download_button` elements side by side:

```
[ ⬇ Download TXT ]   [ ⬇ Download JSON ]   [ ⬇ Download PDF ]
```

Only render these buttons **after** OCR has been run at least once. Use
`st.session_state` to cache `list[PageResult]`.

#### 6.6 Error Handling in the UI

Wrap the entire OCR pipeline call in `try/except`. Map exceptions to Streamlit
messages:

| Exception | `st.error()` message |
|---|---|
| `ValueError` (bad PDF) | "❌ The uploaded file does not appear to be a valid PDF." |
| `RuntimeError` (Tesseract not found) | "❌ Tesseract is not installed or not in PATH. See setup instructions." |
| `RuntimeError` (poppler missing) | "❌ poppler-utils is missing. See setup instructions." |
| Any other `Exception` | "❌ An unexpected error occurred: {e}. Check the terminal for details." |

Never let the app crash to a Python traceback visible to the user.

---

### PHASE 7 — README.md

Include:
- One-command install block (pip + system packages for Ubuntu, macOS, Windows)
- One-command run block
- Screenshot placeholder (`![Screenshot](docs/screenshot.png)`)
- Tesseract language pack installation guide for all 9 language pairs in §1.4
- Table of all sidebar settings with their effect on OCR quality
- Known limitations (see §3 Built-in Critic)

---

## 3. BUILT-IN CRITIC — Self-Evaluation Before Delivery

Before marking any phase complete, run the following checks. If any check **FAILS**,
fix the issue and re-evaluate the full list before proceeding.

### 3.1 Correctness Checks

| # | Check | Pass Condition |
|---|---|---|
| C-1 | Tesseract binary detection | `validate_installation()` raises clear error with install instructions when binary is absent |
| C-2 | Language pack validation | `validate_languages("eng+xyz")` returns `["xyz"]` (missing pack) without crashing |
| C-3 | Non-PDF upload rejection | Uploading a `.jpg` or `.txt` raises `ValueError` before rasterisation is attempted |
| C-4 | `%PDF` magic byte check | A file named `.pdf` but containing random bytes is rejected |
| C-5 | Confidence formula | A blank white page should yield `confidence = 0.0`, not `NaN` or a crash |
| C-6 | Preprocessing idempotency | Running `process()` twice on the same image should not degrade text quality |
| C-7 | Export completeness | All three export formats must include content for every processed page |
| C-8 | Page range validation | Requesting page 999 of a 5-page PDF shows `st.warning()`, not a crash |
| C-9 | Temp file cleanup | After session, no PDF files remain in `/tmp` (use `atexit` or context manager) |
| C-10 | Unicode export | Hindi/Arabic/Chinese text exports correctly in TXT and JSON (UTF-8 BOM for TXT) |

### 3.2 Performance Checks

| # | Check | Pass Condition |
|---|---|---|
| P-1 | Single A4 page at 300 DPI, printed text | OCR completes in < 10 seconds on a modern CPU |
| P-2 | Progress bar | For a 10-page PDF, `st.progress()` updates visibly (not just 0% → 100%) |
| P-3 | Preview render DPI | Page preview uses ≤ 150 DPI (never the OCR DPI) to avoid memory bloat |
| P-4 | Large PDF guard | If page count > 50, show `st.warning("Large document detected. Processing may take a few minutes.")` |

### 3.3 UX Checks

| # | Check | Pass Condition |
|---|---|---|
| U-1 | Fresh state | On first load (no file uploaded), the app shows an upload prompt, not an empty shell |
| U-2 | Re-upload | Uploading a second PDF resets all `st.session_state` OCR results cleanly |
| U-3 | Download buttons disabled pre-OCR | Export buttons are absent (not just disabled) before any OCR has run |
| U-4 | Confidence colour logic | All three colour states (green/yellow/red) are reachable and display correctly |
| U-5 | PSM help text | Each PSM option in the sidebar shows a one-line description (not just a number) |
| U-6 | Sidebar presets | Switching `document_type` to "Handwritten / Mixed" automatically updates preprocessing toggles |

### 3.4 Code Quality Checks

| # | Check | Pass Condition |
|---|---|---|
| Q-1 | No hardcoded paths | Zero occurrences of `/home/`, `/tmp/ocr`, or absolute paths in source code |
| Q-2 | No hardcoded magic numbers | DPI default, denoise strength, etc. live in `dataclass` defaults, not inline literals |
| Q-3 | Module separation | `app.py` imports only from `src/`. No OCR logic lives in `app.py` |
| Q-4 | Type hints | All public functions and `__init__` methods have complete type annotations |
| Q-5 | Docstrings | Every public class and method has a one-line docstring minimum |

### 3.5 Known Risk Register

Document these explicitly in `README.md` under a **Limitations** section:

| Risk | Severity | Mitigation |
|---|---|---|
| Handwriting recognition accuracy for Tesseract is significantly lower than modern VLM-based OCR | High | Use HANDWRITING_PRESET; advise user to upgrade to PaddleOCR or olmOCR for critical handwriting use cases |
| Tesseract cannot natively detect or ignore signature images — signatures will be interpreted as noise characters | Medium | Recommend manual region selection or signature detection pre-step |
| Very low-DPI scans (< 150 DPI) yield poor results even with preprocessing | Medium | Upscale flag mitigates but does not eliminate |
| RTL language rendering in exported PDF may have visual ordering issues with `fpdf2` | Medium | Note in README; suggest alternative PDF viewers |
| Tesseract language packs must be manually installed by the user | High | Provide exact install commands for all 9 language pairs in README |

---

## 4. EXIT CONDITION — Measurable Bar That Ends the Loop

The implementation is **complete and deliverable** when ALL of the following are
simultaneously true. No partial credit.

### 4.1 File Completeness

- [ ] `ocr_app/app.py` exists and is non-empty
- [ ] `ocr_app/requirements.txt` exists with ≥ 8 packages
- [ ] `ocr_app/src/pdf_handler.py` exists
- [ ] `ocr_app/src/preprocessor.py` exists
- [ ] `ocr_app/src/ocr_engine.py` exists
- [ ] `ocr_app/src/postprocessor.py` exists
- [ ] `ocr_app/src/exporter.py` exists
- [ ] `ocr_app/README.md` exists with install + run instructions
- [ ] `ocr_app/.streamlit/config.toml` exists

### 4.2 Static Analysis Gate

Run the following and confirm **zero errors**:

```bash
python -m py_compile app.py src/*.py
python -c "import ast; [ast.parse(open(f).read()) for f in ['app.py','src/pdf_handler.py','src/preprocessor.py','src/ocr_engine.py','src/postprocessor.py','src/exporter.py']]"
```

No `SyntaxError`, no `IndentationError`, no `ImportError` on stdlib-only imports.

### 4.3 Interface Contract Gate

Run the following import + instantiation checks (mocked, no Tesseract required):

```python
from src.pdf_handler   import PDFHandler
from src.preprocessor  import ImagePreprocessor, PreprocessorConfig, HANDWRITING_PRESET, PRINTED_PRESET
from src.ocr_engine    import OCREngine, OCRConfig, PageResult
from src.postprocessor import TextPostprocessor, PostprocessorConfig
from src.exporter      import export_txt, export_json, export_pdf

cfg = OCRConfig()
assert cfg.language == "eng"
assert cfg.psm == 3

pcfg = PreprocessorConfig()
assert pcfg.binarize_method in ("otsu", "adaptive", "sauvola")
```

All assertions must pass.

### 4.4 Critic Gate

All 25 checks in §3 (C-1 through C-10, P-1 through P-4, U-1 through U-6,
Q-1 through Q-5) must be in state **PASS** or **N/A (mock environment)**.

Document each check result as a comment block at the bottom of `README.md`:

```markdown
## Self-Evaluation Results
| Check | Status | Notes |
|-------|--------|-------|
| C-1   | PASS   |       |
...
```

### 4.5 End-to-End Smoke Test Gate

If Tesseract is available in the execution environment:

```bash
cd ocr_app
pip install -r requirements.txt
# generate a minimal 1-page test PDF
python -c "
from fpdf2 import FPDF
pdf = FPDF(); pdf.add_page(); pdf.set_font('Helvetica', size=16)
pdf.cell(0, 10, 'OCR smoke test page — Hello World'); pdf.output('test.pdf')
"
streamlit run app.py &
sleep 5
curl -s http://localhost:8501 | grep -q 'Tesseract OCR' && echo 'UI_LIVE=PASS' || echo 'UI_LIVE=FAIL'
```

Expected terminal output: `UI_LIVE=PASS`

### 4.6 Documentation Gate

`README.md` must contain all of the following sections (checked by heading text):

- [ ] `## Installation`
- [ ] `## Running the App`
- [ ] `## Installing Tesseract Language Packs`
- [ ] `## Configuration Reference`
- [ ] `## Limitations`
- [ ] `## Self-Evaluation Results`

---

## APPENDIX A — Tesseract PSM Quick Reference

```
0  Orientation and script detection (OSD) only.
1  Automatic page segmentation with OSD.
2  Automatic page segmentation, but no OSD, or OCR.
3  Fully automatic page segmentation, but no OSD. (DEFAULT)
4  Assume a single column of text of variable sizes.
5  Assume a single uniform block of vertically aligned text.
6  Assume a single uniform block of text.
7  Treat the image as a single text line.
8  Treat the image as a single word.
9  Treat the image as a single word in a circle.
10 Treat the image as a single character.
11 Sparse text. Find as much text as possible in no particular order.
12 Sparse text with OSD.
13 Raw line. Treat the image as a single text line, bypassing hacks that are Tesseract-specific.
```

## APPENDIX B — Language Pack Install Commands

```bash
# Ubuntu / Debian
sudo apt-get install tesseract-ocr-hin   # Hindi
sudo apt-get install tesseract-ocr-ara   # Arabic
sudo apt-get install tesseract-ocr-chi-sim  # Chinese Simplified
sudo apt-get install tesseract-ocr-chi-tra  # Chinese Traditional
sudo apt-get install tesseract-ocr-fra   # French
sudo apt-get install tesseract-ocr-deu   # German
sudo apt-get install tesseract-ocr-spa   # Spanish
sudo apt-get install tesseract-ocr-jpn   # Japanese

# macOS (Homebrew)
brew install tesseract-lang

# Verify installed languages
tesseract --list-langs
```

---

*End of PROMPT.md — Total sections: Context, Execution Protocol, Built-in Critic, Exit Condition.*
