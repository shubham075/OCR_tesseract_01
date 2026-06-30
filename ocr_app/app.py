"""Main Streamlit application for Tesseract OCR + PDF Reader."""

import sys
import os
import io
import time
import pandas as pd
import streamlit as st
import pytesseract

# Ensure the parent directory is in the path to import from src
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def _auto_detect_tesseract() -> str:
    """Search common Windows installation paths for tesseract.exe.

    Returns the path to tesseract.exe if found, or empty string if not found.
    """
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.environ.get("APPDATA", ""), "Local", "Programs", "Tesseract-OCR", "tesseract.exe"),
        r"D:\Program Files\Tesseract-OCR\tesseract.exe",
        r"D:\Tesseract-OCR\tesseract.exe",
        r"C:\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if path and os.path.isfile(path):
            return path
    return ""


# Auto-configure Tesseract path once at startup
_detected_tess_path = _auto_detect_tesseract()
if _detected_tess_path:
    pytesseract.pytesseract.tesseract_cmd = _detected_tess_path

from src.pdf_handler import PDFHandler
from src.preprocessor import (
    ImagePreprocessor,
    PreprocessorConfig,
    PRINTED_PRESET,
    HANDWRITING_PRESET,
)
from src.ocr_engine import OCREngine, OCRConfig, PageResult
from src.postprocessor import TextPostprocessor, PostprocessorConfig
from src.exporter import export_txt, export_json, export_pdf
from src.digit_recognizer import DigitRecognizer


# 1. Page Config
st.set_page_config(
    page_title="Tesseract OCR — PDF Reader",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Initialize session state variables
if "pdf_handler" not in st.session_state:
    st.session_state.pdf_handler = None
if "ocr_results" not in st.session_state:
    st.session_state.ocr_results = {}
if "current_page_idx" not in st.session_state:
    st.session_state.current_page_idx = 0
if "last_uploaded_file_name" not in st.session_state:
    st.session_state.last_uploaded_file_name = None
if "batch_log" not in st.session_state:
    st.session_state.batch_log = []


# Custom CSS to improve look & feel
st.markdown(
    """
    <style>
    .reportview-container {
        background: #0F1117;
    }
    .stDeployButton {
        display:none;
    }
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("🔍 Tesseract OCR — PDF Reader")
st.markdown("An offline, local pipeline to digitize and clean PDF documents.")


# 2. Sidebar Configuration Panel
st.sidebar.header("⚙️ OCR Pipeline settings")

# Expander 1: Document Settings
with st.sidebar.expander("📄 Document Settings", expanded=True):
    dpi = st.slider("Render DPI", min_value=150, max_value=600, value=300, step=50)
    page_range = st.selectbox(
        "Page Range Selection",
        ["All Pages", "Single Page", "Page Range"]
    )
    
    page_number = 1
    from_page = 1
    to_page = 1
    
    if page_range == "Single Page":
        page_number = st.number_input(
            "Page Number", min_value=1, value=1, step=1
        )
    elif page_range == "Page Range":
        col1, col2 = st.columns(2)
        with col1:
            from_page = st.number_input(
                "From Page", min_value=1, value=1, step=1
            )
        with col2:
            to_page = st.number_input(
                "To Page", min_value=1, value=1, step=1
            )

# Expander 2: OCR Settings
with st.sidebar.expander("🔬 OCR Settings", expanded=True):
    # Show auto-detected path or allow custom override
    if _detected_tess_path:
        st.success(f"✅ Tesseract detected: `{_detected_tess_path}`")
    else:
        st.warning("⚠️ Tesseract not auto-detected. Install from UB Mannheim or set path below.")

    tesseract_cmd = st.text_input(
        "Custom Tesseract Path (override)",
        value=_detected_tess_path,
        help="Example: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
    )
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    lang_option = st.selectbox(
        "Language Pack",
        ["English only (eng)", "English + Hindi (eng+hin)", "Custom String"]
    )
    
    if lang_option == "English only (eng)":
        language = "eng"
    elif lang_option == "English + Hindi (eng+hin)":
        language = "eng+hin"
    else:
        language = st.text_input(
            "Custom Language String", value="eng",
            help="E.g., 'eng+hin+fra' (ensure language packs are installed)"
        )
        
    psm_modes = {
        3: "3 — Fully automatic (default)",
        4: "4 — Single column of variable sizes",
        6: "6 — Uniform block of text",
        7: "7 — Single text line",
        8: "8 — Single word",
        11: "11 — Sparse text (scattered fields)",
        13: "13 — Raw line (handwriting bypass)",
    }
    psm = st.selectbox(
        "Page Segmentation Mode (PSM)",
        options=list(psm_modes.keys()),
        format_func=lambda x: psm_modes[x],
        index=0
    )
    
    oem_modes = {
        3: "3 — LSTM + Legacy (recommended)",
        1: "1 — LSTM only"
    }
    oem = st.radio(
        "OCR Engine Mode (OEM)",
        options=list(oem_modes.keys()),
        format_func=lambda x: oem_modes[x],
        index=0
    )

    st.divider()
    st.markdown("**🔢 Digit Enhancement (CNN)**")
    enable_digit_cnn = st.checkbox(
        "Enable handwritten digit correction",
        value=False,
        help=(
            "Uses a local MNIST-trained ONNX CNN to re-classify single digits where "
            "Tesseract confidence is low. Downloads a ~26 KB model on first use."
        )
    )
    digit_conf_threshold = st.slider(
        "Correction threshold (Tesseract conf %)",
        min_value=10,
        max_value=90,
        value=60,
        step=5,
        disabled=not enable_digit_cnn,
        help="Digits with Tesseract confidence below this value will be re-checked by CNN."
    )

# Expander 3: Preprocessing
with st.sidebar.expander("🛠 Image Preprocessing", expanded=True):
    doc_type = st.radio(
        "Document Style Presets",
        ["Printed Text", "Handwritten / Mixed"]
    )
    
    # Preset selection syncs with UI
    preset = PRINTED_PRESET if doc_type == "Printed Text" else HANDWRITING_PRESET
    
    # Initialize session state for preprocessor widgets on switch
    if "prev_doc_type" not in st.session_state or st.session_state.prev_doc_type != doc_type:
        st.session_state.prev_doc_type = doc_type
        st.session_state.denoise = preset.denoise
        st.session_state.denoise_strength = preset.denoise_strength
        st.session_state.deskew = preset.deskew
        st.session_state.binarize_method = preset.binarize_method
        st.session_state.contrast_enhance = preset.contrast_enhance
        st.session_state.upscale_if_small = preset.upscale_if_small
        st.session_state.remove_borders = preset.remove_borders

    # Sub-controls
    advanced_options = st.checkbox("Show Advanced Options", value=False)
    
    if advanced_options:
        denoise = st.checkbox("Denoise Image", key="denoise")
        denoise_strength = st.slider("Denoise Strength (h)", 5, 30, key="denoise_strength")
        deskew = st.checkbox("Deskew Text Rotation", key="deskew")
        binarize_method = st.selectbox(
            "Binarization Method",
            ["otsu", "adaptive", "sauvola"],
            key="binarize_method"
        )
        contrast_enhance = st.checkbox("Contrast Enhancement (CLAHE)", key="contrast_enhance")
        upscale_if_small = st.checkbox("Upscale Small Images", key="upscale_if_small")
        remove_borders = st.checkbox("Crop Scan Borders", key="remove_borders")
    else:
        # Use preset values implicitly without displaying widgets
        denoise = st.session_state.denoise
        denoise_strength = st.session_state.denoise_strength
        deskew = st.session_state.deskew
        binarize_method = st.session_state.binarize_method
        contrast_enhance = st.session_state.contrast_enhance
        upscale_if_small = st.session_state.upscale_if_small
        remove_borders = st.session_state.remove_borders

# Expander 4: Postprocessing
with st.sidebar.expander("📝 Postprocessing", expanded=False):
    fix_hyphenation = st.checkbox("Fix Word Hyphenation", value=True)
    remove_extra_whitespace = st.checkbox("Remove Extra Whitespace", value=True)
    normalize_unicode = st.checkbox("Normalize Unicode (NFC)", value=True)
    remove_empty_lines = st.checkbox("Remove Empty Lines", value=False)
    strip_control_chars = st.checkbox("Strip Control Characters", value=True)


# Initialize components config
prep_config = PreprocessorConfig(
    grayscale=True,
    denoise=denoise,
    denoise_strength=denoise_strength,
    deskew=deskew,
    binarize=True,  # Binarization is forced for optimal clean text extraction
    binarize_method=binarize_method,
    contrast_enhance=contrast_enhance,
    contrast_clip=3.0 if doc_type == "Handwritten / Mixed" else 2.0,
    upscale_if_small=upscale_if_small,
    remove_borders=remove_borders,
)

ocr_config = OCRConfig(
    language=language,
    psm=psm,
    oem=oem,
    dpi=dpi
)

post_config = PostprocessorConfig(
    remove_extra_whitespace=remove_extra_whitespace,
    remove_empty_lines=remove_empty_lines,
    fix_hyphenation=fix_hyphenation,
    normalize_unicode=normalize_unicode,
    strip_control_chars=strip_control_chars,
)


# File Uploader
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

# Re-upload handling to reset state
if uploaded_file is not None:
    if st.session_state.last_uploaded_file_name != uploaded_file.name:
        st.session_state.last_uploaded_file_name = uploaded_file.name
        st.session_state.ocr_results = {}
        st.session_state.current_page_idx = 0
        st.session_state.batch_log = []
        
        # Clear old text area widget states to avoid rendering stale/empty text
        for key in list(st.session_state.keys()):
            if key.startswith("text_area_page_"):
                del st.session_state[key]
        
        # Load PDF Handler
        try:
            st.session_state.pdf_handler = PDFHandler(uploaded_file.read())
        except ValueError as e:
            st.error("❌ The uploaded file does not appear to be a valid PDF.")
            st.session_state.pdf_handler = None
            st.stop()
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}. Check the terminal for details.")
            st.session_state.pdf_handler = None
            st.stop()
else:
    st.session_state.pdf_handler = None
    st.session_state.last_uploaded_file_name = None
    st.session_state.ocr_results = {}
    st.session_state.current_page_idx = 0
    st.session_state.batch_log = []
    # Clear old text area widget states
    for key in list(st.session_state.keys()):
        if key.startswith("text_area_page_"):
            del st.session_state[key]
    st.info("👋 Upload a PDF file above to begin.")
    st.stop()


pdf_handler = st.session_state.pdf_handler
if pdf_handler is None:
    st.info("👋 Upload a PDF file above to begin.")
    st.stop()

total_pages = pdf_handler.page_count

# Show a warning if document is large
if total_pages > 50:
    st.warning("⚠️ Large document detected. Processing may take a few minutes.")

# Determine target pages based on selection
pages_to_process = []
if page_range == "All Pages":
    pages_to_process = list(range(total_pages))
elif page_range == "Single Page":
    if page_number < 1 or page_number > total_pages:
        st.warning(f"❌ Page number {page_number} is out of bounds (1-{total_pages}).")
        st.stop()
    pages_to_process = [page_number - 1]
elif page_range == "Page Range":
    if from_page < 1 or to_page > total_pages or from_page > to_page:
        st.warning(f"❌ Page range {from_page}-{to_page} is invalid for a {total_pages}-page document.")
        st.stop()
    pages_to_process = list(range(from_page - 1, to_page))


# Page Action: Run OCR
run_ocr = st.button("🚀 Run OCR Pipeline", type="primary")

if run_ocr:
    # Clear text area widget state for the pages being processed to ensure fresh OCR text is displayed
    for page_idx in pages_to_process:
        key = f"text_area_page_{page_idx}"
        if key in st.session_state:
            del st.session_state[key]
            
    try:
        # Validate installation before running
        engine = OCREngine(ocr_config)
        engine.validate_installation()
        
        # Validate language pack
        missing_langs = engine.validate_languages(language)
        if missing_langs:
            st.error(
                f"❌ Missing Tesseract language packs: {missing_langs}. "
                "See the README for installation instructions."
            )
            st.stop()
            
        preprocessor = ImagePreprocessor(prep_config)
        postprocessor = TextPostprocessor(post_config)
        
        # Batch vs Single Page Execution UI
        digit_rec = DigitRecognizer(confidence_threshold=digit_conf_threshold) if enable_digit_cnn else None

        if len(pages_to_process) > 1:
            with st.status("Running Batch OCR Pipeline...") as status:
                progress_bar = st.progress(0)
                st.session_state.batch_log = []
                cnn_corrections_total = 0

                for idx, page_idx in enumerate(pages_to_process):
                    status.update(label=f"Processing page {page_idx + 1} of {total_pages}...")

                    # 1. Render at full OCR DPI
                    img = pdf_handler.render_page(page_idx, dpi=dpi)
                    # 2. Preprocess
                    clean_img = preprocessor.process(img)
                    # 3. OCR Engine
                    result = engine.run_page(clean_img, page_idx)
                    # 4. Postprocess Text
                    result.raw_text = postprocessor.clean(result.raw_text)
                    # 5. Optional CNN digit correction
                    cnn_fixes = 0
                    if digit_rec is not None:
                        try:
                            corrected, corrections = digit_rec.correct_digit_regions(
                                result.raw_text, result.word_data, clean_img
                            )
                            result.raw_text = corrected
                            cnn_fixes = len(corrections)
                            cnn_corrections_total += cnn_fixes
                        except Exception as cnn_err:
                            st.warning(f"CNN digit correction skipped on page {page_idx + 1}: {cnn_err}")

                    st.session_state.ocr_results[page_idx] = result
                    log_entry = {
                        "Page": page_idx + 1,
                        "Confidence": f"{result.confidence:.1f}%",
                        "Words": len(result.word_data[result.word_data["conf"] > -1]) if not result.word_data.empty else 0,
                        "Time (ms)": f"{result.processing_ms:.0f}",
                    }
                    if enable_digit_cnn:
                        log_entry["CNN Fixes"] = cnn_fixes
                    st.session_state.batch_log.append(log_entry)

                    progress_bar.progress((idx + 1) / len(pages_to_process))

                completion_msg = "✅ OCR Batch completed successfully!"
                if enable_digit_cnn:
                    completion_msg += f" ({cnn_corrections_total} digit corrections applied)"
                status.update(label=completion_msg, state="complete")
        else:
            with st.spinner("Processing single page..."):
                page_idx = pages_to_process[0]
                img = pdf_handler.render_page(page_idx, dpi=dpi)
                clean_img = preprocessor.process(img)
                result = engine.run_page(clean_img, page_idx)
                result.raw_text = postprocessor.clean(result.raw_text)
                # Optional CNN digit correction
                if digit_rec is not None:
                    try:
                        corrected, corrections = digit_rec.correct_digit_regions(
                            result.raw_text, result.word_data, clean_img
                        )
                        result.raw_text = corrected
                        if corrections:
                            st.info(f"🔢 CNN corrected {len(corrections)} digit(s) on this page.")
                    except Exception as cnn_err:
                        st.warning(f"CNN digit correction skipped: {cnn_err}")
                st.session_state.ocr_results[page_idx] = result
                st.success("✅ Page processed successfully!")
                
    except RuntimeError as e:
        err_msg = str(e)
        if "Tesseract binary not found" in err_msg:
            st.error("❌ Tesseract is not installed or not in PATH. See setup instructions.")
        elif "poppler-utils is not installed" in err_msg:
            st.error("❌ poppler-utils is missing. See setup instructions.")
        else:
            st.error(f"❌ System dependency error: {err_msg}")
    except ValueError as e:
        st.error(f"❌ The uploaded file does not appear to be a valid PDF.")
    except Exception as e:
        st.error(f"❌ An unexpected error occurred: {str(e)}. Check the terminal for details.")


# 3. Main Area Layout (Two columns)
col_left, col_right = st.columns([1, 1])

# Handle page navigation state
if total_pages > 1:
    col_nav1, col_nav2, col_nav3 = col_left.columns([1, 2, 1])
    with col_nav1:
        if st.button("⬅️ Previous", disabled=st.session_state.current_page_idx == 0):
            st.session_state.current_page_idx -= 1
            st.rerun()
    with col_nav2:
        st.markdown(
            f"<h5 style='text-align: center;'>Page {st.session_state.current_page_idx + 1} of {total_pages}</h5>",
            unsafe_allow_html=True
        )
    with col_nav3:
        if st.button("Next ➡️", disabled=st.session_state.current_page_idx == total_pages - 1):
            st.session_state.current_page_idx += 1
            st.rerun()

current_page = st.session_state.current_page_idx

with col_left:
    st.subheader("PDF Page Preview")
    try:
        # Display preview at a low DPI (min of dpi or 150) to prevent memory bloat
        preview_dpi = min(dpi, 150)
        preview_img = pdf_handler.render_page(current_page, dpi=preview_dpi)
        st.image(preview_img, width='stretch')
    except Exception as e:
        st.error(f"Failed to render page preview: {str(e)}")

with col_right:
    st.subheader("Extracted Text")
    
    # Check if we have OCR result for current page
    page_result = st.session_state.ocr_results.get(current_page)
    
    if page_result:
        # Confidence Badge Colors
        conf = page_result.confidence
        if conf >= 80:
            badge_color = "green"
            badge_text = f"🟢 Confidence: {conf:.1f}% (High)"
        elif conf >= 60:
            badge_color = "orange"
            badge_text = f"🟡 Confidence: {conf:.1f}% (Medium)"
        else:
            badge_color = "red"
            badge_text = f"🔴 Confidence: {conf:.1f}% (Low)"
            
        # Display Stats side-by-side
        stat_col1, stat_col2 = st.columns(2)
        with stat_col1:
            st.markdown(f"**{badge_text}**")
        with stat_col2:
            st.markdown(f"**⏱ OCR Time:** {page_result.processing_ms:.0f} ms")
            
        # Allow editing extracted text
        edited_text = st.text_area(
            "Extracted Text (Editable)",
            value=page_result.raw_text,
            height=500,
            key=f"text_area_page_{current_page}"
        )
        # Update raw text in session state if user edits it
        st.session_state.ocr_results[current_page].raw_text = edited_text
    else:
        st.info("ℹ️ Run the OCR pipeline to extract text for this page.")
        st.text_area("Extracted Text", value="", height=500, disabled=True)


# 4. Summary Table & Export Panel (rendered below main view)
if st.session_state.ocr_results:
    st.markdown("---")
    
    # Display batch log table if multiple pages were processed
    if st.session_state.batch_log:
        st.subheader("📋 Batch Processing Summary")
        df_summary = pd.DataFrame(st.session_state.batch_log)
        st.dataframe(df_summary, width='stretch', hide_index=True)
        
    st.subheader("📥 Export Documents")
    
    # Get all sorted results for exporting
    export_results = [
        st.session_state.ocr_results[p] 
        for p in sorted(st.session_state.ocr_results.keys())
    ]
    
    postprocessor = TextPostprocessor(post_config)
    
    # Export byte conversions
    txt_data = export_txt(export_results, postprocessor)
    json_data = export_json(export_results, original_filename=uploaded_file.name)
    
    try:
        pdf_data = export_pdf(export_results, original_filename=uploaded_file.name)
        pdf_enabled = True
    except Exception as e:
        pdf_data = b""
        pdf_enabled = False
        st.warning(f"PDF generation contains warnings: {str(e)}")

    # Display export buttons in three columns
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    
    with btn_col1:
        st.download_button(
            label="⬇️ Download TXT",
            data=txt_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_ocr.txt",
            mime="text/plain",
            width='stretch'
        )
        
    with btn_col2:
        st.download_button(
            label="⬇️ Download JSON",
            data=json_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_ocr.json",
            mime="application/json",
            width='stretch'
        )
        
    with btn_col3:
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_data,
            file_name=f"{os.path.splitext(uploaded_file.name)[0]}_ocr.pdf",
            mime="application/pdf",
            disabled=not pdf_enabled,
            width='stretch'
        )
