import streamlit as st
import cv2
import pytesseract
from PIL import Image
import numpy as np
import torch
import os
import shutil
import platform
import psutil
import time
from src.models.summarizer import ExtractiveSummarizer, TextSummarizer

# ── Tesseract Path ─────────────────────────────
tesseract_path = shutil.which("tesseract")
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
elif os.path.exists(r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# ── Page Config ───────────────────────────────
st.set_page_config(
    page_title="TextGist AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# ── Hide Streamlit UI Elements ────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    [data-testid="stToolbar"] {display: none;}
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .summary-box {
        background: #1e293b;
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 16px;
        font-size: 16px;
        line-height: 1.7;
    }
    h1 { color: #38bdf8 !important; }
    h2 { color: #7dd3fc !important; }
    h3 { color: #bae6fd !important; }
</style>
""", unsafe_allow_html=True)


# ── Hardware Info ─────────────────────────────
def get_hardware_info():
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "ram_total": round(psutil.virtual_memory().total / (1024**3), 1),
        "ram_used": round(psutil.virtual_memory().used / (1024**3), 1),
        "ram_percent": psutil.virtual_memory().percent,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
    }


# ── Image to Text ─────────────────────────────
def image_to_text(image_file):
    image = Image.open(image_file)
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    thresh = cv2.threshold(
        denoised, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    text = pytesseract.image_to_string(thresh)
    return text.strip(), thresh


# ── Load Models ───────────────────────────────
@st.cache_resource
def load_extractive():
    return ExtractiveSummarizer()

@st.cache_resource
def load_abstractive():
    return TextSummarizer("bart")


# ════════════════════════════════════════════
#  MAIN APP
# ════════════════════════════════════════════

st.title("📝 TextGist AI")
st.markdown("**AI-powered Text Summarizer** — Text aur Images dono support karta hai!")
st.divider()

# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    method = st.selectbox(
        "Summarization Method",
        ["Extractive (Fast)", "Abstractive (AI/BART)"]
    )

    length = st.selectbox(
        "Summary Length",
        ["short", "medium", "long"],
        index=1
    )

    num_sentences = st.slider(
        "Number of Sentences",
        min_value=1, max_value=10, value=3
    )

    st.divider()
    st.header("🖥️ Hardware Info")

    hw = get_hardware_info()
    st.metric("CPU Usage", f"{hw['cpu_percent']}%")
    st.metric("RAM Used", f"{hw['ram_used']} GB")
    st.metric("PyTorch", hw['torch_version'])
    st.metric("GPU", hw['gpu_name'])
    st.metric("Python", hw['python_version'])


# ── Tabs — Sirf 2 ─────────────────────────────
tab1, tab2 = st.tabs([
    "📄 Text Input",
    "🖼️ Image Input (OpenCV)"
])


# ════════════════════════════════════════════
#  TAB 1 — TEXT INPUT
# ════════════════════════════════════════════
with tab1:
    st.subheader("Paste Your Text")

    sample = st.selectbox("Load a sample:", [
        "None",
        "Technology News",
        "Science Article",
        "Business News"
    ])

    SAMPLES = {
        "Technology News": """Artificial intelligence has transformed how we interact with technology. Modern AI systems can understand natural language and generate content with remarkable accuracy. These advances have created new opportunities in healthcare, education, and business. Machine learning algorithms can now recognize patterns in vast datasets. Researchers are working to ensure AI development proceeds responsibly. The next decade promises even more dramatic advances as computational power continues to grow exponentially.""",
        "Science Article": """Scientists at MIT have developed a new type of battery that can charge in under five minutes and last for over 20 years. The breakthrough uses a new nanomaterial coating that prevents degradation over repeated charge cycles. Unlike traditional lithium-ion batteries, the new design uses abundant materials that are cheaper and easier to source. Researchers believe this technology could revolutionize electric vehicles and grid-scale energy storage. Clinical trials for commercial applications are expected to begin within two years.""",
        "Business News": """Apple reported record quarterly earnings of 97 billion dollars driven by strong iPhone sales in emerging markets. The company saw a 23 percent year-over-year growth in its services division. CEO Tim Cook highlighted India and Southeast Asia as the fastest growing regions. Despite global supply chain challenges, Apple maintained healthy profit margins through cost optimization strategies. Analysts raised their price targets for Apple stock following the earnings announcement."""
    }

    input_text = SAMPLES.get(sample, "") if sample != "None" else ""

    text = st.text_area(
        "Your Text",
        value=input_text,
        height=200,
        placeholder="Paste any long article or paragraph here..."
    )

    if st.button("✨ Summarize Now", type="primary"):
        if not text or len(text.strip()) < 50:
            st.error("Text too short! Minimum 50 characters needed.")
        else:
            with st.spinner("Generating summary..."):
                start = time.time()
                try:
                    if "Extractive" in method:
                        model = load_extractive()
                        result = model.summarize(text, num_sentences=num_sentences)
                        summary = result["summary"]
                        compression = result.get("compression_ratio", 0)
                    else:
                        model = load_abstractive()
                        result = model.summarize(text, length=length)
                        summary = result["summary"]
                        compression = result["compression_ratio"]

                    elapsed = round((time.time() - start) * 1000, 1)
                    st.success("Summary Generated!")
                    st.markdown(
                        f'<div class="summary-box">{summary}</div>',
                        unsafe_allow_html=True
                    )