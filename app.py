"""Streamlit web app for splitting double-spread book-scan PDFs.

Upload a PDF, preview each spread with the auto-detected split line drawn in
red, fine-tune any page with a slider (or mark it "don't split" for covers),
then download the resulting single-page PDF.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageDraw

import splitter

PREVIEW_DPI = 90


st.set_page_config(page_title="PDF Spread Splitter", layout="wide")
st.title("📖 PDF Double-Spread Page Splitter")
st.caption(
    "Split scanned book spreads into individual pages. "
    "The gutter is auto-detected; adjust any page below."
)


@st.cache_data(show_spinner=False)
def _render_preview(pdf_bytes: bytes, pno: int, dpi: int = PREVIEW_DPI) -> Image.Image:
    """Render a single page of the uploaded PDF to a PIL image."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[pno].get_pixmap(dpi=dpi)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()


@st.cache_data(show_spinner="Detecting gutters…")
def _auto_ratios(pdf_bytes: bytes, search_frac: float) -> list[float]:
    """Auto-detect a split ratio for every page of the uploaded PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return [
            splitter.detect_split_ratio(doc[p], search_frac=search_frac)
            for p in range(doc.page_count)
        ]
    finally:
        doc.close()


def _draw_split_line(img: Image.Image, ratio: float) -> Image.Image:
    """Return a copy of ``img`` with a red vertical split line at ``ratio``."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    x = int(ratio * out.width)
    line_w = max(2, out.width // 300)
    draw.line([(x, 0), (x, out.height)], fill=(255, 0, 0), width=line_w)
    return out


uploaded = st.file_uploader("Upload a PDF of double-spread scans", type="pdf")

if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded.getvalue()
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
page_count = doc.page_count
doc.close()

with st.sidebar:
    st.header("Settings")
    search_frac = st.slider(
        "Gutter search band (½-width)",
        min_value=0.05,
        max_value=0.45,
        value=0.20,
        step=0.05,
        help="How far from the center to look for the gutter.",
    )
    cover_first = st.checkbox(
        "First page is a cover (don't split)", value=False
    )
    st.write(f"**{page_count}** page(s) in upload.")

auto = _auto_ratios(pdf_bytes, search_frac)

# Per-page state lives in session_state so slider tweaks persist across reruns.
st.subheader("Review & adjust pages")
overrides: list[splitter.SplitPlan] = []

for pno in range(page_count):
    img = _render_preview(pdf_bytes, pno)
    default_split = not (cover_first and pno == 0)

    col_img, col_ctrl = st.columns([3, 1])
    with col_ctrl:
        st.markdown(f"**Page {pno + 1}**")
        split = st.checkbox(
            "Split this page",
            value=default_split,
            key=f"split_{pno}",
        )
        ratio = st.slider(
            "Split position",
            min_value=0.05,
            max_value=0.95,
            value=float(round(auto[pno], 3)),
            step=0.005,
            key=f"ratio_{pno}",
            disabled=not split,
        )
    with col_img:
        st.image(
            _draw_split_line(img, ratio) if split else img,
            use_container_width=True,
        )

    overrides.append(splitter.SplitPlan(split=split, ratio=ratio))

st.divider()

if st.button("✂️ Split and build PDF", type="primary"):
    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out = splitter.split_document(src, overrides)
        try:
            buf = out.tobytes(deflate=True, garbage=3)
            n = out.page_count
        finally:
            out.close()
    finally:
        src.close()

    st.success(f"Done — produced {n} pages.")
    out_name = uploaded.name.rsplit(".", 1)[0] + "_split.pdf"
    st.download_button(
        "⬇️ Download split PDF",
        data=buf,
        file_name=out_name,
        mime="application/pdf",
    )
