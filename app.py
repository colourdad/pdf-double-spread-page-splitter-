"""Streamlit web app for splitting double-spread book-scan PDFs.

Upload a PDF, then for each spread drag the red split line to the gutter
(it follows your cursor live and the auto-detected position is the starting
point). Mark covers / single pages as "don't split". When happy, build and
download the resulting single-page PDF.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import base64

import fitz  # PyMuPDF
import streamlit as st

import splitter
from gutter_picker import gutter_picker

PREVIEW_DPI = 110


st.set_page_config(page_title="PDF Spread Splitter", layout="wide")
st.title("📖 PDF Double-Spread Page Splitter")
st.caption(
    "Drag the red line to the gutter on each spread, then build the split PDF."
)


@st.cache_data(show_spinner=False)
def _preview(pdf_bytes: bytes, pno: int, dpi: int = PREVIEW_DPI):
    """Render page ``pno`` to (data-url, aspect = height/width)."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        pix = doc[pno].get_pixmap(dpi=dpi)
        url = "data:image/png;base64," + base64.b64encode(pix.tobytes("png")).decode()
        return url, (pix.height / pix.width if pix.width else 1.0)
    finally:
        doc.close()


@st.cache_data(show_spinner=False)
def _seed(pdf_bytes: bytes, pno: int) -> float:
    """Auto-detected starting split ratio for one page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return splitter.detect_split_ratio(doc[pno])
    finally:
        doc.close()


uploaded = st.file_uploader("Upload a PDF of double-spread scans", type="pdf")
if uploaded is None:
    st.info("Upload a PDF to get started.")
    st.stop()

pdf_bytes = uploaded.getvalue()
doc = fitz.open(stream=pdf_bytes, filetype="pdf")
page_count = doc.page_count
doc.close()

# Per-page state persists as you navigate between spreads.
state = st.session_state.setdefault("pages", {})  # pno -> {"ratio": float, "split": bool}

with st.sidebar:
    st.header("Navigate")
    cur = st.number_input(
        "Spread (page) to edit",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
    ) - 1
    st.write(f"Page **{cur + 1}** of **{page_count}**")
    st.divider()
    if st.button("Reset this page to auto"):
        state.pop(cur, None)
        st.rerun()

# Initialise this page's state from auto-detection the first time we see it.
entry = state.setdefault(
    cur, {"ratio": _seed(pdf_bytes, cur), "split": True}
)

img_url, aspect = _preview(pdf_bytes, cur)

col_main, col_side = st.columns([4, 1])
with col_side:
    st.markdown(f"### Page {cur + 1}")
    split = st.checkbox("Split this page", value=entry["split"], key=f"split_{cur}")
    entry["split"] = split
    st.caption(
        "Drag the red line (or click) to set the gutter. "
        "Releasing saves the position."
    )
    st.metric("Split position", f"{entry['ratio']:.3f}")

with col_main:
    new_ratio = gutter_picker(
        image_url=img_url,
        ratio=entry["ratio"],
        aspect=aspect,
        disabled=not split,
        key=f"gp_{cur}",
    )
    if new_ratio is not None:
        entry["ratio"] = float(new_ratio)

st.divider()

# ---- Build the full split PDF across all pages --------------------------
if st.button("✂️ Split and build full PDF", type="primary"):
    plan = []
    for pno in range(page_count):
        e = state.get(pno)
        if e is None:  # never visited -> use auto-detected seed
            e = {"ratio": _seed(pdf_bytes, pno), "split": True}
        plan.append(splitter.SplitPlan(split=e["split"], ratio=e["ratio"]))

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        out = splitter.split_document(src, plan)
        try:
            buf = out.tobytes(deflate=True, garbage=3)
            n = out.page_count
        finally:
            out.close()
    finally:
        src.close()

    st.success(f"Done — produced {n} pages.")
    st.download_button(
        "⬇️ Download split PDF",
        data=buf,
        file_name=uploaded.name.rsplit(".", 1)[0] + "_split.pdf",
        mime="application/pdf",
    )
    st.caption(
        "Pages you didn't open were split at their auto-detected position."
    )
