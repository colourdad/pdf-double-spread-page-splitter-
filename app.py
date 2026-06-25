"""Streamlit web app for splitting double-spread book-scan PDFs.

For each spread: drag the red line to the gutter and the green box to crop off
the black scanner border (both are auto-detected as a starting point). Mark
covers / single pages as "don't split". When happy, build and download the
resulting single-page PDF.

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
OUTPUT_DPI = 200


st.set_page_config(page_title="PDF Spread Splitter", layout="wide")
st.title("📖 PDF Double-Spread Page Splitter")
st.caption(
    "Drag the red line to the gutter and the green box to crop the scanner "
    "border, then build the split PDF."
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
def _auto(pdf_bytes: bytes, pno: int):
    """Auto-detected split ratio and crop box for one page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page = doc[pno]
        ratio = splitter.detect_split_ratio(page)
        left, right, top, bottom = splitter.detect_content_box(page)
        return {
            "split": ratio, "left": left, "right": right,
            "top": top, "bottom": bottom, "do_split": True,
        }
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

state = st.session_state.setdefault("pages", {})  # pno -> dict of positions

with st.sidebar:
    st.header("Navigate")
    cur = st.number_input(
        "Spread (page) to edit", min_value=1, max_value=page_count, value=1, step=1
    ) - 1
    st.write(f"Page **{cur + 1}** of **{page_count}**")
    st.divider()
    if st.button("Reset this page to auto"):
        state.pop(cur, None)
        st.rerun()
    st.caption(
        "Green box = crop (drag its edges to remove the black border). "
        "Red line = where the spread is split."
    )

# Initialise this page's state from auto-detection the first time we see it.
entry = state.setdefault(cur, dict(_auto(pdf_bytes, cur)))

img_url, aspect = _preview(pdf_bytes, cur)

col_main, col_side = st.columns([4, 1])

with col_main:
    res = gutter_picker(
        image_url=img_url,
        aspect=aspect,
        split=entry["split"],
        left=entry["left"],
        right=entry["right"],
        top=entry["top"],
        bottom=entry["bottom"],
        disabled=not entry["do_split"],
        key=f"gp_{cur}",
    )
    for k in ("split", "left", "right", "top", "bottom"):
        entry[k] = float(res[k])

with col_side:
    st.markdown(f"### Page {cur + 1}")
    entry["do_split"] = st.checkbox(
        "Split this page", value=entry["do_split"], key=f"split_{cur}"
    )
    st.metric("Split", f"{entry['split']:.3f}")
    st.caption(
        f"crop L {entry['left']:.2f} · R {entry['right']:.2f} · "
        f"T {entry['top']:.2f} · B {entry['bottom']:.2f}"
    )

st.divider()

if st.button("✂️ Split and build full PDF", type="primary"):
    plan = []
    for pno in range(page_count):
        e = state.get(pno) or _auto(pdf_bytes, pno)
        plan.append(
            splitter.SplitPlan(
                split=e["do_split"], ratio=e["split"],
                left=e["left"], right=e["right"], top=e["top"], bottom=e["bottom"],
            )
        )

    bar = st.progress(0.0, text="Starting…")

    def _on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Building page {done} of {total}…")

    src = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        with st.spinner("Rendering and splitting pages…"):
            out = splitter.split_document(
                src, plan, dpi=OUTPUT_DPI, progress=_on_progress
            )
        try:
            bar.progress(1.0, text="Packaging PDF…")
            buf = out.tobytes(deflate=True, garbage=3)
            n = out.page_count
        finally:
            out.close()
    finally:
        src.close()

    bar.empty()
    st.success(f"Done — produced {n} pages.")
    st.download_button(
        "⬇️ Download split PDF",
        data=buf,
        file_name=uploaded.name.rsplit(".", 1)[0] + "_split.pdf",
        mime="application/pdf",
    )
    st.caption("Pages you didn't open used their auto-detected split and crop.")
