"""Core library for splitting double-spread (book scan) PDF pages.

A "double spread" is a single scanned page that actually contains two book
pages side by side, joined at the gutter. This module:

  * auto-detects the gutter / split point on a page using a horizontal
    intensity-gradient analysis (the gutter is usually a darker vertical
    band near the center of the scan), and
  * executes a per-page ``SplitPlan`` to produce an output PDF where each
    spread becomes two separate pages, while non-spread pages (e.g. the
    cover) are passed through untouched.

The split is done with ``Page.show_pdf_page`` clipping so vector content and
text are preserved at full quality rather than rasterized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import fitz  # PyMuPDF
import numpy as np


@dataclass
class SplitPlan:
    """How a single source page should be handled.

    Attributes:
        split: When True the page is cut into a left and right page. When
            False the page is passed through to the output unchanged.
        ratio: Horizontal split position as a fraction of page width in the
            range (0, 1). 0.5 is the exact middle. Only used when ``split``.
    """

    split: bool = True
    ratio: float = 0.5


def _column_brightness(page: "fitz.Page", dpi: int = 100) -> np.ndarray:
    """Return the mean brightness (0-255) of every pixel column of ``page``."""
    pix = page.get_pixmap(colorspace=fitz.csGRAY, dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, : pix.width]
    return arr.mean(axis=0)


def detect_split_ratio(
    page: "fitz.Page",
    search_frac: float = 0.2,
    dpi: int = 100,
) -> float:
    """Auto-detect the gutter split position for a spread page.

    The gutter of a scanned book spread shows up as a vertical band that is
    either darker (binding shadow) or has the steepest brightness change.
    We restrict the search to a central band of width ``2 * search_frac`` so
    that dark content elsewhere on the page does not fool the detector, then
    pick the column with the strongest horizontal gradient, falling back to
    the darkest column.

    Args:
        page: A PyMuPDF page.
        search_frac: Half-width of the central search band, as a fraction of
            page width. 0.2 searches the middle 40% of the page.
        dpi: Render resolution for the analysis. Higher is more precise but
            slower; 100 is plenty for locating a gutter.

    Returns:
        Split position as a fraction of page width in (0, 1).
    """
    col = _column_brightness(page, dpi=dpi)
    width = col.shape[0]
    if width < 4:
        return 0.5

    center = width // 2
    half = max(1, int(search_frac * width))
    lo = max(1, center - half)
    hi = min(width - 1, center + half)
    if hi <= lo:
        return 0.5

    # Strongest horizontal gradient within the central band. A gutter is an
    # edge, so |d(brightness)/dx| spikes there.
    gradient = np.abs(np.diff(col))
    band_grad = gradient[lo:hi]
    band_bright = col[lo:hi]

    # Combine: prefer a strong edge that is also relatively dark.
    grad_score = band_grad / (band_grad.max() + 1e-6)
    dark_score = 1.0 - (band_bright / 255.0)
    score = grad_score + dark_score
    idx = lo + int(np.argmax(score))

    return float(idx) / float(width)


def build_plan(
    doc: "fitz.Document",
    fixed_ratio: Optional[float] = None,
    search_frac: float = 0.2,
    skip_pages: Optional[Iterable[int]] = None,
    dpi: int = 100,
) -> list[SplitPlan]:
    """Build a per-page :class:`SplitPlan` for the whole document.

    Args:
        doc: An open PyMuPDF document.
        fixed_ratio: If given, every split page uses this ratio instead of
            auto-detection. Useful when the scans are perfectly centered.
        search_frac: Passed through to :func:`detect_split_ratio`.
        skip_pages: Zero-based page indices to pass through without splitting
            (e.g. covers or single-page inserts).
        dpi: Render resolution for auto-detection.

    Returns:
        A list of :class:`SplitPlan`, one per source page.
    """
    skip = set(skip_pages or ())
    plan: list[SplitPlan] = []
    for pno in range(doc.page_count):
        if pno in skip:
            plan.append(SplitPlan(split=False))
        elif fixed_ratio is not None:
            plan.append(SplitPlan(split=True, ratio=float(fixed_ratio)))
        else:
            ratio = detect_split_ratio(doc[pno], search_frac=search_frac, dpi=dpi)
            plan.append(SplitPlan(split=True, ratio=ratio))
    return plan


def split_document(
    src: "fitz.Document",
    plan: Sequence[SplitPlan],
) -> "fitz.Document":
    """Apply ``plan`` to ``src`` and return a new split document.

    Each spread page becomes two output pages (left then right). Pages whose
    plan has ``split=False`` are copied through unchanged. The returned
    document is a fresh in-memory PyMuPDF document; the caller is responsible
    for saving and closing it.
    """
    if len(plan) != src.page_count:
        raise ValueError(
            f"plan has {len(plan)} entries but document has {src.page_count} pages"
        )

    out = fitz.open()
    for pno, page_plan in enumerate(plan):
        page = src[pno]
        rect = page.rect

        if not page_plan.split:
            new = out.new_page(width=rect.width, height=rect.height)
            new.show_pdf_page(new.rect, src, pno)
            continue

        ratio = min(max(page_plan.ratio, 0.01), 0.99)
        split_x = rect.x0 + ratio * rect.width
        left = fitz.Rect(rect.x0, rect.y0, split_x, rect.y1)
        right = fitz.Rect(split_x, rect.y0, rect.x1, rect.y1)

        for clip in (left, right):
            new = out.new_page(width=clip.width, height=clip.height)
            new.show_pdf_page(new.rect, src, pno, clip=clip)

    return out


def split_pdf_file(
    input_path: str,
    output_path: str,
    fixed_ratio: Optional[float] = None,
    search_frac: float = 0.2,
    skip_pages: Optional[Iterable[int]] = None,
    dpi: int = 100,
) -> int:
    """Convenience wrapper: open, plan, split and save in one call.

    Returns the number of pages written to ``output_path``.
    """
    src = fitz.open(input_path)
    try:
        plan = build_plan(
            src,
            fixed_ratio=fixed_ratio,
            search_frac=search_frac,
            skip_pages=skip_pages,
            dpi=dpi,
        )
        out = split_document(src, plan)
        try:
            out.save(output_path, deflate=True, garbage=3)
            return out.page_count
        finally:
            out.close()
    finally:
        src.close()
