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
from typing import Callable, Iterable, Optional, Sequence

import fitz  # PyMuPDF
import numpy as np


@dataclass
class SplitPlan:
    """How a single source page should be handled.

    All positions are fractions of the *full* rendered page (in its displayed
    orientation): x-fractions for ``ratio``/``left``/``right`` and y-fractions
    for ``top``/``bottom``.

    Attributes:
        split: When True the page is cut into a left and right page. When
            False the (cropped) page is emitted as a single page.
        ratio: Horizontal split position as a fraction of width in (0, 1).
        left, right: Crop bounds in x; content outside is dropped (removes the
            black scanner border / book-block edge on the sides).
        top, bottom: Crop bounds in y; content outside is dropped (removes the
            black scanner border at top/bottom).
    """

    split: bool = True
    ratio: float = 0.5
    left: float = 0.0
    right: float = 1.0
    top: float = 0.0
    bottom: float = 1.0


def _column_brightness(page: "fitz.Page", dpi: int = 100) -> np.ndarray:
    """Return the mean brightness (0-255) of every pixel column of ``page``."""
    pix = page.get_pixmap(colorspace=fitz.csGRAY, dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, : pix.width]
    return arr.mean(axis=0)


def detect_split_ratio(
    page: "fitz.Page",
    search_frac: float = 0.12,
    dpi: int = 80,
    smooth: int = 15,
) -> float:
    """Auto-detect the gutter split position for a spread page.

    On a flat book-spread scan the gutter is the meeting point of the two
    pages' inner margins, which together form the *brightest* vertical strip
    near the center of the image. We therefore smooth the per-column
    brightness profile (to ignore thin features like the binding shadow or a
    single dark text edge) and pick the brightest column within a narrow
    central band. Restricting to the center also avoids being fooled by the
    dark scanner border that often runs down one side of a scan.

    This is intentionally a *starting estimate*: real scans vary enough that
    the position should be confirmed/adjusted by hand (see the web app's
    draggable split line).

    Args:
        page: A PyMuPDF page.
        search_frac: Half-width of the central search band, as a fraction of
            page width. 0.12 searches the middle 24% of the page.
        dpi: Render resolution for the analysis. 80 is ample for a gutter.
        smooth: Width (in pixels) of the moving-average applied to the column
            profile before picking the peak. Larger ignores finer detail.

    Returns:
        Split position as a fraction of page width in (0, 1).
    """
    col = _column_brightness(page, dpi=dpi).astype(float)
    width = col.shape[0]
    if width < 8:
        return 0.5

    if smooth > 1:
        k = min(int(smooth), width)
        col = np.convolve(col, np.ones(k) / k, mode="same")

    center = width // 2
    half = max(1, int(search_frac * width))
    lo = max(1, center - half)
    hi = min(width - 1, center + half)
    if hi <= lo:
        return 0.5

    idx = lo + int(np.argmax(col[lo:hi]))
    return float(idx) / float(width)


def _lead(profile: np.ndarray, dark: float, bridge: int) -> int:
    """Pixels of background to trim from the start of a 1-D brightness profile.

    Advances while lines are dark (< ``dark``), tolerating up to ``bridge``
    bright lines (a thin bright scan-edge artifact) before a sustained bright
    run — the real page content — stops the trim.
    """
    last = -1
    gap = 0
    for i, v in enumerate(profile):
        if v < dark:
            last = i
            gap = 0
        else:
            gap += 1
            if gap > bridge:
                break
    return last + 1


def detect_content_box(
    page: "fitz.Page",
    dark: float = 130.0,
    bridge: int = 4,
    max_trim: float = 0.18,
    dpi: int = 80,
) -> tuple[float, float, float, float]:
    """Auto-detect a crop box that removes dark scanner borders.

    Scanned spreads often have a black scanner background (and a shadowed
    book-block edge) along the left and bottom. This trims dark lines inward
    from each edge, returning ``(left, right, top, bottom)`` as fractions of
    the page. Trimming on any edge is capped at ``max_trim`` so unusual dark
    content cannot collapse the box — fine-tune by hand if needed.
    """
    pix = page.get_pixmap(colorspace=fitz.csGRAY, dpi=dpi)
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    arr = arr.reshape(pix.height, pix.stride)[:, : pix.width].astype(float)
    h, w = arr.shape
    cm = arr.mean(axis=0)
    rm = arr.mean(axis=1)

    left = _lead(cm, dark, bridge) / w
    right = 1.0 - _lead(cm[::-1], dark, bridge) / w
    top = _lead(rm, dark, bridge) / h
    bottom = 1.0 - _lead(rm[::-1], dark, bridge) / h

    left = min(left, max_trim)
    right = max(right, 1.0 - max_trim)
    top = min(top, max_trim)
    bottom = max(bottom, 1.0 - max_trim)
    return left, right, top, bottom


def build_plan(
    doc: "fitz.Document",
    fixed_ratio: Optional[float] = None,
    search_frac: float = 0.12,
    skip_pages: Optional[Iterable[int]] = None,
    auto_crop: bool = True,
    dpi: int = 80,
) -> list[SplitPlan]:
    """Build a per-page :class:`SplitPlan` for the whole document.

    Args:
        doc: An open PyMuPDF document.
        fixed_ratio: If given, every split page uses this ratio instead of
            auto-detection. Useful when the scans are perfectly centered.
        search_frac: Passed through to :func:`detect_split_ratio`.
        skip_pages: Zero-based page indices to pass through without splitting
            (e.g. covers or single-page inserts). These are still cropped.
        auto_crop: When True, auto-detect a crop box per page to remove dark
            scanner borders (see :func:`detect_content_box`).
        dpi: Render resolution for auto-detection.

    Returns:
        A list of :class:`SplitPlan`, one per source page.
    """
    skip = set(skip_pages or ())
    plan: list[SplitPlan] = []
    for pno in range(doc.page_count):
        page = doc[pno]
        if auto_crop:
            left, right, top, bottom = detect_content_box(page, dpi=dpi)
        else:
            left, right, top, bottom = 0.0, 1.0, 0.0, 1.0

        if pno in skip:
            ratio, split = 0.5, False
        elif fixed_ratio is not None:
            ratio, split = float(fixed_ratio), True
        else:
            ratio = detect_split_ratio(page, search_frac=search_frac, dpi=dpi)
            split = True

        plan.append(
            SplitPlan(
                split=split, ratio=ratio,
                left=left, right=right, top=top, bottom=bottom,
            )
        )
    return plan


def _add_image_page(out: "fitz.Document", img: "Image.Image", dpi: int) -> None:
    """Append a new page sized to a PIL image (at ``dpi``) and draw it."""
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png = buf.getvalue()

    scale = 72.0 / float(dpi)
    page = out.new_page(width=img.width * scale, height=img.height * scale)
    page.insert_image(page.rect, stream=png)


def split_document(
    src: "fitz.Document",
    plan: Sequence[SplitPlan],
    dpi: int = 200,
    progress: Optional[Callable[[int, int], None]] = None,
) -> "fitz.Document":
    """Apply ``plan`` to ``src`` and return a new split document.

    Each spread page becomes two output pages (left then right). Pages whose
    plan has ``split=False`` are copied through unchanged. Pages are rendered
    in their *displayed* orientation (page rotation is respected) and split
    along the vertical line, so the output always matches what you see in the
    preview — regardless of any ``/Rotate`` on the source page. Output pages
    are images, which is appropriate for scanned material.

    Args:
        src: An open PyMuPDF document.
        plan: One :class:`SplitPlan` per source page.
        dpi: Render resolution for the output pages. Higher = sharper/larger.
        progress: Optional callback ``progress(done, total)`` invoked after
            each source page, for driving a progress bar.

    Returns:
        A fresh in-memory PyMuPDF document; the caller saves and closes it.
    """
    from PIL import Image

    if len(plan) != src.page_count:
        raise ValueError(
            f"plan has {len(plan)} entries but document has {src.page_count} pages"
        )

    out = fitz.open()
    for pno, p in enumerate(plan):
        pix = src[pno].get_pixmap(dpi=dpi)  # respects page rotation
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        w, h = img.width, img.height

        # Crop box (pixels), clamped and ordered.
        cl = min(max(int(round(p.left * w)), 0), w - 1)
        cr = min(max(int(round(p.right * w)), cl + 1), w)
        ct = min(max(int(round(p.top * h)), 0), h - 1)
        cb = min(max(int(round(p.bottom * h)), ct + 1), h)

        if not p.split:
            _add_image_page(out, img.crop((cl, ct, cr, cb)), dpi)
        else:
            split_x = min(max(int(round(p.ratio * w)), cl + 1), cr - 1)
            left = img.crop((cl, ct, split_x, cb))
            right = img.crop((split_x, ct, cr, cb))
            for half in (left, right):
                _add_image_page(out, half, dpi)

        if progress is not None:
            progress(pno + 1, len(plan))

    return out


def split_pdf_file(
    input_path: str,
    output_path: str,
    fixed_ratio: Optional[float] = None,
    search_frac: float = 0.12,
    skip_pages: Optional[Iterable[int]] = None,
    detect_dpi: int = 80,
    output_dpi: int = 200,
) -> int:
    """Convenience wrapper: open, plan, split and save in one call.

    Args:
        detect_dpi: Resolution used for gutter auto-detection (fast, low).
        output_dpi: Resolution of the rendered output pages (quality).

    Returns the number of pages written to ``output_path``.
    """
    src = fitz.open(input_path)
    try:
        plan = build_plan(
            src,
            fixed_ratio=fixed_ratio,
            search_frac=search_frac,
            skip_pages=skip_pages,
            dpi=detect_dpi,
        )
        out = split_document(src, plan, dpi=output_dpi)
        try:
            out.save(output_path, deflate=True, garbage=3)
            return out.page_count
        finally:
            out.close()
    finally:
        src.close()
