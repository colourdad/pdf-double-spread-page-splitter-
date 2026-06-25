"""A bidirectional Streamlit component: a draggable gutter / split line.

Renders a page image with a vertical red line that the user can click or
drag. The line follows the cursor live (handled entirely in the browser);
the chosen position is sent back to Python when the drag ends. The returned
value is the split position as a fraction of image width in (0, 1).

The frontend is plain HTML/JS (``frontend/index.html``) so there is no npm
build step — it implements the Streamlit component message protocol directly.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component = components.declare_component("gutter_picker", path=_FRONTEND_DIR)


def gutter_picker(
    image_url: str,
    ratio: float,
    aspect: float,
    disabled: bool = False,
    key: Optional[str] = None,
) -> Optional[float]:
    """Show a draggable split line over ``image_url``.

    Args:
        image_url: A data URL (e.g. ``data:image/png;base64,...``) or path for
            the page preview image.
        ratio: Initial line position as a fraction of width in (0, 1).
        aspect: Image height / width, used to size the component.
        disabled: When True the line is shown but not draggable.
        key: Streamlit widget key (use a per-page unique key).

    Returns:
        The split ratio after the most recent drag, or ``ratio`` until the
        user interacts.
    """
    value = _component(
        image_url=image_url,
        ratio=float(ratio),
        aspect=float(aspect),
        disabled=bool(disabled),
        key=key,
        default=float(ratio),
    )
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(ratio)
