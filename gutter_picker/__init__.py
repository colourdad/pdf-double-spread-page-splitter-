"""A bidirectional Streamlit component: draggable split line + crop box.

Renders a page image with a red vertical split line and a green crop
rectangle (left/right/top/bottom edges). All five controls can be clicked or
dragged; they follow the cursor live in the browser and the positions are
sent back to Python when a drag ends.

The frontend is plain HTML/JS (``frontend/index.html``) so there is no npm
build step — it implements the Streamlit component message protocol directly.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import streamlit.components.v1 as components

_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component = components.declare_component("gutter_picker", path=_FRONTEND_DIR)


def gutter_picker(
    image_url: str,
    aspect: float,
    split: float,
    left: float,
    right: float,
    top: float,
    bottom: float,
    disabled: bool = False,
    key: Optional[str] = None,
) -> Dict[str, float]:
    """Show a draggable split line + crop box over ``image_url``.

    All positions are fractions of the image (x for split/left/right, y for
    top/bottom). Returns a dict with keys ``split, left, right, top, bottom``
    reflecting the latest drag (or the passed-in values until the user
    interacts).
    """
    defaults = {
        "split": float(split), "left": float(left), "right": float(right),
        "top": float(top), "bottom": float(bottom),
    }
    value = _component(
        image_url=image_url,
        aspect=float(aspect),
        disabled=bool(disabled),
        key=key,
        default=defaults,
        **defaults,
    )
    if isinstance(value, dict):
        out = dict(defaults)
        for k in out:
            try:
                out[k] = float(value[k])
            except (KeyError, TypeError, ValueError):
                pass
        return out
    return defaults
