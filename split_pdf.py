#!/usr/bin/env python3
"""Command-line tool to split double-spread (book scan) PDF pages.

Examples:
    # Auto-detect the gutter on every page
    python split_pdf.py scan.pdf pages.pdf

    # Treat the first page as a single-page cover (don't split it)
    python split_pdf.py scan.pdf pages.pdf --cover

    # Force a fixed centered split instead of auto-detection
    python split_pdf.py scan.pdf pages.pdf --ratio 0.5

    # Widen the gutter search to the middle 60% of the page
    python split_pdf.py scan.pdf pages.pdf --search-range 0.3

    # Pass specific pages through unchanged (1-based on the CLI)
    python split_pdf.py scan.pdf pages.pdf --skip 1 12
"""

from __future__ import annotations

import argparse
import sys

import splitter


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split double-spread book-scan PDF pages into single pages.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="Path to the source PDF.")
    p.add_argument("output", help="Path to write the split PDF.")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--ratio",
        type=float,
        default=None,
        metavar="R",
        help="Fixed split position as a fraction of width (0-1). "
        "Disables auto-detection.",
    )
    mode.add_argument(
        "--search-range",
        type=float,
        default=0.2,
        metavar="F",
        help="Half-width of the central gutter search band, as a fraction of "
        "page width (auto-detection only). 0.2 = search the middle 40%%.",
    )

    p.add_argument(
        "--cover",
        action="store_true",
        help="Pass the first page through unchanged (single-page cover).",
    )
    p.add_argument(
        "--skip",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="1-based page numbers to pass through without splitting.",
    )
    p.add_argument(
        "--detect-dpi",
        type=int,
        default=80,
        help="Render resolution used for gutter auto-detection.",
    )
    p.add_argument(
        "--output-dpi",
        type=int,
        default=200,
        help="Render resolution of the output page images.",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.ratio is not None and not (0.0 < args.ratio < 1.0):
        print("error: --ratio must be between 0 and 1", file=sys.stderr)
        return 2

    # Collect zero-based skip indices from --cover and --skip.
    skip: set[int] = set()
    if args.cover:
        skip.add(0)
    if args.skip:
        skip.update(n - 1 for n in args.skip)

    try:
        n = splitter.split_pdf_file(
            args.input,
            args.output,
            fixed_ratio=args.ratio,
            search_frac=args.search_range,
            skip_pages=skip,
            detect_dpi=args.detect_dpi,
            output_dpi=args.output_dpi,
        )
    except FileNotFoundError:
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface any PDF error to the user
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {n} pages to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
