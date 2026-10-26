#!/usr/bin/env python3
"""Export the Rakuten report notebook as A4 PDFs or printable HTML."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter
from nbconvert.preprocessors import ExecutePreprocessor
from traitlets.config import Config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_CANDIDATES = (
    PROJECT_ROOT / "notebooks" / "01_dataviz_and_audit.ipynb",
)
DEFAULT_NOTEBOOK = next(
    (path for path in NOTEBOOK_CANDIDATES if path.exists()),
    NOTEBOOK_CANDIDATES[0],
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports" / "01_dataViz_and_audit"

PRINT_CSS = """
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        color: #6b7280;
        font-family: Arial, sans-serif;
        font-size: 8pt;
    }
}

*, *:before, *:after {
    box-sizing: border-box;
}

html, body {
    color: #1f2937;
    font-family: Arial, Helvetica, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    margin: 0;
    padding: 0;
    width: 100%;
}

body, .jp-Notebook, .jp-NotebookPanel-notebook, .jp-Cell, .jp-Cell-outputWrapper, .jp-OutputArea, .jp-OutputArea-child, .jp-OutputArea-output {
    box-sizing: border-box;
    margin: 0;
    max-width: 100%;
    padding: 0;
    width: 100%;
}

h1, h2, h3, h4 {
    color: #111827;
    font-weight: 600;
    break-after: avoid;
    page-break-after: avoid;
}

h1 {
    border-bottom: 2px solid #991b1b;
    font-size: 20pt;
    margin: 0 0 8mm 0;
    padding-bottom: 2mm;
}

.section-cell {
    break-before: page;
    page-break-before: always;
}

h2 {
    font-size: 15pt;
    margin: 7mm 0 3mm 0;
}

h3 {
    font-size: 12pt;
    margin: 5mm 0 2mm 0;
}

p, li {
    orphans: 3;
    widows: 3;
}

table {
    border-collapse: collapse;
    table-layout: auto;
    break-inside: auto;
    font-size: 8pt;
    margin: 4mm 0;
    max-width: 100%;
    width: 100%;
}

tr {
    break-inside: avoid;
    page-break-inside: avoid;
}

th, td {
    border: 0.5pt solid #d1d5db;
    padding: 1.5mm 1.8mm;
    text-align: left;
    vertical-align: top;
    overflow-wrap: break-word;
    word-break: break-word;
    white-space: normal;
}

th {
    background: #f3f4f6;
    font-weight: 600;
}

img, svg {
    break-inside: avoid;
    display: block;
    height: auto;
    margin: 3mm auto;
    max-height: 220mm;
    max-width: 100% !important;
    object-fit: contain;
}

figure, .jp-RenderedImage, .jp-OutputArea-output, .output_png {
    break-inside: avoid;
    max-width: 100%;
}

.jp-Cell-inputWrapper, .jp-InputArea, .input_area {
    break-inside: avoid;
    max-width: 100%;
}

pre, code {
    font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
}

pre {
    background: #f8fafc;
    border: 0.5pt solid #d1d5db;
    border-radius: 3px;
    font-size: 7.2pt;
    line-height: 1.35;
    overflow-wrap: anywhere;
    padding: 2.5mm;
    white-space: pre-wrap;
}

.jp-InputPrompt, .jp-OutputPrompt, .prompt, .anchor-link {
    display: none !important;
}

.jp-OutputArea, .output_area {
    margin: 2mm 0 4mm 0;
}

.jp-RenderedHTMLCommon {
    overflow: visible;
}

blockquote {
    border-left: 3px solid #9ca3af;
    color: #4b5563;
    margin-left: 0;
    padding-left: 4mm;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create A4 PDF or HTML versions of Rakuten notebooks."
    )
    parser.add_argument(
        "notebooks",
        nargs="*",
        type=Path,
        help="One or more notebook paths to export.",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=None,
        help="Single notebook path to export (alternative to positional argument).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Custom output directory. Defaults to reports/<notebook_name>.",
    )
    parser.add_argument(
        "--format",
        choices=("pdf", "html"),
        default="pdf",
        help="Use html to export without WeasyPrint and print to PDF in a browser.",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "with-code", "without-code"),
        default="both",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the notebook in memory before exporting it.",
    )
    return parser.parse_args()


def load_notebook(path: Path, execute: bool) -> nbformat.NotebookNode:
    notebook = nbformat.read(path, as_version=4)
    if not execute:
        return notebook

    executor = ExecutePreprocessor(timeout=600, kernel_name="python3")
    executor.preprocess(notebook, {"metadata": {"path": str(PROJECT_ROOT)}})
    return notebook


def notebook_html(notebook: nbformat.NotebookNode, include_code: bool) -> str:
    config = Config()
    config.HTMLExporter.exclude_input = not include_code
    config.HTMLExporter.exclude_input_prompt = True
    config.HTMLExporter.exclude_output_prompt = True
    config.HTMLExporter.embed_images = True

    exporter = HTMLExporter(config=config, template_name="lab")
    html, _ = exporter.from_notebook_node(copy.deepcopy(notebook))

    document = BeautifulSoup(html, "html.parser")
    top_level_headings = document.find_all("h1")
    for heading in top_level_headings[1:]:
        cell = heading.find_parent(
            lambda tag: "jp-Cell" in tag.get("class", [])
        )
        if cell is not None:
            cell["class"] = [*cell.get("class", []), "section-cell"]

    return str(document)


def write_pdf(notebook: nbformat.NotebookNode, destination: Path, include_code: bool) -> None:
    try:
        from weasyprint import CSS, HTML
    except (ImportError, OSError) as error:
        raise SystemExit(
            "WeasyPrint or one of its system libraries is unavailable.\n"
            "On macOS, install the libraries with: brew install pango\n"
            "Alternatively, rerun this command with --format html, then open\n"
            "the exported HTML in a browser and print it to PDF.\n"
            f"Details: {error}"
        ) from error

    html = notebook_html(notebook, include_code)
    document = HTML(string=html, base_url=str(PROJECT_ROOT))
    document.write_pdf(destination, stylesheets=[CSS(string=PRINT_CSS)])
    print(f"Created {destination}")


def write_html(notebook: nbformat.NotebookNode, destination: Path, include_code: bool) -> None:
    document = BeautifulSoup(notebook_html(notebook, include_code), "html.parser")
    base = document.new_tag("base", href=PROJECT_ROOT.as_uri() + "/")
    document.head.insert(0, base)
    style = document.new_tag("style", media="print")
    style.string = PRINT_CSS
    document.head.append(style)
    destination.write_text(str(document), encoding="utf-8")
    print(f"Created {destination}")


def get_default_output_dir(notebook_path: Path) -> Path:
    stem = notebook_path.stem
    if stem == "01_dataviz_and_audit":
        return PROJECT_ROOT / "reports" / "01_dataViz_and_audit"
    return PROJECT_ROOT / "reports" / stem


def export_single_notebook(
    notebook_path: Path,
    output_dir: Path | None,
    fmt: str,
    mode: str,
    execute: bool,
) -> None:
    if not notebook_path.exists():
        raise SystemExit(f"Notebook not found: {notebook_path}")

    target_dir = output_dir or get_default_output_dir(notebook_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    notebook = load_notebook(notebook_path, execute)
    writer = write_html if fmt == "html" else write_pdf
    stem = notebook_path.stem

    if mode in {"both", "with-code"}:
        writer(
            notebook,
            target_dir / f"{stem}_with_code.{fmt}",
            include_code=True,
        )

    if mode in {"both", "without-code"}:
        writer(
            notebook,
            target_dir / f"{stem}_without_code.{fmt}",
            include_code=False,
        )


def main() -> None:
    args = parse_args()
    notebook_paths = list(args.notebooks)
    if args.notebook:
        notebook_paths.append(args.notebook)

    if not notebook_paths:
        notebook_paths = [DEFAULT_NOTEBOOK]

    for nb_path in notebook_paths:
        export_single_notebook(
            notebook_path=nb_path.resolve(),
            output_dir=args.output_dir.resolve() if args.output_dir else None,
            fmt=args.format,
            mode=args.mode,
            execute=args.execute,
        )


if __name__ == "__main__":
    main()
