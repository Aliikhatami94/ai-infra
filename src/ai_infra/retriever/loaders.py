"""File loaders for the Retriever module.

Loads various file formats and extracts text content with metadata.
Uses LangChain loaders internally where available.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

# Type alias for loaded documents: (text, metadata)
LoadedDocument = tuple[str, dict[str, Any]]


def load_file(path: str) -> list[LoadedDocument]:
    """Load a file and return its text content with metadata.

    Automatically detects the file type and uses the appropriate loader.

    Args:
        path: Path to the file to load.

    Returns:
        List of (text, metadata) tuples. Some formats (like PDF) may
        return multiple documents (one per page).

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file type is not supported.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = Path(path).suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".txt": load_text,
        ".md": load_text,
        ".markdown": load_text,
        ".docx": load_docx,
        ".doc": load_docx,
        ".csv": load_csv,
        ".json": load_json,
        ".html": load_html,
        ".htm": load_html,
    }

    loader = loaders.get(ext)
    if loader is None:
        supported = ", ".join(sorted(loaders.keys()))
        raise ValueError(f"Unsupported file type: {ext}\nSupported types: {supported}")

    return loader(path)


def load_directory(
    path: str,
    pattern: str = "*",
    recursive: bool = True,
) -> list[LoadedDocument]:
    """Load all files from a directory.

    Args:
        path: Path to the directory.
        pattern: Glob pattern for file matching (e.g., "*.pdf", "*.md").
        recursive: Whether to search subdirectories.

    Returns:
        List of (text, metadata) tuples from all loaded files.

    Raises:
        FileNotFoundError: If the directory doesn't exist.
    """
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory not found: {path}")

    dir_path = Path(path)
    all_documents: list[LoadedDocument] = []

    # Get matching files
    if recursive:
        files = list(dir_path.rglob(pattern))
    else:
        files = list(dir_path.glob(pattern))

    # Filter to only files (not directories)
    files = [f for f in files if f.is_file()]

    # Load each file
    for file_path in sorted(files):
        try:
            docs = load_file(str(file_path))
            all_documents.extend(docs)
        except ValueError:
            # Skip unsupported file types
            continue

    return all_documents


# ==============================================================================
# Individual file type loaders
# ==============================================================================


def load_pdf(path: str) -> list[LoadedDocument]:
    """Load a PDF file, returning one document per page.

    Args:
        path: Path to the PDF file.

    Returns:
        List of (text, metadata) tuples, one per page.
    """
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError(
            "pypdf is required for PDF loading. Install it with: pip install pypdf"
        ) from e

    documents: list[LoadedDocument] = []
    reader = PdfReader(path)

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():  # Skip empty pages
            metadata = {
                "source": path,
                "page": i + 1,  # 1-indexed
                "total_pages": len(reader.pages),
            }
            documents.append((text, metadata))

    return documents


def load_text(path: str) -> list[LoadedDocument]:
    """Load a plain text or markdown file.

    Args:
        path: Path to the text file.

    Returns:
        List containing a single (text, metadata) tuple.
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    metadata = {
        "source": path,
        "file_type": Path(path).suffix.lower(),
    }
    return [(text, metadata)]


def load_docx(path: str) -> list[LoadedDocument]:
    """Load a DOCX file.

    Args:
        path: Path to the DOCX file.

    Returns:
        List containing a single (text, metadata) tuple.
    """
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX loading. Install it with: pip install python-docx"
        ) from e

    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n\n".join(paragraphs)

    metadata = {
        "source": path,
        "file_type": ".docx",
    }
    return [(text, metadata)]


def load_csv(path: str) -> list[LoadedDocument]:
    """Load a CSV file, converting rows to text.

    Each row is converted to a text representation with column names.

    Args:
        path: Path to the CSV file.

    Returns:
        List of (text, metadata) tuples, one per row.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is required for CSV loading. Install it with: pip install pandas"
        ) from e

    df = pd.read_csv(path)
    documents: list[LoadedDocument] = []

    for i, row in df.iterrows():
        # Convert row to readable text format
        parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        text = "\n".join(parts)

        if text.strip():
            metadata = {
                "source": path,
                "row_index": int(i) if isinstance(i, int) else str(i),
                "file_type": ".csv",
            }
            documents.append((text, metadata))

    return documents


def load_json(path: str) -> list[LoadedDocument]:
    """Load a JSON file.

    Handles both JSON objects and JSON arrays.

    Args:
        path: Path to the JSON file.

    Returns:
        List of (text, metadata) tuples.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    documents: list[LoadedDocument] = []

    if isinstance(data, list):
        # JSON array - one document per item
        for i, item in enumerate(data):
            text = json.dumps(item, indent=2, ensure_ascii=False)
            metadata = {
                "source": path,
                "item_index": i,
                "file_type": ".json",
            }
            documents.append((text, metadata))
    else:
        # Single JSON object
        text = json.dumps(data, indent=2, ensure_ascii=False)
        metadata = {
            "source": path,
            "file_type": ".json",
        }
        documents.append((text, metadata))

    return documents


def load_html(path: str) -> list[LoadedDocument]:
    """Load an HTML file, extracting text content.

    Args:
        path: Path to the HTML file.

    Returns:
        List containing a single (text, metadata) tuple.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ImportError(
            "beautifulsoup4 is required for HTML loading. "
            "Install it with: pip install beautifulsoup4"
        ) from e

    with open(path, encoding="utf-8") as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()

    # Get text
    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    # Try to extract title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    metadata = {
        "source": path,
        "file_type": ".html",
    }
    if title:
        metadata["title"] = title

    return [(text, metadata)]
