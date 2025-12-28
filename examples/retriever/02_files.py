#!/usr/bin/env python
"""File Loading with Retriever Example.

This example demonstrates:
- Loading individual files with add_file()
- Supported file formats (PDF, DOCX, MD, TXT, HTML, etc.)
- Automatic chunking for large files
- Preserving file metadata and source
- Handling different encodings

The Retriever can ingest documents directly from files,
automatically detecting format and extracting text.
"""

import os
import tempfile
from pathlib import Path

from ai_infra import Retriever

# =============================================================================
# Example 1: Loading Text Files
# =============================================================================


def load_text_file():
    """Load a simple text file."""
    print("=" * 60)
    print("1. Loading Text Files")
    print("=" * 60)

    retriever = Retriever()

    # Create a sample text file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("""Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables
computers to learn from data without being explicitly programmed.

There are three main types of machine learning:
1. Supervised learning - learns from labeled data
2. Unsupervised learning - finds patterns in unlabeled data
3. Reinforcement learning - learns through trial and error

Machine learning is used in many applications including image
recognition, natural language processing, and recommendation systems.
""")
        temp_path = f.name

    try:
        print(f"\nLoading: {temp_path}")

        # Add the file to the retriever
        retriever.add_file(temp_path)

        print("\n✓ File loaded")
        print(f"  Documents: {len(retriever)}")

        # Search the content
        results = retriever.search("types of ML", k=2)
        print("\nSearch 'types of ML':")
        for r in results:
            print(f"  {r.score:.3f}: {r.document.text[:60]}...")
    finally:
        os.unlink(temp_path)


# =============================================================================
# Example 2: Loading Markdown Files
# =============================================================================


def load_markdown_file():
    """Load a Markdown file with formatting."""
    print("\n" + "=" * 60)
    print("2. Loading Markdown Files")
    print("=" * 60)

    retriever = Retriever()

    # Create a sample markdown file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write("""# Python Quick Reference

## Variables and Types

Python is dynamically typed. Common types include:
- `int` - integers like 42
- `float` - floating point like 3.14
- `str` - strings like "hello"
- `list` - ordered collections
- `dict` - key-value mappings

## Functions

Define functions with `def`:

```python
def greet(name):
    return f"Hello, {name}!"
```

## Classes

Create classes with `class`:

```python
class Person:
    def __init__(self, name):
        self.name = name
```
""")
        temp_path = f.name

    try:
        print(f"\nLoading Markdown: {Path(temp_path).name}")

        retriever.add_file(temp_path)

        print("\n✓ Markdown loaded")
        print(f"  Documents: {len(retriever)}")

        results = retriever.search("how to define functions", k=1)
        print("\nSearch 'how to define functions':")
        if results:
            print(f"  {results[0].document.text[:100]}...")
    finally:
        os.unlink(temp_path)


# =============================================================================
# Example 3: Loading with Metadata
# =============================================================================


def load_with_metadata():
    """Add metadata when loading files."""
    print("\n" + "=" * 60)
    print("3. Loading with Metadata")
    print("=" * 60)

    retriever = Retriever()

    # Create sample files
    files_data = [
        ("python_basics.txt", "Python is great for beginners and data science."),
        ("javascript_intro.txt", "JavaScript runs in web browsers and Node.js."),
    ]

    temp_paths = []
    for name, content in files_data:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix=name.replace(".txt", "_")
        ) as f:
            f.write(content)
            temp_paths.append(f.name)

    try:
        # Load files with metadata
        retriever.add_file(
            temp_paths[0],
            metadata={"language": "python", "category": "tutorial"},
        )
        retriever.add_file(
            temp_paths[1],
            metadata={"language": "javascript", "category": "intro"},
        )

        print(f"\n✓ Loaded {len(temp_paths)} files with metadata")

        # Search with metadata filter
        results = retriever.search(
            "programming language",
            k=2,
            filter={"language": "python"},
        )

        print("\nSearch with filter {language: 'python'}:")
        for r in results:
            print(f"  {r.score:.3f}: {r.document.text[:50]}...")
            print(f"           metadata: {r.document.metadata}")
    finally:
        for path in temp_paths:
            os.unlink(path)


# =============================================================================
# Example 4: Loading JSON Files
# =============================================================================


def load_json_file():
    """Load JSON data as documents."""
    print("\n" + "=" * 60)
    print("4. Loading JSON Files")
    print("=" * 60)

    import json

    retriever = Retriever()

    # Create sample JSON
    data = {
        "products": [
            {
                "name": "Widget Pro",
                "price": 99,
                "description": "Premium widget with extra features",
            },
            {
                "name": "Widget Basic",
                "price": 49,
                "description": "Entry-level widget for beginners",
            },
            {
                "name": "Widget Max",
                "price": 199,
                "description": "Maximum power widget for professionals",
            },
        ]
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2)
        temp_path = f.name

    try:
        print("\nLoading JSON file...")

        retriever.add_file(temp_path)

        print("\n✓ JSON loaded")
        print(f"  Documents: {len(retriever)}")

        results = retriever.search("professional widget", k=1)
        if results:
            print("\nSearch 'professional widget':")
            print(f"  {results[0].document.text[:80]}...")
    finally:
        os.unlink(temp_path)


# =============================================================================
# Example 5: Chunking Large Files
# =============================================================================


def chunking_large_files():
    """Demonstrate automatic chunking of large content."""
    print("\n" + "=" * 60)
    print("5. Chunking Large Files")
    print("=" * 60)

    retriever = Retriever()

    # Create a large file
    paragraphs = []
    for i in range(50):
        paragraphs.append(f"""
Section {i + 1}: Topic Discussion

This is paragraph {i + 1} of our large document. It contains
information about topic {i + 1}, which is very important for
understanding the overall subject matter. Each section builds
on the previous ones to create a comprehensive overview.
""")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(paragraphs))
        temp_path = f.name

    try:
        file_size = os.path.getsize(temp_path)
        print(f"\nFile size: {file_size:,} bytes")

        retriever.add_file(temp_path)

        print("\n✓ Large file loaded and chunked")
        print(f"  Total chunks: {len(retriever)}")

        results = retriever.search("Section 25", k=1)
        if results:
            print("\nSearch 'Section 25':")
            print(f"  {results[0].document.text[:100]}...")
    finally:
        os.unlink(temp_path)


# =============================================================================
# Example 6: File Source Tracking
# =============================================================================


def source_tracking():
    """Track file sources for attribution."""
    print("\n" + "=" * 60)
    print("6. File Source Tracking")
    print("=" * 60)

    retriever = Retriever()

    # Create files with different names
    with tempfile.TemporaryDirectory() as tmpdir:
        files = [
            ("api_docs.txt", "The API endpoint /users returns user data"),
            ("guide.txt", "Follow these steps to set up authentication"),
        ]

        for name, content in files:
            path = Path(tmpdir) / name
            path.write_text(content)
            retriever.add_file(str(path))

        print(f"\n✓ Loaded {len(files)} files")

        # Search and show sources
        results = retriever.search("API users", k=2)
        print("\nResults with source tracking:")
        for r in results:
            print(f"  Score: {r.score:.3f}")
            print(f"  Source: {r.document.source}")
            print(f"  Text: {r.document.text[:50]}...")
            print()


# =============================================================================
# Example 7: Supported Formats
# =============================================================================


def supported_formats():
    """List supported file formats."""
    print("\n" + "=" * 60)
    print("7. Supported File Formats")
    print("=" * 60)

    formats = {
        "Text": [".txt", ".text"],
        "Markdown": [".md", ".markdown"],
        "JSON": [".json"],
        "HTML": [".html", ".htm"],
        "Python": [".py"],
        "PDF": [".pdf"],
        "Word": [".docx", ".doc"],
        "CSV": [".csv"],
        "YAML": [".yaml", ".yml"],
    }

    print("\nSupported formats by category:")
    for category, extensions in formats.items():
        print(f"\n  {category}:")
        for ext in extensions:
            print(f"    • {ext}")

    print("\nNote: Some formats (PDF, DOCX) require additional dependencies.")
    print("Install with: pip install 'ai-infra[docs]'")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("File Loading Examples")
    print("=" * 60)
    print("\nThe Retriever can load various file formats directly.")
    print("It automatically extracts text and creates chunks.\n")

    load_text_file()
    load_markdown_file()
    load_with_metadata()
    load_json_file()
    chunking_large_files()
    source_tracking()
    supported_formats()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. add_file() loads and processes files automatically")
    print("  2. Multiple formats supported (txt, md, json, pdf, etc.)")
    print("  3. Large files are automatically chunked")
    print("  4. Source tracking for attribution")
    print("  5. Add metadata for filtering")


if __name__ == "__main__":
    main()
