#!/usr/bin/env python
"""Directory Loading with Retriever Example.

This example demonstrates:
- Loading entire directories with add_directory()
- Using glob patterns to filter files
- Recursive vs non-recursive directory scanning
- Excluding files and directories
- Batch loading with progress tracking

Perfect for ingesting documentation sites, code repositories,
or any collection of files.
"""

import tempfile
from pathlib import Path

from ai_infra import Retriever

# =============================================================================
# Example 1: Basic Directory Loading
# =============================================================================


def basic_directory():
    """Load all files from a directory."""
    print("=" * 60)
    print("1. Basic Directory Loading")
    print("=" * 60)

    retriever = Retriever()

    # Create a sample directory with files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample files
        files = {
            "intro.txt": "Welcome to our documentation.",
            "setup.txt": "Installation requires Python 3.11+",
            "usage.txt": "Import the library with 'import ai_infra'",
        }

        for name, content in files.items():
            (Path(tmpdir) / name).write_text(content)

        print(f"\nDirectory: {tmpdir}")
        print(f"Files: {list(files.keys())}")

        # Load entire directory
        retriever.add_directory(tmpdir)

        print(f"\n[OK] Loaded {len(retriever)} documents")

        results = retriever.search("how to install", k=1)
        if results:
            print("\nSearch 'how to install':")
            print(f"  {results[0].document.text}")


# =============================================================================
# Example 2: Glob Pattern Filtering
# =============================================================================


def glob_patterns():
    """Use glob patterns to filter files."""
    print("\n" + "=" * 60)
    print("2. Glob Pattern Filtering")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mixed file types
        files = {
            "readme.md": "# Project README\nMain documentation.",
            "guide.md": "# User Guide\nHow to use the product.",
            "config.json": '{"key": "value"}',
            "script.py": "print('Hello World')",
            "notes.txt": "Some plain text notes.",
        }

        for name, content in files.items():
            (Path(tmpdir) / name).write_text(content)

        print(f"\nAll files: {list(files.keys())}")

        # Load only markdown files
        retriever = Retriever()
        retriever.add_directory(tmpdir, pattern="*.md")

        print(f"\n[OK] Pattern '*.md' loaded {len(retriever)} documents")

        results = retriever.search("documentation", k=2)
        print("\nResults from markdown files only:")
        for r in results:
            source = Path(r.document.source).name if r.document.source else "unknown"
            print(f"  {source}: {r.document.text[:40]}...")


# =============================================================================
# Example 3: Recursive Directory Loading
# =============================================================================


def recursive_loading():
    """Load files from nested directories."""
    print("\n" + "=" * 60)
    print("3. Recursive Directory Loading")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested structure
        structure = {
            "docs/getting-started.md": "# Getting Started\nBegin here.",
            "docs/api/endpoints.md": "# API Endpoints\n/users, /items",
            "docs/api/auth.md": "# Authentication\nUse API keys.",
            "src/main.py": "def main(): pass",
            "tests/test_main.py": "def test_main(): pass",
        }

        for path, content in structure.items():
            file_path = Path(tmpdir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        print("\nDirectory structure:")
        for path in structure:
            print(f"  {path}")

        # Load recursively
        retriever = Retriever()
        retriever.add_directory(tmpdir, pattern="**/*.md", recursive=True)

        print(f"\n[OK] Recursively loaded {len(retriever)} markdown files")

        results = retriever.search("API authentication", k=2)
        print("\nSearch results:")
        for r in results:
            source = r.document.source
            if source:
                rel_source = str(Path(source).relative_to(tmpdir))
                print(f"  {rel_source}: {r.document.text[:30]}...")


# =============================================================================
# Example 4: Multiple Patterns
# =============================================================================


def multiple_patterns():
    """Load files matching multiple patterns."""
    print("\n" + "=" * 60)
    print("4. Multiple Patterns")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        files = {
            "readme.md": "Main README",
            "docs.txt": "Text documentation",
            "config.json": '{"config": true}',
            "style.css": "body { margin: 0; }",
        }

        for name, content in files.items():
            (Path(tmpdir) / name).write_text(content)

        # Load multiple patterns by calling add_directory multiple times
        retriever = Retriever()

        patterns = ["*.md", "*.txt"]
        for pattern in patterns:
            retriever.add_directory(tmpdir, pattern=pattern)

        print(f"\nPatterns: {patterns}")
        print(f"[OK] Loaded {len(retriever)} documents")

        for i, result in enumerate(retriever.search("", k=10)):
            source = Path(result.document.source).name if result.document.source else "?"
            print(f"  {i + 1}. {source}")


# =============================================================================
# Example 5: Exclude Patterns
# =============================================================================


def exclude_patterns():
    """Exclude certain files or directories."""
    print("\n" + "=" * 60)
    print("5. Exclude Patterns")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create structure with files to exclude
        structure = {
            "src/main.py": "Main application code",
            "src/utils.py": "Utility functions",
            "tests/test_main.py": "Test code - should skip",
            "docs/readme.md": "Documentation",
            ".git/config": "Git config - should skip",
            "node_modules/pkg/index.js": "NPM package - should skip",
        }

        for path, content in structure.items():
            file_path = Path(tmpdir) / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        retriever = Retriever()

        # Load only from src/ and docs/
        for subdir in ["src", "docs"]:
            dir_path = Path(tmpdir) / subdir
            if dir_path.exists():
                retriever.add_directory(str(dir_path), pattern="*.*")

        print(f"\n[OK] Loaded {len(retriever)} documents (excluding tests, .git, node_modules)")

        print("\nLoaded files:")
        for result in retriever.search("", k=10):
            if result.document.source:
                rel = Path(result.document.source).relative_to(tmpdir)
                print(f"  {rel}")


# =============================================================================
# Example 6: Loading Code Files
# =============================================================================


def load_code_files():
    """Load source code files for code search."""
    print("\n" + "=" * 60)
    print("6. Loading Code Files")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample Python files
        code_files = {
            "utils.py": '''"""Utility functions for data processing."""

def clean_text(text: str) -> str:
    """Remove whitespace and normalize text."""
    return text.strip().lower()

def validate_email(email: str) -> bool:
    """Check if email format is valid."""
    return "@" in email and "." in email
''',
            "api.py": '''"""REST API endpoints."""

from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/users")
def get_users():
    """Return list of all users."""
    return jsonify({"users": []})

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})
''',
        }

        for name, content in code_files.items():
            (Path(tmpdir) / name).write_text(content)

        retriever = Retriever()
        retriever.add_directory(tmpdir, pattern="*.py")

        print(f"\n[OK] Loaded {len(retriever)} Python file chunks")

        queries = [
            "email validation function",
            "flask API endpoint",
            "text processing utility",
        ]

        for query in queries:
            results = retriever.search(query, k=1)
            if results:
                source = Path(results[0].document.source).name
                print(f"\n  '{query}'")
                print(f"  → {source}: {results[0].document.text[:50]}...")


# =============================================================================
# Example 7: Directory with Metadata
# =============================================================================


def directory_with_metadata():
    """Add metadata to all files in a directory."""
    print("\n" + "=" * 60)
    print("7. Directory with Metadata")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create versioned documentation
        for version in ["v1", "v2"]:
            version_dir = Path(tmpdir) / version
            version_dir.mkdir()
            (version_dir / "api.md").write_text(f"# API Reference ({version})")
            (version_dir / "guide.md").write_text(f"# User Guide ({version})")

        retriever = Retriever()

        # Load each version with metadata
        for version in ["v1", "v2"]:
            retriever.add_directory(
                str(Path(tmpdir) / version),
                pattern="*.md",
                metadata={"version": version},
            )

        print("\n[OK] Loaded documentation for v1 and v2")

        # Search only v2
        results = retriever.search("API", k=2, filter={"version": "v2"})
        print("\nSearch 'API' filtered to v2:")
        for r in results:
            print(f"  {r.document.metadata.get('version')}: {r.document.text[:30]}...")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Directory Loading Examples")
    print("=" * 60)
    print("\nadd_directory() loads entire directories at once.")
    print("Great for documentation, code repos, knowledge bases.\n")

    basic_directory()
    glob_patterns()
    recursive_loading()
    multiple_patterns()
    exclude_patterns()
    load_code_files()
    directory_with_metadata()

    print("\n" + "=" * 60)
    print("Examples complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. add_directory() loads all files matching pattern")
    print("  2. Use glob patterns (*.md, **/*.py) for filtering")
    print("  3. recursive=True for nested directories")
    print("  4. Add metadata for filtering by version, category, etc.")
    print("  5. Perfect for docs, code repos, knowledge bases")


if __name__ == "__main__":
    main()
