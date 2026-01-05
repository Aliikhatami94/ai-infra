"""Unit tests for the API docs extraction script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add scripts directory to path for importing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from extract_api_docs import (
    extract_class,
    extract_function,
    extract_parameter,
    load_class,
    resolve_member,
)


class TestExtractParameter:
    """Tests for extract_parameter function."""

    def test_required_parameter(self) -> None:
        """Test extraction of required parameter."""
        param = MagicMock()
        param.name = "model"
        param.annotation = "str"
        param.default = None

        result = extract_parameter(param)

        assert result["name"] == "model"
        assert result["type"] == "str"
        assert result["default"] is None
        assert result["required"] is True

    def test_optional_parameter_with_default(self) -> None:
        """Test extraction of optional parameter with default value."""
        param = MagicMock()
        param.name = "temperature"
        param.annotation = "float"
        param.default = "0.7"

        result = extract_parameter(param)

        assert result["name"] == "temperature"
        assert result["type"] == "float"
        assert result["default"] == "0.7"
        assert result["required"] is False

    def test_parameter_without_type_annotation(self) -> None:
        """Test extraction of parameter without type annotation."""
        param = MagicMock()
        param.name = "data"
        param.annotation = None
        param.default = None

        result = extract_parameter(param)

        assert result["name"] == "data"
        assert result["type"] is None
        assert result["required"] is True

    def test_self_parameter_not_required(self) -> None:
        """Test that self parameter is not marked as required."""
        param = MagicMock()
        param.name = "self"
        param.annotation = None
        param.default = None

        result = extract_parameter(param)

        assert result["name"] == "self"
        assert result["required"] is False

    def test_cls_parameter_not_required(self) -> None:
        """Test that cls parameter is not marked as required."""
        param = MagicMock()
        param.name = "cls"
        param.annotation = None
        param.default = None

        result = extract_parameter(param)

        assert result["name"] == "cls"
        assert result["required"] is False


class TestExtractFunction:
    """Tests for extract_function function."""

    def test_simple_method(self) -> None:
        """Test extraction of a simple method."""
        func = MagicMock()
        func.name = "run"
        func.docstring = MagicMock()
        func.docstring.value = "Run the agent."
        func.returns = "str"
        func.is_async = False

        param = MagicMock()
        param.name = "prompt"
        param.annotation = "str"
        param.default = None
        func.parameters = [param]

        result = extract_function(func)

        assert result["name"] == "run"
        assert result["docstring"] == "Run the agent."
        assert result["returns"] == "str"
        assert result["is_async"] is False
        assert "prompt: str" in result["signature"]

    def test_async_method(self) -> None:
        """Test extraction of async method."""
        func = MagicMock()
        func.name = "arun"
        func.docstring = None
        func.returns = "str"
        func.is_async = True
        func.parameters = []

        result = extract_function(func)

        assert result["name"] == "arun"
        assert result["is_async"] is True

    def test_method_without_docstring(self) -> None:
        """Test extraction of method without docstring."""
        func = MagicMock()
        func.name = "helper"
        func.docstring = None
        func.returns = None
        func.is_async = False
        func.parameters = []

        result = extract_function(func)

        assert result["name"] == "helper"
        assert result["docstring"] is None
        assert result["returns"] is None

    def test_method_with_multiple_parameters(self) -> None:
        """Test extraction of method with multiple parameters."""
        func = MagicMock()
        func.name = "configure"
        func.docstring = MagicMock()
        func.docstring.value = "Configure the model."
        func.returns = "None"
        func.is_async = False

        # Create parameters
        self_param = MagicMock()
        self_param.name = "self"
        self_param.annotation = None
        self_param.default = None

        model_param = MagicMock()
        model_param.name = "model"
        model_param.annotation = "str"
        model_param.default = None

        temp_param = MagicMock()
        temp_param.name = "temperature"
        temp_param.annotation = "float"
        temp_param.default = "0.7"

        func.parameters = [self_param, model_param, temp_param]

        result = extract_function(func)

        # self should be excluded from signature
        assert "self" not in result["signature"]
        assert "model: str" in result["signature"]
        assert "temperature: float = 0.7" in result["signature"]

    def test_method_with_complex_return_type(self) -> None:
        """Test extraction of method with complex return type."""
        func = MagicMock()
        func.name = "stream"
        func.docstring = None
        func.returns = "Iterator[str]"
        func.is_async = False
        func.parameters = []

        result = extract_function(func)

        assert result["returns"] == "Iterator[str]"
        assert "-> Iterator[str]" in result["signature"]


class TestExtractClass:
    """Tests for extract_class function."""

    def test_class_with_docstring(self) -> None:
        """Test extraction of class with docstring."""
        from griffe import Class

        cls = MagicMock(spec=Class)
        cls.name = "LLM"
        cls.docstring = MagicMock()
        cls.docstring.value = "Language model wrapper."
        cls.bases = ["BaseModel"]
        cls.members = {}

        result = extract_class(cls, "ai_infra.llm")

        assert result["name"] == "LLM"
        assert result["module"] == "ai_infra.llm"
        assert result["docstring"] == "Language model wrapper."
        assert "BaseModel" in result["bases"]

    def test_class_without_docstring(self) -> None:
        """Test extraction of class without docstring."""
        from griffe import Class

        cls = MagicMock(spec=Class)
        cls.name = "Helper"
        cls.docstring = None
        cls.bases = []
        cls.members = {}

        result = extract_class(cls, "ai_infra.utils")

        assert result["name"] == "Helper"
        assert result["docstring"] is None

    def test_class_with_init_parameters(self) -> None:
        """Test extraction of class __init__ parameters."""
        from griffe import Class, Function

        cls = MagicMock(spec=Class)
        cls.name = "Agent"
        cls.docstring = MagicMock()
        cls.docstring.value = "AI Agent."
        cls.bases = []

        # Create __init__ method - must be a real Function mock
        init_func = MagicMock(spec=Function)
        init_func.name = "__init__"
        init_func.docstring = None
        init_func.returns = None
        init_func.is_async = False

        self_param = MagicMock()
        self_param.name = "self"
        self_param.annotation = None
        self_param.default = None

        model_param = MagicMock()
        model_param.name = "model"
        model_param.annotation = "str"
        model_param.default = None

        init_func.parameters = [self_param, model_param]

        cls.members = {"__init__": init_func}

        result = extract_class(cls, "ai_infra.llm")

        # Parameters should exclude self
        assert len(result["parameters"]) == 1
        assert result["parameters"][0]["name"] == "model"

    def test_class_excludes_private_methods(self) -> None:
        """Test that private methods are excluded."""
        from griffe import Class, Function

        cls = MagicMock(spec=Class)
        cls.name = "LLM"
        cls.docstring = None
        cls.bases = []

        # Public method
        public_func = MagicMock(spec=Function)
        public_func.name = "run"
        public_func.docstring = None
        public_func.returns = None
        public_func.is_async = False
        public_func.parameters = []

        # Private method
        private_func = MagicMock(spec=Function)
        private_func.name = "_internal"
        private_func.docstring = None
        private_func.returns = None
        private_func.is_async = False
        private_func.parameters = []

        cls.members = {"run": public_func, "_internal": private_func}

        result = extract_class(cls, "ai_infra.llm")

        method_names = [m["name"] for m in result["methods"]]
        assert "run" in method_names
        assert "_internal" not in method_names

    def test_class_includes_init_in_methods(self) -> None:
        """Test that __init__ is included in methods."""
        from griffe import Class, Function

        cls = MagicMock(spec=Class)
        cls.name = "LLM"
        cls.docstring = None
        cls.bases = []

        init_func = MagicMock(spec=Function)
        init_func.name = "__init__"
        init_func.docstring = MagicMock()
        init_func.docstring.value = "Initialize the LLM."
        init_func.returns = None
        init_func.is_async = False
        init_func.parameters = []

        cls.members = {"__init__": init_func}

        result = extract_class(cls, "ai_infra.llm")

        method_names = [m["name"] for m in result["methods"]]
        assert "__init__" in method_names


class TestResolveMember:
    """Tests for resolve_member function."""

    def test_resolve_function(self) -> None:
        """Test resolving a regular function."""
        from griffe import Function

        func = MagicMock(spec=Function)
        result = resolve_member(func)
        assert result == func

    def test_resolve_alias(self) -> None:
        """Test resolving an alias to its target."""
        from griffe import Alias

        target_func = MagicMock()
        alias = MagicMock(spec=Alias)
        alias.target = target_func

        result = resolve_member(alias)
        assert result == target_func

    def test_resolve_alias_with_error(self) -> None:
        """Test handling alias resolution errors."""
        from griffe import Alias

        alias = MagicMock(spec=Alias)
        # Make target raise an exception when accessed
        type(alias).target = property(lambda self: (_ for _ in ()).throw(Exception("err")))

        # Should return None on error
        result = resolve_member(alias)
        assert result is None


class TestLoadClass:
    """Tests for load_class function."""

    def test_load_valid_class(self) -> None:
        """Test loading a valid class from ai_infra."""
        search_paths = [Path(__file__).parent.parent.parent / "src"]
        cls = load_class("ai_infra.llm.LLM", search_paths)

        assert cls is not None
        assert cls.name == "LLM"

    def test_load_invalid_class_path(self) -> None:
        """Test loading with invalid class path format."""
        search_paths = [Path(__file__).parent.parent.parent / "src"]
        cls = load_class("invalid_path", search_paths)

        assert cls is None

    def test_load_nonexistent_class(self) -> None:
        """Test loading a class that doesn't exist."""
        search_paths = [Path(__file__).parent.parent.parent / "src"]
        cls = load_class("ai_infra.llm.NonExistent", search_paths)

        assert cls is None


class TestIntegration:
    """Integration tests for the extraction script."""

    def test_full_extraction_llm_class(self) -> None:
        """Test full extraction of LLM class."""
        search_paths = [Path(__file__).parent.parent.parent / "src"]
        cls = load_class("ai_infra.llm.LLM", search_paths)

        assert cls is not None

        result = extract_class(cls, "ai_infra.llm")

        # Verify structure
        assert result["name"] == "LLM"
        assert result["module"] == "ai_infra.llm"
        assert result["docstring"] is not None
        assert isinstance(result["parameters"], list)
        assert isinstance(result["methods"], list)

        # Should have common LLM methods
        method_names = [m["name"] for m in result["methods"]]
        assert "__init__" in method_names

    def test_json_serializable(self) -> None:
        """Test that extracted data is JSON serializable."""
        search_paths = [Path(__file__).parent.parent.parent / "src"]
        cls = load_class("ai_infra.llm.LLM", search_paths)

        assert cls is not None

        result = extract_class(cls, "ai_infra.llm")

        # Should not raise
        json_str = json.dumps(result)
        assert isinstance(json_str, str)

        # Should round-trip
        parsed = json.loads(json_str)
        assert parsed["name"] == "LLM"

    def test_extraction_to_file(self, tmp_path: Path) -> None:
        """Test extraction writes valid JSON file."""
        search_paths = [Path(__file__).parent.parent.parent / "src"]
        cls = load_class("ai_infra.llm.LLM", search_paths)

        assert cls is not None

        result = extract_class(cls, "ai_infra.llm")

        output_file = tmp_path / "llm.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        # Verify file exists and is valid JSON
        assert output_file.exists()
        with open(output_file) as f:
            loaded = json.load(f)

        assert loaded["name"] == "LLM"
        assert "methods" in loaded
        assert "parameters" in loaded
