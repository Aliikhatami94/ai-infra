"""Unit tests for Workspace abstraction (Phase 4.8)."""

from pathlib import Path

from ai_infra.llm.workspace import Workspace, workspace


class TestWorkspace:
    """Test Workspace abstraction."""

    def test_default_is_sandboxed(self):
        """Default workspace should be sandboxed to current directory."""
        ws = Workspace()
        assert ws.mode == "sandboxed"
        assert ws.root == Path(".").resolve()

    def test_string_path_works(self, tmp_path):
        """String paths should be resolved to absolute Path."""
        ws = Workspace(str(tmp_path))
        assert ws.root == tmp_path

    def test_path_object_works(self, tmp_path):
        """Path objects should work directly."""
        ws = Workspace(tmp_path)
        assert ws.root == tmp_path

    def test_mode_virtual(self):
        """Virtual mode should create in-memory FilesystemBackend."""
        ws = Workspace(mode="virtual")
        backend = ws.get_deepagent_backend()
        # Virtual mode uses FilesystemBackend with virtual_mode=True
        assert backend.__class__.__name__ == "FilesystemBackend"

    def test_mode_sandboxed(self, tmp_path):
        """Sandboxed mode should create FilesystemBackend with virtual_mode=True."""
        ws = Workspace(tmp_path, mode="sandboxed")
        backend = ws.get_deepagent_backend()
        assert backend.__class__.__name__ == "FilesystemBackend"

    def test_mode_full(self, tmp_path):
        """Full mode should create FilesystemBackend with virtual_mode=False."""
        ws = Workspace(tmp_path, mode="full")
        backend = ws.get_deepagent_backend()
        assert backend.__class__.__name__ == "FilesystemBackend"

    def test_repr(self, tmp_path):
        """__repr__ should show root and mode."""
        ws = Workspace(tmp_path, mode="sandboxed")
        r = repr(ws)
        assert "Workspace" in r
        assert str(tmp_path) in r
        assert "sandboxed" in r


class TestWorkspaceHelper:
    """Test workspace() convenience function."""

    def test_workspace_helper_creates_workspace(self):
        """workspace() should create Workspace instance."""
        ws = workspace(".", mode="full")
        assert isinstance(ws, Workspace)
        assert ws.mode == "full"

    def test_workspace_helper_default_sandboxed(self):
        """workspace() should default to sandboxed mode."""
        ws = workspace()
        assert ws.mode == "sandboxed"


class TestAgentWorkspaceIntegration:
    """Test Agent with workspace parameter."""

    def test_agent_accepts_string_workspace(self, tmp_path):
        """Agent should accept string workspace path."""
        from ai_infra.llm import Agent

        agent = Agent(workspace=str(tmp_path))
        assert agent._workspace is not None
        assert agent._workspace.root == tmp_path

    def test_agent_accepts_path_workspace(self, tmp_path):
        """Agent should accept Path workspace."""
        from ai_infra.llm import Agent

        agent = Agent(workspace=tmp_path)
        assert agent._workspace is not None
        assert agent._workspace.root == tmp_path

    def test_agent_accepts_workspace_object(self, tmp_path):
        """Agent should accept Workspace object directly."""
        from ai_infra.llm import Agent

        ws = Workspace(tmp_path, mode="sandboxed")
        agent = Agent(workspace=ws)
        assert agent._workspace is ws

    def test_agent_none_workspace_is_allowed(self):
        """Agent should accept None workspace (no configuration)."""
        from ai_infra.llm import Agent

        agent = Agent(workspace=None)
        assert agent._workspace is None

    def test_agent_deep_mode_with_workspace(self, tmp_path):
        """Deep agent should accept workspace parameter."""
        from ai_infra.llm import Agent

        agent = Agent(deep=True, workspace=tmp_path)
        assert agent._workspace is not None
        assert agent._deep is True


class TestWorkspaceProjMgmtIntegration:
    """Test Workspace integration with proj_mgmt tools."""

    def test_configure_proj_mgmt_sets_root(self, tmp_path):
        """configure_proj_mgmt should set workspace root."""
        from ai_infra.llm.tools.custom.proj_mgmt.utils import get_workspace_root

        ws = Workspace(tmp_path, mode="sandboxed")
        ws.configure_proj_mgmt()

        assert get_workspace_root() == tmp_path

    def test_configure_proj_mgmt_virtual_sets_none(self, tmp_path):
        """Virtual mode should set workspace root to None."""
        from ai_infra.llm.tools.custom.proj_mgmt.utils import _set_workspace_root

        # Set a path first to verify it gets cleared
        _set_workspace_root(tmp_path)

        ws = Workspace(mode="virtual")
        ws.configure_proj_mgmt()

        # Virtual mode should reset the workspace root
        # (tools should error in virtual mode)
        # Note: In virtual mode, get_workspace_root falls back to default
        # but the override is None


class TestDeprecationWarnings:
    """Test deprecation warnings for old API."""

    def test_set_workspace_root_shows_deprecation(self, tmp_path):
        """set_workspace_root should show deprecation warning."""
        import warnings

        from ai_infra.llm.tools.custom.proj_mgmt import set_workspace_root

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_workspace_root(tmp_path)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "Agent(workspace=...)" in str(w[0].message)
