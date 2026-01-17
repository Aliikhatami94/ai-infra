"""Researcher subagent for finding information.

Phase 3.3 of EXECUTOR_1.md: Specialized agent for searching
documentation, researching best practices, and finding examples.

Phase 3.4 of EXECUTOR_1.md: Enhanced with research tools for
web search, documentation lookup, and package registry search.

The ResearcherAgent handles:
- Web search for technical information
- Documentation lookup from package registries
- Package search across PyPI, npm, crates.io
- Local documentation search via shell
- Best practices research
- Example code discovery
- Technical investigation

Example:
    ```python
    from ai_infra.executor.agents import ResearcherAgent

    agent = ResearcherAgent()
    result = await agent.execute(task, context)
    ```
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ai_infra.callbacks import MetricsCallbacks
from ai_infra.executor.agents.base import (
    RESEARCHER_SYSTEM_PROMPT,
    ExecutionContext,
    SubAgent,
    SubAgentResult,
)
from ai_infra.executor.agents.registry import SubAgentRegistry, SubAgentType
from ai_infra.executor.tools.research import (
    lookup_docs,
    search_packages,
    web_search,
)
from ai_infra.llm.agent import Agent
from ai_infra.llm.shell.session import SessionConfig, ShellSession
from ai_infra.llm.shell.tool import create_shell_tool, set_current_session
from ai_infra.logging import get_logger

if TYPE_CHECKING:
    from ai_infra.executor.todolist import TodoItem

__all__ = ["ResearcherAgent"]

logger = get_logger("executor.agents.researcher")


# =============================================================================
# Researcher Prompt Template
# =============================================================================

RESEARCHER_PROMPT = """You are a technical researcher finding information.

## Task Context
Task: {task_title}
Description: {task_description}

## Project Context
- Workspace: {workspace}
- Project Type: {project_type}
{enriched_context}
## Available Research Tools

You have access to the following specialized research tools:

### 1. web_search
Search the web for technical information:
```
web_search(query="how to implement OAuth2 in FastAPI", max_results=5)
```

### 2. lookup_docs
Look up package documentation:
```
lookup_docs(package="fastapi", registry="pypi")
lookup_docs(package="react", registry="npm")
```

### 3. search_packages
Search package registries for libraries:
```
search_packages(query="async http client", registry="pypi")
search_packages(query="state management", registry="npm")
```

### 4. Shell commands
For local codebase exploration:
```bash
# Find relevant docs
find . -name "*.md" | xargs grep -l "keyword"

# Search existing code
grep -r "pattern" src/
```

## Research Process

1. **Understand the task** - What information is needed?
2. **Search locally first** - Check the codebase for existing solutions
3. **Use research tools** - Search web/docs for external information
4. **Find packages** - Search registries for relevant libraries
5. **Synthesize findings** - Combine all information into actionable results

## Output Format

After research, provide:

### Summary
Brief overview of findings.

### Key Discoveries
- Discovery 1
- Discovery 2

### Relevant Code Examples
```python
# Example code found
```

### Recommendations
- Recommendation 1
- Recommendation 2
"""


# =============================================================================
# ResearcherAgent
# =============================================================================


@SubAgentRegistry.register(SubAgentType.RESEARCHER)
class ResearcherAgent(SubAgent):
    """Specialized agent for research and information gathering.

    Searches documentation, finds examples, and researches
    best practices to inform development decisions.
    """

    name = "Researcher"
    description = "Searches for information and best practices"
    model = "claude-sonnet-4-20250514"
    system_prompt = RESEARCHER_SYSTEM_PROMPT

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float = 300.0,
        shell_timeout: float = 60.0,
    ) -> None:
        """Initialize the researcher agent.

        Args:
            model: Optional model override.
            timeout: Maximum execution time.
            shell_timeout: Timeout for shell commands.
        """
        super().__init__(model=model)
        self._timeout = timeout
        self._shell_timeout = shell_timeout

    def _get_tools(self) -> list[Any]:
        """Get tools for this agent.

        Returns research tools for web search, documentation lookup,
        and package registry search. Shell tool is added during execute()
        since it requires an active session.
        """
        return [web_search, lookup_docs, search_packages]

    async def execute(
        self,
        task: TodoItem,
        context: ExecutionContext,
    ) -> SubAgentResult:
        """Execute a research task.

        Args:
            task: The research task.
            context: Execution context.

        Returns:
            SubAgentResult with research findings.
        """
        start_time = time.perf_counter()

        logger.info(f"ResearcherAgent researching: {task.title}")

        # Create shell session
        session_config = SessionConfig(workspace_root=context.workspace)
        session = ShellSession(session_config)

        try:
            await session.start()
            set_current_session(session)

            # Create shell tool
            shell_tool = create_shell_tool(
                session=session,
                default_timeout=self._shell_timeout,
            )

            # Combine shell tool with research tools
            tools = [shell_tool, *self._get_tools()]

            # Format the prompt with enriched context (Phase 16.5.5)
            # Phase 16.5.5.10: Include enriched context in prompt
            enriched_context = context.format_for_prompt()
            if enriched_context:
                enriched_context = "\n" + enriched_context + "\n"

            prompt = RESEARCHER_PROMPT.format(
                task_title=task.title,
                task_description=task.description or "Research this topic",
                workspace=str(context.workspace),
                project_type=context.project_type,
                enriched_context=enriched_context,
            )

            # Create metrics callback to track token usage
            metrics_cb = MetricsCallbacks()

            # Create agent with callbacks
            agent = Agent(
                tools=tools,
                model_name=self._model,
                system=prompt,
                callbacks=metrics_cb,
            )

            # Run research
            result = await agent.arun(f"Research: {task.title}\n\n{task.description or ''}")

            # Extract output
            output = self._extract_output(result)

            duration_ms = (time.perf_counter() - start_time) * 1000

            # Get token metrics from callback
            llm_metrics = metrics_cb.get_summary().get("llm", {})

            logger.info(
                f"ResearcherAgent completed in {duration_ms:.0f}ms, "
                f"{llm_metrics.get('total_tokens', 0)} tokens"
            )

            return SubAgentResult(
                success=True,
                output=output,
                metrics={
                    "duration_ms": duration_ms,
                    "commands_run": len(session.command_history),
                    "tokens_in": llm_metrics.get("total_tokens", 0),
                    "tokens_out": 0,
                    "total_tokens": llm_metrics.get("total_tokens", 0),
                    "model": self._model,
                    "agent_type": self.name,
                    "llm_calls": llm_metrics.get("calls", 0),
                },
            )

        except TimeoutError as e:
            logger.warning(f"ResearcherAgent timed out: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=f"Research timed out after {self._timeout}s",
                metrics={"duration_ms": duration_ms},
            )

        except Exception as e:
            logger.exception(f"ResearcherAgent failed: {e}")
            duration_ms = (time.perf_counter() - start_time) * 1000
            return SubAgentResult(
                success=False,
                error=str(e),
                metrics={"duration_ms": duration_ms},
            )

        finally:
            set_current_session(None)
            await session.close()

    def _extract_output(self, result: Any) -> str:
        """Extract text output from agent result."""
        if hasattr(result, "content"):
            return str(result.content)
        elif isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return str(result.get("content", result.get("output", str(result))))
        return str(result)
