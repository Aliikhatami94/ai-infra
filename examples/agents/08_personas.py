#!/usr/bin/env python
"""Config-Driven Agent Personas Example.

This example demonstrates:
- Loading agents from YAML persona files
- Creating personas programmatically
- Tool whitelists and blacklists
- Approval lists for sensitive tools
- Overriding persona settings

Personas provide a declarative way to define agent behavior,
making it easy to manage different agent configurations.

Required API Keys:
- OPENAI_API_KEY or ANTHROPIC_API_KEY
"""

import asyncio
from pathlib import Path

from ai_infra import Agent
from ai_infra.llm.personas import Persona, build_tool_filter

# =============================================================================
# Sample Tools
# =============================================================================


def query_database(query: str) -> str:
    """Execute a database query.

    Args:
        query: SQL query string.

    Returns:
        Query results.
    """
    # Simulated query results
    return f"Results for: {query}\n[3 rows returned]"


def create_chart(data: str, chart_type: str = "bar") -> str:
    """Create a chart from data.

    Args:
        data: Data to visualize.
        chart_type: Type of chart (bar, line, pie).

    Returns:
        Chart creation result.
    """
    return f"Created {chart_type} chart with data: {data}"


def delete_record(record_id: str) -> str:
    """Delete a database record.

    Args:
        record_id: ID of record to delete.

    Returns:
        Deletion result.
    """
    return f"Deleted record: {record_id}"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email.

    Args:
        to: Recipient email.
        subject: Email subject.
        body: Email body.

    Returns:
        Send result.
    """
    return f"Email sent to {to}: {subject}"


def search_files(pattern: str) -> str:
    """Search files matching a pattern.

    Args:
        pattern: Glob pattern.

    Returns:
        Matching files.
    """
    return f"Found 5 files matching: {pattern}"


# =============================================================================
# Loading from YAML File
# =============================================================================


async def from_yaml_file():
    """Load agent from a YAML persona file."""
    print("=" * 60)
    print("Loading Agent from YAML Persona File")
    print("=" * 60)

    # Check if persona file exists
    persona_path = Path(__file__).parent / "personas" / "analyst.yaml"

    if persona_path.exists():
        print(f"Loading persona from: {persona_path}")

        # Load agent from YAML file
        agent = Agent.from_persona(
            str(persona_path),
            tools=[query_database, create_chart, delete_record, send_email],
        )

        print("Created agent with persona: analyst")
        print("  - System prompt loaded from YAML")
        print("  - Tool restrictions applied")
        print()

        result = await agent.arun("Query the database for sales data and create a chart.")
        print(f"Result: {result}")
    else:
        print(f"Persona file not found: {persona_path}")
        print(
            'Run: python -c "from examples.agents._08_personas import create_sample_personas; create_sample_personas()"'
        )


# =============================================================================
# Inline Persona Configuration
# =============================================================================


async def inline_persona():
    """Create agent with inline persona configuration."""
    print("\n" + "=" * 60)
    print("Inline Persona Configuration")
    print("=" * 60)

    # Define persona inline (no YAML file needed)
    agent = Agent.from_persona(
        name="data_analyst",
        prompt="""You are a senior data analyst.
        - Always verify data accuracy before making claims
        - Use charts to visualize trends
        - Never delete data without explicit confirmation""",
        allowed_tools=["query_database", "create_chart", "search_files"],
        deny=["delete_record"],  # Blocked even if passed to tools
        approve=["send_email"],  # Requires human approval
        temperature=0.3,
        tools=[query_database, create_chart, delete_record, send_email, search_files],
    )

    print("Created inline persona with:")
    print("  - allowed_tools: [query_database, create_chart, search_files]")
    print("  - deny: [delete_record]")
    print("  - approve: [send_email]")
    print()

    result = await agent.arun("Search for all CSV files and create a summary chart.")
    print(f"Result: {result}")


# =============================================================================
# Persona Object
# =============================================================================


def persona_object():
    """Create and use Persona objects directly."""
    print("\n" + "=" * 60)
    print("Using Persona Objects")
    print("=" * 60)

    # Create persona programmatically
    analyst_persona = Persona(
        name="analyst",
        prompt="You are a data analyst. Be precise and thorough.",
        tools=["query_database", "create_chart"],
        deny=["delete_record"],
        approve=["send_email"],
        temperature=0.3,
        metadata={
            "department": "Data Science",
            "version": "2.0",
        },
    )

    print(f"Persona: {analyst_persona.name}")
    print(f"  Tools: {analyst_persona.tools}")
    print(f"  Deny: {analyst_persona.deny}")
    print(f"  Approve: {analyst_persona.approve}")
    print(f"  Metadata: {analyst_persona.metadata}")

    # Convert to dict (for serialization)
    as_dict = analyst_persona.to_dict()
    print(f"\nAs dict: {as_dict}")

    # Create from dict
    from_dict = Persona.from_dict(as_dict)
    print(f"From dict: {from_dict.name}")


# =============================================================================
# Tool Filtering
# =============================================================================


def tool_filtering_demo():
    """Demonstrate tool filtering with allow/deny lists."""
    print("\n" + "=" * 60)
    print("Tool Filtering (Allow/Deny Lists)")
    print("=" * 60)

    # Whitelist only (allow specific tools)
    print("\n1. Whitelist only:")
    whitelist_filter = build_tool_filter(
        allowed=["query_database", "create_chart"],
        denied=None,
    )
    test_tools = ["query_database", "create_chart", "delete_record", "send_email"]
    for tool in test_tools:
        allowed = whitelist_filter(tool) if whitelist_filter else True
        status = "[OK]" if allowed else "[X]"
        print(f"   {status} {tool}")

    # Blacklist only (deny specific tools)
    print("\n2. Blacklist only:")
    blacklist_filter = build_tool_filter(
        allowed=None,
        denied=["delete_record", "drop_table"],
    )
    for tool in test_tools:
        allowed = blacklist_filter(tool) if blacklist_filter else True
        status = "[OK]" if allowed else "[X]"
        print(f"   {status} {tool}")

    # Combined (whitelist + blacklist)
    print("\n3. Combined (whitelist with additional blacklist):")
    combined_filter = build_tool_filter(
        allowed=["query_database", "create_chart", "delete_record"],
        denied=["delete_record"],  # Further restricts whitelist
    )
    for tool in test_tools:
        allowed = combined_filter(tool) if combined_filter else True
        status = "[OK]" if allowed else "[X]"
        print(f"   {status} {tool}")


# =============================================================================
# Overriding Persona Settings
# =============================================================================


async def override_persona():
    """Override persona settings at creation time."""
    print("\n" + "=" * 60)
    print("Overriding Persona Settings")
    print("=" * 60)

    persona_path = Path(__file__).parent / "personas" / "analyst.yaml"

    if not persona_path.exists():
        print("Creating persona inline for demo...")

        # Create with overrides
        _agent = Agent.from_persona(
            name="analyst_override",
            prompt="You are a data analyst (with overrides).",
            allowed_tools=["query_database"],  # Override to restrict
            temperature=0.1,  # Override temperature
            provider="openai",  # Override provider
            model_name="gpt-4o-mini",  # Override model
            tools=[query_database, create_chart],
        )
    else:
        # Override settings from YAML
        _agent = Agent.from_persona(
            str(persona_path),
            prompt="Overridden prompt: Focus only on sales data.",  # Override YAML prompt
            temperature=0.1,  # Override YAML temperature
            tools=[query_database, create_chart],
        )

    print("Persona settings can be overridden:")
    print("  - prompt: Override system prompt")
    print("  - temperature: Override temperature")
    print("  - provider/model_name: Override model selection")
    print("  - allowed_tools/deny/approve: Override tool lists")


# =============================================================================
# Multiple Personas
# =============================================================================


async def multiple_personas():
    """Use multiple personas for different tasks."""
    print("\n" + "=" * 60)
    print("Multiple Personas for Different Tasks")
    print("=" * 60)

    # Analyst persona - read-only, analytical
    _analyst = Agent.from_persona(
        name="analyst",
        prompt="You analyze data and create visualizations. Read-only access.",
        allowed_tools=["query_database", "create_chart"],
        deny=["delete_record", "send_email"],
        temperature=0.3,
        tools=[query_database, create_chart, delete_record, send_email],
    )

    # Admin persona - full access with approval
    _admin = Agent.from_persona(
        name="admin",
        prompt="You are a database administrator with full access.",
        allowed_tools=["query_database", "delete_record"],
        approve=["delete_record"],  # Requires approval
        temperature=0.1,
        tools=[query_database, create_chart, delete_record, send_email],
    )

    # Communication persona - can send emails
    _communicator = Agent.from_persona(
        name="communicator",
        prompt="You help with sending reports and communications.",
        allowed_tools=["query_database", "send_email"],
        approve=["send_email"],
        tools=[query_database, create_chart, delete_record, send_email],
    )

    print("Created personas for different roles:")
    print("  - analyst: Read-only data analysis")
    print("  - admin: Full access with delete approval")
    print("  - communicator: Reports and email")


# =============================================================================
# Save and Load Personas
# =============================================================================


def save_and_load_personas():
    """Save personas to YAML and load them back."""
    print("\n" + "=" * 60)
    print("Saving and Loading Personas")
    print("=" * 60)

    # Create a persona
    support_persona = Persona(
        name="support",
        prompt="""You are a customer support agent.
        - Be helpful and friendly
        - Escalate complex issues
        - Never share sensitive data""",
        tools=["search_knowledge_base", "create_ticket", "respond_to_customer"],
        deny=["access_billing", "modify_account"],
        approve=["escalate_issue"],
        temperature=0.7,
        metadata={"team": "Support", "tier": "L1"},
    )

    # Save to YAML
    save_path = Path("/tmp/support_persona.yaml")
    support_persona.save_yaml(save_path)
    print(f"Saved persona to: {save_path}")

    # Show saved content
    print("\nSaved YAML content:")
    print(save_path.read_text())

    # Load it back
    loaded = Persona.from_yaml(save_path)
    print(f"\nLoaded persona: {loaded.name}")
    print(f"  Tools: {loaded.tools}")


# =============================================================================
# Create Sample Persona Files
# =============================================================================


def create_sample_personas():
    """Create sample persona YAML files."""
    personas_dir = Path(__file__).parent / "personas"
    personas_dir.mkdir(exist_ok=True)

    # Analyst persona
    analyst = Persona(
        name="analyst",
        prompt="""You are a senior data analyst.
- Always verify data accuracy before making claims
- Use charts to visualize trends
- Explain your reasoning clearly
- Never delete data without explicit confirmation""",
        tools=["query_database", "create_chart", "search_files"],
        deny=["delete_record", "drop_table"],
        approve=["send_email", "publish_report"],
        temperature=0.3,
    )
    analyst.save_yaml(personas_dir / "analyst.yaml")

    # Support persona
    support = Persona(
        name="support",
        prompt="""You are a customer support specialist.
- Be helpful, friendly, and patient
- Search the knowledge base before answering
- Create tickets for unresolved issues
- Escalate complex problems to L2 support""",
        tools=["search_knowledge_base", "create_ticket", "get_customer_info"],
        deny=["modify_billing", "delete_account"],
        approve=["escalate_issue", "issue_refund"],
        temperature=0.7,
    )
    support.save_yaml(personas_dir / "support.yaml")

    # Developer persona
    developer = Persona(
        name="developer",
        prompt="""You are a software developer.
- Write clean, well-documented code
- Follow best practices and coding standards
- Run tests before committing
- Review code for security issues""",
        tools=["read_file", "write_file", "run_tests", "git_commit"],
        deny=["rm_rf", "drop_database"],
        approve=["deploy_production", "merge_to_main"],
        temperature=0.2,
    )
    developer.save_yaml(personas_dir / "developer.yaml")

    print(f"Created sample personas in: {personas_dir}")
    print("  - analyst.yaml")
    print("  - support.yaml")
    print("  - developer.yaml")


# =============================================================================
# Persona YAML Format Reference
# =============================================================================


def yaml_format_reference():
    """Show the YAML format reference for personas."""
    print("\n" + "=" * 60)
    print("Persona YAML Format Reference")
    print("=" * 60)

    print("""
# Persona YAML Format

name: analyst                    # Required: Persona identifier
prompt: |                        # Required: System prompt
  You are a data analyst.
  Be precise and data-driven.

tools:                           # Optional: Allowed tool names (whitelist)
  - query_database
  - create_chart
  - search_files

deny:                            # Optional: Blocked tool names (blacklist)
  - delete_record
  - drop_table

approve:                         # Optional: Tools requiring human approval
  - send_email
  - publish_report

# Model configuration (optional)
provider: openai                 # LLM provider override
model_name: gpt-4o-mini          # Model override
temperature: 0.3                 # Temperature override
max_tokens: 4000                 # Max tokens override

# Custom metadata (optional, any key/value)
department: Data Science
version: "2.0"
author: Jane Doe
""")


if __name__ == "__main__":
    # Create sample persona files
    print("Creating sample persona files...")
    create_sample_personas()
    print()

    # Run examples
    asyncio.run(from_yaml_file())
    asyncio.run(inline_persona())
    persona_object()
    tool_filtering_demo()
    asyncio.run(override_persona())
    asyncio.run(multiple_personas())
    save_and_load_personas()
    yaml_format_reference()
