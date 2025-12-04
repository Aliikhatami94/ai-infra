from __future__ import annotations

import typer
from svc_infra.cli.foundation.typer_bootstrap import pre_cli

from ai_infra.cli.cmds import (
    _HELP,
    register_chat,
    register_discovery,
    register_imagegen,
    register_mcp,
    register_multimodal,
    register_stdio_publisher,
)

app = typer.Typer(no_args_is_help=True, add_completion=False, help=_HELP)
pre_cli(app)
register_chat(app)
register_stdio_publisher(app)
register_discovery(app)
register_imagegen(app)
register_multimodal(app)
register_mcp(app)


def main():
    app()


if __name__ == "__main__":
    main()
