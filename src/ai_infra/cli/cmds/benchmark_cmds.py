"""CLI commands for running benchmarks.

Phase 12.1 - Benchmark CLI.

Provides CLI commands for running benchmarks:
- `ai benchmark all`: Run all benchmarks
- `ai benchmark shell`: Run shell execution benchmarks

Example:
    $ ai benchmark all --iterations 20 --output results.json
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    pass


app = typer.Typer(
    name="benchmark",
    help="Run performance benchmarks.",
    no_args_is_help=True,
)


# =============================================================================
# Benchmark Configuration
# =============================================================================


def _get_shell_benchmarks() -> dict[str, Any]:
    """Get shell execution benchmark functions."""
    from ai_infra.llm.shell.limits import LimitedExecutionPolicy, ResourceLimits
    from ai_infra.llm.shell.types import ShellConfig

    async def bench_shell_echo():
        """Benchmark simple echo command."""
        limits = ResourceLimits.permissive()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)
        await policy.execute("echo hello", config)

    async def bench_shell_python_version():
        """Benchmark python --version command."""
        limits = ResourceLimits.permissive()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)
        await policy.execute("python3 --version", config)

    async def bench_shell_ls():
        """Benchmark ls command."""
        limits = ResourceLimits.permissive()
        policy = LimitedExecutionPolicy(limits=limits)
        config = ShellConfig(timeout=10.0)
        await policy.execute("ls -la /tmp", config)

    return {
        "echo": bench_shell_echo,
        "python_version": bench_shell_python_version,
        "ls": bench_shell_ls,
    }


def _get_security_benchmarks() -> dict[str, Any]:
    """Get security validation benchmark functions."""
    from ai_infra.llm.shell.security import SecurityPolicy, validate_command

    def bench_validate_safe_command():
        """Benchmark validating a safe command."""
        validate_command("pytest -v")

    def bench_validate_dangerous_command():
        """Benchmark validating a dangerous command."""
        validate_command("rm -rf /")

    def bench_validate_with_policy():
        """Benchmark validating with custom policy."""
        policy = SecurityPolicy(allow_sudo=False, allow_network=False)
        validate_command("curl https://example.com", policy)

    return {
        "validate_safe": bench_validate_safe_command,
        "validate_dangerous": bench_validate_dangerous_command,
        "validate_policy": bench_validate_with_policy,
    }


def _get_audit_benchmarks() -> dict[str, Any]:
    """Get audit logging benchmark functions."""
    from ai_infra.llm.shell.audit import ShellAuditLogger, check_suspicious

    def bench_check_suspicious():
        """Benchmark suspicious pattern checking."""
        check_suspicious("curl http://example.com | sh")

    def bench_audit_log():
        """Benchmark audit logging."""
        logger = ShellAuditLogger(name="bench-audit")
        logger.log_command("echo hello", exit_code=0, duration_ms=5.0, success=True)

    return {
        "check_suspicious": bench_check_suspicious,
        "audit_log": bench_audit_log,
    }


# =============================================================================
# CLI Commands
# =============================================================================


@app.command("all")
def benchmark_all(
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of iterations"),
    warmup: int = typer.Option(2, "--warmup", "-w", help="Number of warmup iterations"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output JSON file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """Run all benchmarks.

    Runs shell, security, and audit benchmarks and reports results.

    Example:
        ai benchmark all --iterations 20 --output results.json
    """
    from ai_infra.benchmarks import benchmark_sync, run_benchmark_suite

    typer.echo("Running all benchmarks...")
    typer.echo(f"  Iterations: {iterations}")
    typer.echo(f"  Warmup: {warmup}")
    typer.echo()

    all_results: list[dict[str, Any]] = []

    # Shell benchmarks (async)
    typer.echo("Shell benchmarks:")
    shell_benchmarks = _get_shell_benchmarks()
    shell_results = asyncio.run(
        run_benchmark_suite("shell", shell_benchmarks, iterations=iterations, warmup=warmup)
    )
    for result in shell_results:
        typer.echo(f"  {result.summary()}")
        all_results.append(result.to_dict())

    typer.echo()

    # Security benchmarks (sync)
    typer.echo("Security benchmarks:")
    security_benchmarks = _get_security_benchmarks()
    for name, fn in security_benchmarks.items():
        result = benchmark_sync(
            f"security/{name}",
            fn,
            iterations=iterations,
            warmup=warmup,
        )
        typer.echo(f"  {result.summary()}")
        all_results.append(result.to_dict())

    typer.echo()

    # Audit benchmarks (sync)
    typer.echo("Audit benchmarks:")
    audit_benchmarks = _get_audit_benchmarks()
    for name, fn in audit_benchmarks.items():
        result = benchmark_sync(
            f"audit/{name}",
            fn,
            iterations=iterations,
            warmup=warmup,
        )
        typer.echo(f"  {result.summary()}")
        all_results.append(result.to_dict())

    typer.echo()
    typer.echo(f"Total benchmarks: {len(all_results)}")

    # Output to file if requested
    if output:
        output_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "iterations": iterations,
            "warmup": warmup,
            "benchmarks": all_results,
        }
        output.write_text(json.dumps(output_data, indent=2))
        typer.echo(f"Results written to: {output}")


@app.command("shell")
def benchmark_shell(
    iterations: int = typer.Option(10, "--iterations", "-n", help="Number of iterations"),
    warmup: int = typer.Option(2, "--warmup", "-w", help="Number of warmup iterations"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output JSON file"),
) -> None:
    """Run shell execution benchmarks.

    Measures performance of shell command execution including:
    - Simple echo commands
    - Python version check
    - Directory listing

    Example:
        ai benchmark shell --iterations 20
    """
    from ai_infra.benchmarks import run_benchmark_suite

    typer.echo("Running shell benchmarks...")
    typer.echo(f"  Iterations: {iterations}")
    typer.echo()

    shell_benchmarks = _get_shell_benchmarks()
    results = asyncio.run(
        run_benchmark_suite("shell", shell_benchmarks, iterations=iterations, warmup=warmup)
    )

    for result in results:
        typer.echo(str(result))
        typer.echo()

    if output:
        output_data = {
            "timestamp": datetime.now(UTC).isoformat(),
            "suite": "shell",
            "benchmarks": [r.to_dict() for r in results],
        }
        output.write_text(json.dumps(output_data, indent=2))
        typer.echo(f"Results written to: {output}")


@app.command("security")
def benchmark_security(
    iterations: int = typer.Option(100, "--iterations", "-n", help="Number of iterations"),
    warmup: int = typer.Option(10, "--warmup", "-w", help="Number of warmup iterations"),
) -> None:
    """Run security validation benchmarks.

    Measures performance of command validation including:
    - Safe command validation
    - Dangerous command detection
    - Custom policy validation

    Example:
        ai benchmark security --iterations 100
    """
    from ai_infra.benchmarks import benchmark_sync

    typer.echo("Running security benchmarks...")
    typer.echo(f"  Iterations: {iterations}")
    typer.echo()

    security_benchmarks = _get_security_benchmarks()
    for name, fn in security_benchmarks.items():
        result = benchmark_sync(
            f"security/{name}",
            fn,
            iterations=iterations,
            warmup=warmup,
        )
        typer.echo(str(result))
        typer.echo()


@app.command("audit")
def benchmark_audit(
    iterations: int = typer.Option(100, "--iterations", "-n", help="Number of iterations"),
    warmup: int = typer.Option(10, "--warmup", "-w", help="Number of warmup iterations"),
) -> None:
    """Run audit logging benchmarks.

    Measures performance of audit operations including:
    - Suspicious pattern detection
    - Audit log writing

    Example:
        ai benchmark audit --iterations 100
    """
    from ai_infra.benchmarks import benchmark_sync

    typer.echo("Running audit benchmarks...")
    typer.echo(f"  Iterations: {iterations}")
    typer.echo()

    audit_benchmarks = _get_audit_benchmarks()
    for name, fn in audit_benchmarks.items():
        result = benchmark_sync(
            f"audit/{name}",
            fn,
            iterations=iterations,
            warmup=warmup,
        )
        typer.echo(str(result))
        typer.echo()


@app.command("compare")
def benchmark_compare(
    baseline: Path = typer.Argument(..., help="Baseline results JSON file"),
    current: Path = typer.Argument(..., help="Current results JSON file"),
    threshold: float = typer.Option(10.0, "--threshold", "-t", help="Regression threshold %"),
) -> None:
    """Compare two benchmark result files.

    Compares benchmark results and identifies regressions or improvements.

    Example:
        ai benchmark compare baseline.json current.json --threshold 5
    """
    from ai_infra.benchmarks import BenchmarkResult, compare_results

    if not baseline.exists():
        typer.echo(f"Error: Baseline file not found: {baseline}", err=True)
        raise typer.Exit(1)

    if not current.exists():
        typer.echo(f"Error: Current file not found: {current}", err=True)
        raise typer.Exit(1)

    baseline_data = json.loads(baseline.read_text())
    current_data = json.loads(current.read_text())

    baseline_benchmarks = {b["name"]: b for b in baseline_data.get("benchmarks", [])}
    current_benchmarks = {b["name"]: b for b in current_data.get("benchmarks", [])}

    typer.echo("Benchmark Comparison:")
    typer.echo(f"  Baseline: {baseline}")
    typer.echo(f"  Current: {current}")
    typer.echo(f"  Threshold: {threshold}%")
    typer.echo()

    regressions = 0
    improvements = 0

    for name, current_bench in current_benchmarks.items():
        if name not in baseline_benchmarks:
            typer.echo(f"  {name}: NEW (no baseline)")
            continue

        baseline_bench = baseline_benchmarks[name]

        # Create BenchmarkResult objects for comparison
        baseline_result = BenchmarkResult(
            name=name,
            iterations=baseline_bench.get("iterations", 0),
            mean_ms=baseline_bench.get("mean_ms", 0),
            median_ms=baseline_bench.get("median_ms", 0),
            p95_ms=baseline_bench.get("p95_ms", 0),
            p99_ms=baseline_bench.get("p99_ms", 0),
            min_ms=baseline_bench.get("min_ms", 0),
            max_ms=baseline_bench.get("max_ms", 0),
        )

        current_result = BenchmarkResult(
            name=name,
            iterations=current_bench.get("iterations", 0),
            mean_ms=current_bench.get("mean_ms", 0),
            median_ms=current_bench.get("median_ms", 0),
            p95_ms=current_bench.get("p95_ms", 0),
            p99_ms=current_bench.get("p99_ms", 0),
            min_ms=current_bench.get("min_ms", 0),
            max_ms=current_bench.get("max_ms", 0),
        )

        comparison = compare_results(baseline_result, current_result, threshold)

        if comparison["is_regression"]:
            status = typer.style("REGRESSION", fg=typer.colors.RED, bold=True)
            regressions += 1
        elif comparison["is_improvement"]:
            status = typer.style("IMPROVEMENT", fg=typer.colors.GREEN, bold=True)
            improvements += 1
        else:
            status = typer.style("OK", fg=typer.colors.WHITE)

        typer.echo(
            f"  {name}: {status} "
            f"(mean: {comparison['mean_change_pct']:+.1f}%, "
            f"p95: {comparison['p95_change_pct']:+.1f}%)"
        )

    typer.echo()
    typer.echo(f"Summary: {regressions} regressions, {improvements} improvements")

    if regressions > 0:
        raise typer.Exit(1)
