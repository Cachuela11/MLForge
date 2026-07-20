from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from config import settings
from agent_harness import AgentRunSpec, default_harness
from codex_runtime import CodexCliRuntime


async def agent(
    system_prompt: str,
    user_text: str,
    *,
    cwd: Path,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    node_name: str = "agent",
    run_id: str = "",
    attempt: int = 1,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Run one workflow node as an agent call.

    Stages call this function instead of talking to Codex directly.

    In mock mode, it returns deterministic fake text so the pipeline can be
    tested cheaply.

    In codex mode, it creates a CodexCliRuntime and delegates this single node to
    `codex exec`. The stage receives only the final text result and decides where
    to save it.
    """

    result = await default_harness.run(
        AgentRunSpec(
            system_prompt=system_prompt,
            user_text=user_text,
            workspace=cwd,
            node_name=node_name,
            run_id=run_id,
            attempt=attempt,
            metadata=metadata or {},
        ),
        on_event=on_event,
    )
    return result.final_text


def codex_status() -> dict[str, str | bool]:
    codex = CodexCliRuntime(
        codex_bin=settings.codex_bin,
        model=settings.codex_model,
        reasoning_effort=settings.codex_reasoning_effort,
        verbosity=settings.codex_verbosity,
        sandbox=settings.codex_sandbox,
        timeout=settings.codex_timeout,
        inherit_proxy=settings.codex_inherit_proxy,
        sandbox_provider=settings.codex_sandbox_provider,
        docker_image=settings.codex_docker_image,
        docker_bin=settings.codex_docker_bin,
        docker_codex_bin=settings.codex_docker_codex_bin,
        docker_gpus=settings.codex_docker_gpus,
    )
    status = codex.status()
    status["runtime"] = settings.runtime
    return status
