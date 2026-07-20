from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from codex_runtime import CodexCliRuntime


EventSink = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class AgentRunSpec:
    system_prompt: str
    user_text: str
    workspace: Path
    run_id: str = ""
    node_name: str = "agent"
    attempt: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentRunResult:
    run_id: str
    status: str
    runtime: str
    final_text: str
    started_at: float
    finished_at: float
    error: str = ""


class AgentHarness:
    """Runs one agent node with a stable contract around runtime providers.

    Workflow routing stays outside this class. The harness owns provider
    selection, normalized events, and an auditable record for a single run.
    """

    async def run(
        self,
        spec: AgentRunSpec,
        *,
        on_event: EventSink | None = None,
    ) -> AgentRunResult:
        run_id = spec.run_id or self._new_run_id(spec.node_name)
        started_at = time.time()
        self._emit(on_event, {"type": "agent.run.started", "run_id": run_id, "node": spec.node_name})
        try:
            final_text = await self._run_provider(spec, on_event=on_event)
            result = AgentRunResult(
                run_id=run_id,
                status="completed",
                runtime=settings.runtime.lower(),
                final_text=final_text,
                started_at=started_at,
                finished_at=time.time(),
            )
        except BaseException as exc:
            result = AgentRunResult(
                run_id=run_id,
                status="failed",
                runtime=settings.runtime.lower(),
                final_text="",
                started_at=started_at,
                finished_at=time.time(),
                error=str(exc) or type(exc).__name__,
            )
            self._save_record(spec, result)
            self._emit(on_event, {"type": "agent.run.failed", "run_id": run_id, "error": result.error})
            raise

        self._save_record(spec, result)
        self._emit(on_event, {"type": "agent.run.completed", "run_id": run_id, "node": spec.node_name})
        return result

    def prepare_workspace(
        self,
        workspace: Path,
        *,
        context_files: dict[Path, str] | None = None,
        generated_files: dict[str, str] | None = None,
    ) -> Path:
        """Materialize declared context without making workflow decisions."""
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "artifacts").mkdir(parents=True, exist_ok=True)
        for source, relative_target in (context_files or {}).items():
            if not source.is_file():
                continue
            target = self._safe_workspace_path(workspace, relative_target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for relative_target, content in (generated_files or {}).items():
            target = self._safe_workspace_path(workspace, relative_target)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        return workspace

    def sync_artifacts(
        self,
        workspace: Path,
        destination: Path,
        *,
        strip_top_level: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Copy durable artifacts and return a provider-neutral manifest."""
        source = workspace / "artifacts"
        if not source.exists():
            return []
        destination.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            if relative.parts and relative.parts[0] in (strip_top_level or set()):
                relative = (
                    Path(*relative.parts[1:])
                    if len(relative.parts) > 1
                    else Path(path.name)
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            manifest.append({"path": relative.as_posix(), "size_bytes": target.stat().st_size})
        return manifest

    async def _run_provider(self, spec: AgentRunSpec, *, on_event: EventSink | None) -> str:
        if settings.runtime.lower() == "codex":
            runtime = CodexCliRuntime(
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
            return await runtime.run(
                instruction=spec.system_prompt,
                user_text=spec.user_text,
                cwd=spec.workspace,
                on_event=on_event,
            )
        return (
            f"{spec.system_prompt.strip()}\n\n"
            "Generated from input:\n"
            f"{spec.user_text.strip()}\n"
        )

    @staticmethod
    def _new_run_id(node_name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in node_name)
        return f"{safe}-{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _safe_workspace_path(workspace: Path, relative_path: str) -> Path:
        root = workspace.resolve()
        target = (root / relative_path).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Workspace path escapes root: {relative_path}") from exc
        return target

    @staticmethod
    def _emit(sink: EventSink | None, event: dict[str, Any]) -> None:
        if sink:
            sink(event)

    @staticmethod
    def _save_record(spec: AgentRunSpec, result: AgentRunResult) -> None:
        run_dir = spec.workspace / ".harness" / "runs" / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        public_spec = {
            "run_id": result.run_id,
            "node_name": spec.node_name,
            "attempt": spec.attempt,
            "workspace": str(spec.workspace),
            "metadata": spec.metadata,
        }
        (run_dir / "spec.json").write_text(
            json.dumps(public_spec, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_dir / "result.json").write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if result.final_text:
            (run_dir / "output.txt").write_text(result.final_text, encoding="utf-8")


default_harness = AgentHarness()
