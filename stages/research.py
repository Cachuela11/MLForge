from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_runtime import agent
from agent_harness import default_harness
from config import settings
from db import ResearchDB
from prompts import (
    DECOMPOSE_SYSTEM,
    EVALUATE_SYSTEM,
    EXECUTE_SYSTEM,
    STRATEGY_SYSTEM,
    VERIFY_SYSTEM,
)
from stage import Stage
from utils import parse_json_fenced


class ResearchGraphState(TypedDict, total=False):
    strategy: str
    plan: list[dict[str, Any]]
    completed: list[dict[str, Any]]
    evaluation: dict[str, Any]
    summary: str


class TaskGraphState(TypedDict, total=False):
    task: dict[str, Any]
    completed: dict[str, dict[str, Any]]
    workspace: str
    attempt: int
    max_attempts: int
    output: str
    previous_output: str
    retry_review: str
    verification: dict[str, Any]


class ResearchStage(Stage):
    def __init__(self, db: ResearchDB) -> None:
        super().__init__("research")
        self.db = db

    async def execute(self) -> str:
        self.db.artifacts_dir()

        graph = StateGraph(ResearchGraphState)
        graph.add_node("strategy", self._strategy_node)
        graph.add_node("decompose", self._decompose_node)
        graph.add_node("execute_verify", self._execute_verify_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("summarize", self._summarize_node)
        graph.add_edge(START, "strategy")
        graph.add_edge("strategy", "decompose")
        graph.add_edge("decompose", "execute_verify")
        graph.add_edge("execute_verify", "evaluate")
        graph.add_edge("evaluate", "summarize")
        graph.add_edge("summarize", END)
        result = await graph.compile().ainvoke({})
        return str(result.get("summary", ""))

    async def _strategy_node(self, state: ResearchGraphState) -> ResearchGraphState:
        print("[research] strategy")
        self.emit(phase="strategy", status="running")
        strategy = await self._strategy()
        self.emit(phase="strategy", status="completed")
        return {"strategy": strategy}

    async def _decompose_node(self, state: ResearchGraphState) -> ResearchGraphState:
        strategy = str(state.get("strategy", ""))
        print("[research] decompose")
        self.emit(phase="decompose", status="running")
        plan = await self._decompose(strategy)
        self.emit(phase="decompose", status="completed")
        return {"plan": plan}

    async def _execute_verify_node(self, state: ResearchGraphState) -> ResearchGraphState:
        plan = state.get("plan", [])
        print("[research] execute + verify")
        self.emit(phase="execute", status="running")
        completed = await self._execute_and_verify(plan)
        self.emit(phase="execute", status="completed")
        return {"completed": completed, "plan": plan}

    async def _evaluate_node(self, state: ResearchGraphState) -> ResearchGraphState:
        strategy = str(state.get("strategy", ""))
        plan = state.get("plan", [])
        completed = state.get("completed", [])
        print("[research] evaluate")
        self.emit(phase="evaluate", status="running")
        evaluation = await self._evaluate(strategy, plan, completed)
        self.emit(phase="evaluate", status="completed")
        return {"evaluation": evaluation}

    async def _summarize_node(self, state: ResearchGraphState) -> ResearchGraphState:
        strategy = str(state.get("strategy", ""))
        plan = state.get("plan", [])
        completed = state.get("completed", [])
        evaluation = state.get("evaluation", {})
        summary = self._build_results_summary(strategy, plan, completed, evaluation)
        self.db.save_results_summary(summary["data"], summary["markdown"])
        return {"summary": summary["markdown"]}

    async def _strategy(self) -> str:
        existing = self.db.get_strategy()
        if existing:
            return existing

        strategy = await agent(
            STRATEGY_SYSTEM,
            self._build_strategy_user(),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("strategy"),
            node_name="research.strategy",
        )
        self.db.save_strategy(strategy)
        return strategy

    async def _decompose(self, strategy: str) -> list[dict[str, Any]]:
        existing = self.db.get_plan()
        if existing:
            return existing

        response = await agent(
            DECOMPOSE_SYSTEM,
            self._build_decompose_user(strategy),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("decompose"),
            node_name="research.decompose",
        )
        data = parse_json_fenced(response, default={})
        tasks = self._normalize_tasks(data.get("tasks", []))
        if not tasks:
            tasks = self._fallback_plan()

        tree = {
            "id": "0",
            "description": "Research stage root",
            "children": tasks,
        }
        self.db.save_plan_tree(tree)
        self.db.save_plan(tasks)
        return tasks

    async def _execute_and_verify(self, tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        completed: dict[str, dict[str, Any]] = {}
        semaphore = asyncio.Semaphore(max(1, settings.api_concurrency))
        batch_index = 0
        while True:
            pending = [
                task for task in tasks
                if task["id"] not in completed and task.get("status") != "replaced"
            ]
            if not pending:
                break

            had_recompose = False
            batches = self._topological_batches(pending, precompleted=set(completed.keys()))
            for batch in batches:
                batch_index += 1
                self.emit(
                    phase="execute",
                    status="batch_running",
                    batch=batch_index,
                    task_ids=[task["id"] for task in batch],
                )
                dependency_snapshot = dict(completed)
                results = await asyncio.gather(
                    *[
                        self._execute_one_task(task, dependency_snapshot, semaphore)
                        for task in batch
                    ],
                    return_exceptions=True,
                )

                for task, result in zip(batch, results):
                    task_id = task["id"]
                    if isinstance(result, Exception):
                        task["status"] = "failed"
                        self.db.save_plan(tasks)
                        self.emit(
                            phase="execute",
                            status="failed",
                            task_id=task_id,
                            error=str(result),
                        )
                        raise result

                    verification = result.get("verification", {})
                    if verification.get("redecompose"):
                        task["status"] = "replaced"
                        self.emit(
                            phase="decompose",
                            status="redecomposing",
                            task_id=task_id,
                            review=verification.get("review", ""),
                        )
                        await self._redecompose_task(
                            task=task,
                            result=str(result.get("output", "")),
                            review=str(verification.get("review", "")),
                            tasks=tasks,
                        )
                        had_recompose = True
                        continue

                    passed = bool(verification.get("pass"))
                    task["status"] = "completed" if passed else "failed"
                    if not passed:
                        self.db.save_plan(tasks)
                        raise RuntimeError(
                            f"Task {task_id} failed verification after all attempts: "
                            f"{verification.get('review', '')}"
                        )
                    completed[task_id] = result
                    task["summary"] = result.get("summary", "")
                    self.emit(phase="execute", status="completed", task_id=task_id)

                self.db.save_plan(tasks)
                self.emit(
                    phase="execute",
                    status="batch_completed",
                    batch=batch_index,
                    task_ids=[task["id"] for task in batch],
                )
                if had_recompose:
                    break

            if not had_recompose and not any(
                task["id"] not in completed and task.get("status") != "replaced"
                for task in tasks
            ):
                break

        return list(completed.values())

    async def _execute_one_task(
        self,
        task: dict[str, Any],
        completed: dict[str, dict[str, Any]],
        semaphore: asyncio.Semaphore,
    ) -> dict[str, Any]:
        async with semaphore:
            task_id = task["id"]
            workspace = self._prepare_task_workspace(task, completed)
            print(f"[research] execute task {task_id}: {task.get('title', '')}")
            self.emit(
                phase="execute",
                status="running",
                task_id=task_id,
                task=task,
                workspace=str(workspace),
            )
            graph = StateGraph(TaskGraphState)
            graph.add_node("execute", self._task_execute_node)
            graph.add_node("verify", self._task_verify_node)
            graph.add_edge(START, "execute")
            graph.add_edge("execute", "verify")
            graph.add_conditional_edges(
                "verify",
                self._route_after_task_verification,
                {"retry": "execute", "done": END},
            )
            result = await graph.compile().ainvoke(
                {
                    "task": task,
                    "completed": completed,
                    "workspace": str(workspace),
                    "attempt": 0,
                    "max_attempts": max(1, settings.task_max_attempts),
                    "output": self.db.get_task_output(task_id),
                    "verification": self.db.get_verification(task_id),
                }
            )
            final_output = str(result.get("output", ""))
            final_verification = result.get("verification", {})
            return {
                "task": task,
                "output": final_output,
                "verification": final_verification,
                "summary": self._extract_summary(final_output),
            }

    async def _task_execute_node(self, state: TaskGraphState) -> TaskGraphState:
        task = state["task"]
        task_id = str(task["id"])
        attempt = int(state.get("attempt", 0)) + 1
        max_attempts = int(state.get("max_attempts", 1))
        workspace = Path(str(state["workspace"]))
        previous_output = str(state.get("output", ""))
        previous_verification = state.get("verification", {})
        retry_review = str(previous_verification.get("review", ""))
        self.emit(
            phase="execute",
            status="running",
            task_id=task_id,
            task=task,
            attempt=attempt,
            max_attempts=max_attempts,
            workspace=str(workspace),
        )
        output = await agent(
            EXECUTE_SYSTEM,
            self._build_workspace_execute_user(
                task,
                state.get("completed", {}),
                workspace,
                attempt=attempt,
                previous_output=previous_output,
                retry_review=retry_review,
            ),
            cwd=workspace,
            on_event=self.agent_output_sink("execute", task_id=task_id),
            node_name="research.execute",
            run_id=f"research.execute.{self.db.safe_id(task_id)}.attempt_{attempt}",
            attempt=attempt,
            metadata={"task_id": task_id},
        )
        self.db.save_task_output(task_id, output)
        self.db.save_text(
            f"tasks/{self.db.safe_id(task_id)}.attempt_{attempt}.md", output
        )
        self._sync_task_artifacts(task_id, workspace)
        return {
            "attempt": attempt,
            "previous_output": previous_output,
            "output": output,
            "retry_review": retry_review,
            "verification": {},
        }

    async def _task_verify_node(self, state: TaskGraphState) -> TaskGraphState:
        task = state["task"]
        task_id = str(task["id"])
        attempt = int(state.get("attempt", 1))
        max_attempts = int(state.get("max_attempts", 1))
        print(f"[research] verify task {task_id} attempt {attempt}")
        self.emit(
            phase="verify",
            status="running",
            task_id=task_id,
            task=task,
            attempt=attempt,
            max_attempts=max_attempts,
        )
        verification = await self._verify(task, str(state.get("output", "")))
        self.db.save_verification(task_id, verification)
        self.db.save_json(
            f"verifications/{self.db.safe_id(task_id)}.attempt_{attempt}.json",
            verification,
        )
        self.emit(
            phase="verify",
            status="completed",
            task_id=task_id,
            verification=verification,
            attempt=attempt,
        )
        return {
            "verification": verification,
            "retry_review": str(verification.get("review", "")),
        }

    def _route_after_task_verification(self, state: TaskGraphState) -> str:
        verification = state.get("verification", {})
        if verification.get("pass") or verification.get("redecompose"):
            return "done"
        attempt = int(state.get("attempt", 1))
        max_attempts = int(state.get("max_attempts", 1))
        if attempt < max_attempts:
            self.emit(
                phase="execute",
                status="retrying",
                task_id=str(state.get("task", {}).get("id", "")),
                attempt=attempt + 1,
                max_attempts=max_attempts,
                review=str(verification.get("review", "")),
            )
            return "retry"
        return "done"

    async def _verify(self, task: dict[str, Any], output: str) -> dict[str, Any]:
        response = await agent(
            VERIFY_SYSTEM,
            self._build_verify_user(task, output),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("verify", task_id=str(task.get("id", ""))),
            node_name="research.verify",
        )
        data = parse_json_fenced(response, default={})
        if "pass" not in data:
            return {
                "pass": False,
                "review": "Verify agent did not return valid JSON.",
                "redecompose": False,
            }
        return {
            "pass": bool(data.get("pass")),
            "review": str(data.get("review", "")),
            "redecompose": bool(data.get("redecompose", False)),
        }

    async def _evaluate(
        self,
        strategy: str,
        plan: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> dict[str, Any]:
        existing = self.db.get_evaluation()
        if existing:
            return existing

        response = await agent(
            EVALUATE_SYSTEM,
            self._build_evaluate_user(strategy, plan, completed),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("evaluate"),
            node_name="research.evaluate",
        )
        data = parse_json_fenced(response, default={})
        evaluation = {
            "feedback": str(data.get("feedback", "")),
            "suggestions": data.get("suggestions", []),
            "ready_for_report": bool(data.get("ready_for_report", True)),
        }
        if not isinstance(evaluation["suggestions"], list):
            evaluation["suggestions"] = [str(evaluation["suggestions"])]
        self.db.save_evaluation(evaluation)
        return evaluation

    def _build_strategy_user(self) -> str:
        return "\n\n".join(
            [
                "# Task",
                self.db.get_task(),
                "# Competition metadata",
                self._format_jsonish(self.db.get_competition_info()),
                "# Calibration",
                self.db.get_calibration(),
                "# Request",
                "请制定 research stage 的技术策略。输出中文。",
            ]
        )

    def _build_decompose_user(self, strategy: str) -> str:
        return "\n\n".join(
            [
                "# Task",
                self.db.get_task(),
                "# Competition metadata",
                self._format_jsonish(self.db.get_competition_info()),
                "# Calibration",
                self.db.get_calibration(),
                "# Strategy",
                strategy,
                "# Request",
                "请拆成原子任务 DAG。只输出 JSON。",
            ]
        )

    def _build_execute_user(
        self,
        task: dict[str, Any],
        completed: dict[str, dict[str, Any]],
    ) -> str:
        dependency_summaries = []
        for dep_id in task.get("dependencies", []):
            dep = completed.get(dep_id)
            if dep:
                dependency_summaries.append(f"- [{dep_id}] {dep.get('summary') or '(no summary)'}")

        artifact = task.get("artifact") or f"artifacts/{task['id']}/result.md"
        return "\n\n".join(
            [
                "# Task brief",
                self.db.get_task(),
                "# Competition metadata",
                self._format_jsonish(self.db.get_competition_info()),
                "# Strategy",
                self.db.get_strategy(),
                "# Current atomic task",
                self._format_jsonish(task),
                "# Dependency summaries",
                "\n".join(dependency_summaries) or "(none)",
                "# Artifact requirement",
                f"请把本任务持久产物写到 `{artifact}`。"
                "如果需要额外文件，也放在同一个 task artifact 目录下。",
            ]
        )

    def _build_workspace_execute_user(
        self,
        task: dict[str, Any],
        completed: dict[str, dict[str, Any]],
        workspace: Path,
        *,
        attempt: int = 1,
        previous_output: str = "",
        retry_review: str = "",
    ) -> str:
        artifact = task.get("artifact") or f"artifacts/{task['id']}/result.md"
        workspace_artifact = self._workspace_artifact_path(task)
        parts = [
            "# Workspace contract",
            "\n".join(
                [
                    f"- Current task workspace: `{workspace}`",
                    f"- Session directory: `{self.db.session_dir}`",
                    f"- Kaggle data directory: `{settings.dataset_dir}`",
                    f"- Attempt: {attempt}/{max(1, settings.task_max_attempts)}",
                    "- Execute this task in the current workspace.",
                    "- Read copied files in this workspace first: `task.md`, `competition.json`, `strategy.md`, `plan_list.json`, `current_task.json`, and `dependencies.md`.",
                    "- Write durable outputs under `./artifacts/` in the current workspace.",
                    f"- KaggleForge will sync `./artifacts/` back to session `artifacts/{self.db.safe_id(task['id'])}/` after this agent call.",
                    f"- The decompose-requested artifact path is `{artifact}`.",
                    f"- In this workspace, prefer writing the primary artifact as `{workspace_artifact}`.",
                    "- Do not rely on conversation memory from other tasks. Use dependency summaries and persisted files only.",
                ]
            ),
        ]
        if attempt > 1 or retry_review:
            parts.extend(
                [
                    "# Previous Output",
                    previous_output.strip() or "(no previous output captured)",
                    "# Review Feedback To Address",
                    retry_review.strip() or "(no review feedback captured)",
                    "# Retry Instructions",
                    "\n".join(
                        [
                            "- This is a targeted retry, not a blind rerun.",
                            "- Fix the specific issues from the review feedback.",
                            "- Reuse existing workspace files and artifacts when they are already correct.",
                            "- Do not fabricate missing metrics or command results; rerun or inspect files as needed.",
                            "- Clearly state what changed in this attempt.",
                        ]
                    ),
                ]
            )
        parts.append(self._build_execute_user(task, completed))
        return "\n\n".join(parts)

    def _workspace_artifact_path(self, task: dict[str, Any]) -> str:
        task_id = str(task["id"])
        artifact = str(task.get("artifact") or f"artifacts/{task_id}/result.md").replace("\\", "/")
        prefix = f"artifacts/{task_id}/"
        safe_prefix = f"artifacts/{self.db.safe_id(task_id)}/"
        if artifact.startswith(prefix):
            return f"artifacts/{artifact[len(prefix):]}"
        if artifact.startswith(safe_prefix):
            return f"artifacts/{artifact[len(safe_prefix):]}"
        if artifact.startswith("artifacts/"):
            return artifact
        return f"artifacts/{artifact}"

    def _prepare_task_workspace(
        self,
        task: dict[str, Any],
        completed: dict[str, dict[str, Any]],
    ) -> Path:
        task_id = task["id"]
        workspace = self.db.task_workspace_dir(task_id)
        context_files: dict[Path, str] = {}
        for name in (
            "source.md",
            "task.md",
            "competition.json",
            "calibration.md",
            "strategy.md",
            "plan_list.json",
            "plan_tree.json",
            "results_summary.md",
            "results_summary.json",
        ):
            src = self.db.session_dir / name
            if src.exists() and src.is_file():
                context_files[src] = name

        current_task = dict(task)
        current_task["workspace"] = str(workspace)
        current_task["session_dir"] = str(self.db.session_dir)
        current_task["dataset_dir"] = str(settings.dataset_dir)

        dependency_lines = []
        for dep_id in task.get("dependencies", []):
            dep = completed.get(dep_id)
            if not dep:
                continue
            dependency_lines.extend(
                [
                    f"## Dependency {dep_id}",
                    "",
                    f"- Summary: {dep.get('summary') or '(no summary)'}",
                    f"- Verification pass: {dep.get('verification', {}).get('pass')}",
                    f"- Task output: `../../tasks/{self.db.safe_id(dep_id)}.md`",
                    "",
                ]
            )
        default_harness.prepare_workspace(
            workspace,
            context_files=context_files,
            generated_files={
                "current_task.json": json.dumps(
                    current_task, indent=2, ensure_ascii=False
                ),
                "dependencies.md": "\n".join(dependency_lines) or "(none)\n",
            },
        )
        return workspace

    def _sync_task_artifacts(self, task_id: str, workspace: Path) -> None:
        safe_id = self.db.safe_id(task_id)
        default_harness.sync_artifacts(
            workspace,
            self.db.task_artifacts_dir(task_id),
            strip_top_level={safe_id, str(task_id)},
        )

    async def _redecompose_task(
        self,
        *,
        task: dict[str, Any],
        result: str,
        review: str,
        tasks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        task_id = str(task["id"])
        existing_ids = {str(item["id"]) for item in tasks}
        user_text = self._build_redecompose_user(task, result, review, tasks)
        response = await agent(
            DECOMPOSE_SYSTEM,
            user_text,
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("decompose", task_id=task_id),
            node_name="research.redecompose",
            metadata={"task_id": task_id},
        )
        data = parse_json_fenced(response, default={})
        raw_children = data.get("tasks", [])
        children = self._normalize_redecomposed_tasks(
            raw_children,
            parent=task,
            existing_ids=existing_ids,
        )
        if not children:
            task["status"] = "failed"
            self.db.save_plan(tasks)
            return []

        terminal_ids = self._terminal_task_ids(children)
        parent_dependencies = list(task.get("dependencies", []))
        insert_at = next(
            (index for index, item in enumerate(tasks) if item["id"] == task_id),
            len(tasks),
        )
        tasks[:] = [item for item in tasks if item["id"] != task_id]
        for offset, child in enumerate(children):
            tasks.insert(insert_at + offset, child)

        child_ids = {child["id"] for child in children}
        for item in tasks:
            if item["id"] in child_ids:
                continue
            deps = list(item.get("dependencies", []))
            if task_id in deps:
                rewritten = []
                for dep in deps:
                    if dep == task_id:
                        rewritten.extend(terminal_ids)
                    else:
                        rewritten.append(dep)
                item["dependencies"] = list(dict.fromkeys(rewritten))

        tree = self.db.get_plan_tree() or {
            "id": "0",
            "description": "Research stage root",
            "children": [],
        }
        self._replace_tree_node(tree, task_id, task, children)
        self.db.save_plan_tree(tree)
        self.db.save_plan(tasks)
        self.emit(
            phase="decompose",
            status="redecomposed",
            task_id=task_id,
            child_ids=[child["id"] for child in children],
        )
        return children

    def _build_redecompose_user(
        self,
        task: dict[str, Any],
        result: str,
        review: str,
        tasks: list[dict[str, Any]],
    ) -> str:
        siblings = [
            {"id": item["id"], "title": item.get("title", ""), "description": item.get("description", "")}
            for item in tasks
            if item["id"] != task["id"]
        ]
        return "\n\n".join(
            [
                "# Task",
                self.db.get_task(),
                "# Competition metadata",
                self._format_jsonish(self.db.get_competition_info()),
                "# Calibration",
                self.db.get_calibration(),
                "# Strategy",
                self.db.get_strategy(),
                "# Failed task to redecompose",
                self._format_jsonish(task),
                "# Execute output",
                result or "(empty)",
                "# Verify review",
                review or "(empty)",
                "# Sibling tasks",
                self._format_jsonish(siblings),
                "# Request",
                "\n".join(
                    [
                        "请只把 failed task 拆成 2 到 4 个更小的原子子任务。",
                        "不要重写整个 plan，不要包含 sibling tasks。",
                        "子任务必须共同替代原 failed task，并尽量复用已有输出中可靠的部分。",
                        "dependencies 只能引用同一组新子任务中的更早 id；原父任务依赖会由 KaggleForge 自动继承。",
                        "只输出 JSON，格式仍为 {\"tasks\": [...]}。",
                    ]
                ),
            ]
        )

    def _normalize_redecomposed_tasks(
        self,
        raw_tasks: Any,
        *,
        parent: dict[str, Any],
        existing_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw_tasks, list):
            return []
        parent_id = str(parent["id"])
        parent_deps = [str(dep) for dep in parent.get("dependencies", [])]
        normalized = []
        local_map: dict[str, str] = {}
        used = set(existing_ids)
        used.discard(parent_id)

        for index, raw in enumerate(raw_tasks, start=1):
            if not isinstance(raw, dict):
                continue
            raw_id = str(raw.get("id") or index).strip() or str(index)
            child_id = raw_id if raw_id.startswith(f"{parent_id}_") else f"{parent_id}_{raw_id}"
            child_id = self.db.safe_id(child_id)
            while child_id in used:
                child_id = f"{child_id}_{index}"
            used.add(child_id)
            local_map[raw_id] = child_id

            raw_deps = raw.get("dependencies", [])
            if not isinstance(raw_deps, list):
                raw_deps = []
            deps = []
            for dep in raw_deps:
                dep_key = str(dep)
                if dep_key in local_map:
                    deps.append(local_map[dep_key])
            if not deps:
                deps = list(parent_deps)

            artifact = str(raw.get("artifact") or f"artifacts/{child_id}/result.md")
            normalized.append(
                {
                    "id": child_id,
                    "title": str(raw.get("title") or f"{parent.get('title', 'Task')} / {index}"),
                    "description": str(raw.get("description") or raw.get("title") or ""),
                    "dependencies": list(dict.fromkeys(deps)),
                    "artifact": self._rewrite_child_artifact(artifact, child_id),
                    "status": "pending",
                    "parent": parent_id,
                }
            )
        return normalized

    def _rewrite_child_artifact(self, artifact: str, child_id: str) -> str:
        artifact = artifact.replace("\\", "/")
        if artifact.startswith("artifacts/"):
            parts = artifact.split("/")
            if len(parts) >= 3:
                return "/".join(["artifacts", child_id, *parts[2:]])
            return f"artifacts/{child_id}/result.md"
        return f"artifacts/{child_id}/{Path(artifact).name or 'result.md'}"

    @staticmethod
    def _terminal_task_ids(tasks: list[dict[str, Any]]) -> list[str]:
        depended = {
            str(dep)
            for task in tasks
            for dep in task.get("dependencies", [])
        }
        terminals = [task["id"] for task in tasks if task["id"] not in depended]
        return terminals or [tasks[-1]["id"]]

    def _replace_tree_node(
        self,
        node: dict[str, Any],
        task_id: str,
        original: dict[str, Any],
        children: list[dict[str, Any]],
    ) -> bool:
        tree_children = node.setdefault("children", [])
        for index, child in enumerate(list(tree_children)):
            if child.get("id") == task_id:
                tree_children[index] = {
                    "id": task_id,
                    "title": original.get("title", ""),
                    "description": original.get("description", ""),
                    "status": "replaced",
                    "children": children,
                }
                return True
            if self._replace_tree_node(child, task_id, original, children):
                return True
        if node.get("id") == "0":
            tree_children.append({
                "id": task_id,
                "title": original.get("title", ""),
                "description": original.get("description", ""),
                "status": "replaced",
                "children": children,
            })
            return True
        return False

    @staticmethod
    def _build_verify_user(task: dict[str, Any], output: str) -> str:
        return "\n\n".join(
            [
                "# Atomic task",
                ResearchStage._format_jsonish(task),
                "# Execute output",
                output,
                "# Request",
                "请判断该任务是否通过。只输出 JSON。",
            ]
        )

    def _build_evaluate_user(
        self,
        strategy: str,
        plan: list[dict[str, Any]],
        completed: list[dict[str, Any]],
    ) -> str:
        task_lines = []
        for item in completed:
            task = item["task"]
            verification = item["verification"]
            task_lines.append(
                f"- [{task['id']}] {task.get('title', '')}: "
                f"{item.get('summary') or '(no summary)'} | pass={verification.get('pass')}"
            )
        return "\n\n".join(
            [
                "# Task",
                self.db.get_task(),
                "# Strategy",
                strategy,
                "# Plan",
                self._format_jsonish(plan),
                "# Completed task summaries",
                "\n".join(task_lines),
                "# Artifacts",
                self._format_jsonish(self.db.list_artifacts()),
                "# Request",
                "请评估本轮 research 结果是否足够进入 report stage。只输出 JSON。",
            ]
        )

    @staticmethod
    def _normalize_tasks(raw_tasks: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_tasks, list):
            return []

        normalized = []
        seen = set()
        for index, raw in enumerate(raw_tasks, start=1):
            if not isinstance(raw, dict):
                continue
            task_id = str(raw.get("id") or index)
            if task_id in seen:
                continue
            seen.add(task_id)
            dependencies = raw.get("dependencies", [])
            if not isinstance(dependencies, list):
                dependencies = []
            dependencies = [str(dep) for dep in dependencies if str(dep) in seen]
            artifact = str(raw.get("artifact") or f"artifacts/{task_id}/result.md")
            normalized.append(
                {
                    "id": task_id,
                    "title": str(raw.get("title") or f"Task {task_id}"),
                    "description": str(raw.get("description") or raw.get("title") or ""),
                    "dependencies": dependencies,
                    "artifact": artifact,
                    "status": "pending",
                }
            )
        return normalized

    @staticmethod
    def _topological_order(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        remaining = {task["id"]: task for task in tasks}
        completed: set[str] = set()
        ordered: list[dict[str, Any]] = []
        while remaining:
            ready = [
                task_id for task_id, task in remaining.items()
                if all(dep in completed for dep in task.get("dependencies", []))
            ]
            if not ready:
                ready = list(remaining.keys())
            for task_id in ready:
                ordered.append(remaining.pop(task_id))
                completed.add(task_id)
        return ordered

    @staticmethod
    def _topological_batches(
        tasks: list[dict[str, Any]],
        precompleted: set[str] | None = None,
    ) -> list[list[dict[str, Any]]]:
        remaining = {task["id"]: task for task in tasks}
        completed: set[str] = set(precompleted or set())
        batches: list[list[dict[str, Any]]] = []
        while remaining:
            ready = [
                task_id for task_id, task in remaining.items()
                if all(dep in completed for dep in task.get("dependencies", []))
            ]
            if not ready:
                ready = list(remaining.keys())
            batch = [remaining.pop(task_id) for task_id in ready]
            batches.append(batch)
            completed.update(ready)
        return batches

    @staticmethod
    def _extract_summary(output: str) -> str:
        for line in reversed(output.strip().splitlines()):
            stripped = line.strip()
            if stripped.upper().startswith("SUMMARY:"):
                return stripped[len("SUMMARY:"):].strip()
        return ""

    @staticmethod
    def _format_jsonish(data: Any) -> str:
        import json

        return json.dumps(data, ensure_ascii=False, indent=2)

    @staticmethod
    def _fallback_plan() -> list[dict[str, Any]]:
        return [
            {
                "id": "1",
                "title": "数据概览",
                "description": "读取 competition.json 指定的数据目录，检查 train/test/sample_submission 的字段、行数、缺失值和提交格式，输出数据概览报告。",
                "dependencies": [],
                "artifact": "artifacts/1/profile.md",
                "status": "pending",
            },
            {
                "id": "2",
                "title": "Baseline 脚本",
                "description": "基于数据概览实现一个可复现 baseline 训练脚本，包含固定随机种子、简单预处理、本地验证和 submission 生成逻辑。",
                "dependencies": ["1"],
                "artifact": "artifacts/2/baseline.py",
                "status": "pending",
            },
            {
                "id": "3",
                "title": "验证结果总结",
                "description": "运行或检查 baseline 输出，记录验证指标、主要发现、潜在问题和下一步改进方向。",
                "dependencies": ["2"],
                "artifact": "artifacts/3/validation_report.md",
                "status": "pending",
            },
        ]

    def _build_results_summary(
        self,
        strategy: str,
        plan: list[dict[str, Any]],
        completed: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        artifacts = self.db.list_artifacts()
        data = {
            "strategy_file": "strategy.md",
            "plan_file": "plan_list.json",
            "task_count": len(plan),
            "completed_count": len(completed),
            "artifacts": artifacts,
            "evaluation": evaluation,
        }

        lines = [
            "# Research Results Summary",
            "",
            "## Strategy",
            "",
            strategy.strip(),
            "",
            "## Completed Tasks",
            "",
        ]
        for item in completed:
            task = item["task"]
            verification = item["verification"]
            lines.extend(
                [
                    f"### [{task['id']}] {task.get('title', '')}",
                    "",
                    f"- Artifact: `{task.get('artifact', '')}`",
                    f"- Verified: `{verification.get('pass')}`",
                    f"- Summary: {item.get('summary') or '(no summary)'}",
                    "",
                ]
            )
        lines.extend(
            [
                "## Artifacts",
                "",
                *[f"- `{item['path']}` ({item['size_bytes']} bytes)" for item in artifacts],
                "",
                "## Evaluation",
                "",
                self._format_jsonish(evaluation),
                "",
            ]
        )
        return {"data": data, "markdown": "\n".join(lines)}
