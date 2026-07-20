from __future__ import annotations

import asyncio
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from agent_runtime import agent
from config import settings
from db import ResearchDB
from kaggle_integration import (
    build_kaggle_task,
    extract_competition_id,
    fetch_competition,
)
from prompts import CALIBRATE_SYSTEM
from stage import Stage


class IntakeGraphState(TypedDict, total=False):
    task: str
    calibration: str


class IntakeStage(Stage):
    def __init__(self, db: ResearchDB) -> None:
        super().__init__("intake")
        self.db = db

    async def execute(self) -> str:
        graph = StateGraph(IntakeGraphState)
        graph.add_node("load_kaggle", self._load_kaggle_node)
        graph.add_node("calibrate", self._calibrate_node)
        graph.add_edge(START, "load_kaggle")
        graph.add_edge("load_kaggle", "calibrate")
        graph.add_edge("calibrate", END)
        result = await graph.compile().ainvoke({})
        return str(result.get("calibration", ""))

    async def _load_kaggle_node(self, state: IntakeGraphState) -> IntakeGraphState:
        self.emit(phase="kaggle", status="running")
        task = await self._load_kaggle_task()
        self.emit(phase="kaggle", status="completed")
        return {"task": task}

    async def _calibrate_node(self, state: IntakeGraphState) -> IntakeGraphState:
        task = str(state.get("task", ""))
        calibration = self.db.get_calibration()

        if not calibration:
            self.emit(phase="calibrate", status="running")
            calibration = await agent(
                CALIBRATE_SYSTEM,
                self._build_calibrate_user(task),
                cwd=self.db.session_dir,
                on_event=self.agent_output_sink("calibrate"),
                node_name="intake.calibrate",
            )
            self.db.save_calibration(calibration)
            self.emit(phase="calibrate", status="completed")
        else:
            self.emit(phase="calibrate", status="completed", cached=True)

        return {"calibration": calibration}

    async def _load_kaggle_task(self) -> str:
        raw_input = self.db.get_source()
        competition_id = extract_competition_id(raw_input)
        if not competition_id:
            raise RuntimeError(
                "IntakeStage 目前只接受 Kaggle competition URL，例如："
                "https://www.kaggle.com/competitions/titanic"
            )

        info = await asyncio.to_thread(
            fetch_competition,
            competition_id,
            settings.dataset_dir,
        )
        self.db.save_competition_info(info)

        task = build_kaggle_task(info)
        self.db.save_task(task)
        return task

    @staticmethod
    def _build_capability_profile() -> str:
        timeout = f"{settings.codex_timeout} 秒" if settings.codex_timeout else "未显式设置"
        model = settings.codex_model or "Codex 默认模型"
        reasoning = settings.codex_reasoning_effort or "Codex 默认推理强度"
        verbosity = settings.codex_verbosity or "Codex 默认输出详细度"
        return "\n".join(
            [
                f"- Runtime: {settings.runtime}",
                f"- Agent executor: {settings.codex_bin} exec",
                f"- Model: {model}",
                f"- Reasoning effort: {reasoning}",
                f"- Verbosity: {verbosity}",
                f"- Sandbox: {settings.codex_sandbox}",
                f"- Sandbox provider: {settings.codex_sandbox_provider}",
                f"- Docker image: {settings.codex_docker_image or '未配置'}",
                f"- Single-agent timeout: {timeout}",
                "- Execution model: one agent node maps to one `codex exec` call.",
                "- Handoff rule: every agent should leave a concrete file artifact for the next node.",
            ]
        )

    def _build_calibrate_user(self, task: str) -> str:
        return (
            "# Kaggle task\n\n"
            f"{task}\n\n"
            "# Runtime capability profile\n\n"
            f"{self._build_capability_profile()}\n\n"
            "# Calibration request\n\n"
            "请定义这个项目里“一次 agent 执行”的原子操作边界。"
            "后续 Strategy/Decompose/Execute/Verify/Evaluate stage 会依赖这个边界来拆任务。"
        )
