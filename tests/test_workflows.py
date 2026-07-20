from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from agent_harness import AgentHarness, AgentRunSpec
from db import ResearchDB
from stages.intake import IntakeStage
from stages.report import ReportStage
from stages.research import ResearchStage


def make_db(root: Path) -> ResearchDB:
    db = ResearchDB(str(root))
    db.create_session("https://www.kaggle.com/competitions/test-competition")
    db.save_source("https://www.kaggle.com/competitions/test-competition")
    return db


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_harness_writes_auditable_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            AgentHarness().prepare_workspace(
                workspace, generated_files={"context/task.md": "task"}
            )
            (workspace / "artifacts/result.txt").write_text("result", encoding="utf-8")
            result = await AgentHarness().run(
                AgentRunSpec(
                    system_prompt="system",
                    user_text="input",
                    workspace=workspace,
                    run_id="test-run",
                    node_name="test.node",
                )
            )
            self.assertEqual(result.status, "completed")
            self.assertTrue((workspace / ".harness/runs/test-run/spec.json").is_file())
            self.assertTrue((workspace / ".harness/runs/test-run/result.json").is_file())
            destination = workspace / "synced"
            manifest = AgentHarness().sync_artifacts(workspace, destination)
            self.assertEqual(manifest[0]["path"], "result.txt")
            self.assertTrue((destination / "result.txt").is_file())

    async def test_intake_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db = make_db(Path(temp))
            stage = IntakeStage(db)

            async def fake_load(this: IntakeStage) -> str:
                db.save_task("test task")
                db.save_competition_info({"id": "test-competition"})
                return "test task"

            stage._load_kaggle_task = types.MethodType(fake_load, stage)
            output = await stage.execute()
            self.assertTrue(output)
            self.assertTrue(db.get_calibration())

    async def test_research_stage_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db = make_db(Path(temp))
            stage = ResearchStage(db)
            plan = [{"id": "1", "title": "test", "dependencies": [], "status": "pending"}]

            async def strategy(this: ResearchStage) -> str:
                return "strategy"

            async def decompose(this: ResearchStage, value: str):
                self.assertEqual(value, "strategy")
                return plan

            async def execute_verify(this: ResearchStage, value):
                self.assertIs(value, plan)
                return [{"task": plan[0], "output": "done", "verification": {"pass": True}, "summary": "done"}]

            async def evaluate(this: ResearchStage, strategy_value, plan_value, completed):
                return {"feedback": "ok", "suggestions": [], "ready_for_report": True}

            stage._strategy = types.MethodType(strategy, stage)
            stage._decompose = types.MethodType(decompose, stage)
            stage._execute_and_verify = types.MethodType(execute_verify, stage)
            stage._evaluate = types.MethodType(evaluate, stage)
            output = await stage.execute()
            self.assertIn("Research Results Summary", output)

    async def test_report_graph(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            db = make_db(Path(temp))
            db.save_task("test task")
            db.save_strategy("test strategy")
            db.save_plan([])
            output = await ReportStage(db).execute()
            self.assertTrue(output)
            self.assertTrue(db.read_text("paper_polished.md"))

    def test_task_retry_route(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            stage = ResearchStage(make_db(Path(temp)))
            retry = stage._route_after_task_verification(
                {"task": {"id": "1"}, "attempt": 1, "max_attempts": 2, "verification": {"pass": False}}
            )
            done = stage._route_after_task_verification(
                {"task": {"id": "1"}, "attempt": 2, "max_attempts": 2, "verification": {"pass": False}}
            )
            self.assertEqual(retry, "retry")
            self.assertEqual(done, "done")


if __name__ == "__main__":
    unittest.main()
