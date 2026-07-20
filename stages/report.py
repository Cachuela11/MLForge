from __future__ import annotations

import json
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_runtime import agent
from db import ResearchDB
from stage import Stage
from utils import parse_json_fenced


REPORT_WRITER_SYSTEM = """你是 KaggleForge 的 Report Writer agent。
你的任务是根据 research stage 的真实产物，写一份中文技术总结报告。

写作原则：
- 只使用输入材料中已经存在的事实、指标、文件和结论，不要编造分数、表格、图片或提交结果。
- 如果某个任务因为执行环境问题没有真实完成，要在报告中诚实说明，不要把待补跑内容写成已完成实验。
- 报告应面向一个想复现实验的人，讲清楚任务、数据、策略、执行结果、验证结论、产物位置和后续改进。
- 可以采用论文式结构，但不要为了形式强行写空洞章节。
- 引用 artifacts 文件时，只引用 Artifact Manifest 中真实存在的路径。
- 输出完整 Markdown，不要输出 JSON，不要写额外解释。"""


REPORT_REVIEWER_SYSTEM = """你是 KaggleForge 的 Report Reviewer agent。
你的任务是审查报告初稿是否忠实反映了 research stage 的产物。

重点检查：
1. 数据准确性：报告中的数字、文件、任务状态是否能在输入材料中找到依据。
2. 完整性：是否覆盖 task.md、strategy、plan、task outputs、verifications、evaluation 和 artifacts。
3. 诚实性：是否把失败、未验证、环境限制误写成成功结果。
4. 可复现性：读者是否能根据报告找到关键脚本、结果文件和下一步动作。
5. 结构和可读性：是否像一份清晰的技术总结，而不是流水账。

只输出 JSON：
{
  "issues": [
    {"section": "章节名", "problem": "问题", "suggestion": "修改建议"}
  ],
  "approved": false
}

如果没有实质问题，输出：
{"issues": [], "approved": true}"""


REPORT_POLISH_SYSTEM = """你是 KaggleForge 的 Report Polish agent。
你会收到一份中文技术报告初稿、审稿意见和事实锚点。

你的任务：
- 根据审稿意见修订报告。
- 改善结构、表达和可读性。
- 保留所有事实、指标、文件路径和限制说明，不要新增未经证实的结论。
- 如果审稿意见为空，也要做一次轻量润色，让报告更自然、紧凑。

输出完整 Markdown 正文，不要输出 JSON，不要解释修改过程。"""


class ReportGraphState(TypedDict, total=False):
    context_data: dict[str, Any]
    context_markdown: str
    draft: str
    review: dict[str, Any]
    final: str


class ReportStage(Stage):
    def __init__(self, db: ResearchDB) -> None:
        super().__init__("report")
        self.db = db

    async def execute(self) -> str:
        graph = StateGraph(ReportGraphState)
        graph.add_node("collect", self._collect_node)
        graph.add_node("writer", self._writer_node)
        graph.add_node("reviewer", self._reviewer_node)
        graph.add_node("polish", self._polish_node)
        graph.add_edge(START, "collect")
        graph.add_edge("collect", "writer")
        graph.add_edge("writer", "reviewer")
        graph.add_edge("reviewer", "polish")
        graph.add_edge("polish", END)
        result = await graph.compile().ainvoke({})
        return str(result.get("final", ""))

    async def _collect_node(self, state: ReportGraphState) -> ReportGraphState:
        print("[report] collect context")
        self.emit(phase="collect", status="running")
        context = self._build_report_context()
        self.db.save_json("report_context.json", context["data"])
        self.db.save_text("report_context.md", context["markdown"])
        self.emit(phase="collect", status="completed")
        return {"context_data": context["data"], "context_markdown": context["markdown"]}

    async def _writer_node(self, state: ReportGraphState) -> ReportGraphState:
        print("[report] writer")
        self.emit(phase="writer", status="running")
        draft = await agent(
            REPORT_WRITER_SYSTEM,
            str(state.get("context_markdown", "")),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("writer"),
            node_name="report.writer",
        )
        self.db.save_paper(draft)
        self.emit(phase="writer", status="completed")
        return {"draft": draft}

    async def _reviewer_node(self, state: ReportGraphState) -> ReportGraphState:
        print("[report] reviewer")
        self.emit(phase="reviewer", status="running")
        review_raw = await agent(
            REPORT_REVIEWER_SYSTEM,
            self._build_review_user(
                str(state.get("context_markdown", "")), str(state.get("draft", ""))
            ),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("reviewer"),
            node_name="report.reviewer",
        )
        review = self._normalize_review(parse_json_fenced(review_raw, default={}))
        self.db.save_text("report_review.md", review_raw)
        self.db.save_json("report_review.json", review)
        self.emit(phase="reviewer", status="completed", review=review)
        return {"review": review}

    async def _polish_node(self, state: ReportGraphState) -> ReportGraphState:
        print("[report] polish")
        self.emit(phase="polish", status="running")
        final = await agent(
            REPORT_POLISH_SYSTEM,
            self._build_polish_user(
                str(state.get("context_markdown", "")),
                str(state.get("draft", "")),
                state.get("review", {}),
            ),
            cwd=self.db.session_dir,
            on_event=self.agent_output_sink("polish"),
            node_name="report.polish",
        )
        final = final.rstrip() + "\n\n" + self._build_metadata_appendix(
            state.get("context_data", {})
        )
        self.db.save_paper_polished(final)
        self.emit(phase="polish", status="completed")
        return {"final": final}

    def _build_report_context(self) -> dict[str, Any]:
        tasks = self.db.get_plan()
        task_records = []
        for task in tasks:
            task_id = str(task.get("id", ""))
            task_records.append({
                "task": task,
                "output": self.db.get_task_output(task_id),
                "verification": self.db.get_verification(task_id),
            })

        data = {
            "session_id": self.db.research_id,
            "source": self.db.get_source(),
            "task": self.db.get_task(),
            "competition": self.db.get_competition_info(),
            "calibration": self.db.get_calibration(),
            "strategy": self.db.get_strategy(),
            "plan": tasks,
            "task_records": task_records,
            "evaluation": self.db.get_evaluation(),
            "results_summary": self.db.read_text("results_summary.md"),
            "results_summary_json": self.db.read_json("results_summary.json", {}),
            "artifacts": self.db.list_artifacts(),
        }
        return {"data": data, "markdown": self._render_context_markdown(data)}

    def _render_context_markdown(self, data: dict[str, Any]) -> str:
        lines = [
            "# Report Input Package",
            "",
            "以下内容是最终报告唯一可信事实来源。报告必须以这些材料为准。",
            "",
            "## Session",
            f"- Session ID: `{data.get('session_id')}`",
            f"- Source: {data.get('source')}",
            "",
            "## Kaggle Task",
            data.get("task") or "(missing task.md)",
            "",
            "## Competition Metadata",
            self._format_jsonish(data.get("competition")),
            "",
            "## Calibration",
            data.get("calibration") or "(missing calibration.md)",
            "",
            "## Strategy",
            data.get("strategy") or "(missing strategy.md)",
            "",
            "## Plan",
            self._format_jsonish(data.get("plan")),
            "",
            "## Task Outputs And Verification",
        ]

        records = data.get("task_records", [])
        if not records:
            lines.append("(no task outputs)")
        for record in records:
            task = record.get("task", {})
            task_id = task.get("id", "")
            lines.extend([
                "",
                f"### Task {task_id}: {task.get('title', '')}",
                "",
                "#### Task Definition",
                self._format_jsonish(task),
                "",
                "#### Execute Output",
                record.get("output") or "(missing task output)",
                "",
                "#### Verification",
                self._format_jsonish(record.get("verification")),
            ])

        lines.extend([
            "",
            "## Evaluation",
            self._format_jsonish(data.get("evaluation")),
            "",
            "## Results Summary",
            data.get("results_summary") or "(missing results_summary.md)",
            "",
            "## Artifact Manifest",
        ])
        artifacts = data.get("artifacts", [])
        if artifacts:
            lines.extend([f"- `{item['path']}` ({item['size_bytes']} bytes)" for item in artifacts])
        else:
            lines.append("- No artifacts found.")

        lines.extend([
            "",
            "## Expected Report Structure",
            "- 标题",
            "- 摘要",
            "- 任务与数据",
            "- 方法策略",
            "- 执行过程与产物",
            "- 验证结果",
            "- 结论与限制",
            "- 可复现文件清单",
        ])
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _build_review_user(context_markdown: str, draft: str) -> str:
        return "\n\n".join([
            "# Fact Package",
            context_markdown,
            "# Draft Report",
            draft,
            "# Request",
            "请审查初稿。只输出 JSON。",
        ])

    @staticmethod
    def _build_polish_user(context_markdown: str, draft: str, review: dict[str, Any]) -> str:
        return "\n\n".join([
            "# Fact Package",
            context_markdown,
            "# Draft Report",
            draft,
            "# Review JSON",
            json.dumps(review, indent=2, ensure_ascii=False),
            "# Request",
            "请输出修订后的完整中文 Markdown 报告。",
        ])

    @staticmethod
    def _normalize_review(data: Any) -> dict[str, Any]:
        if not isinstance(data, dict):
            return {"issues": [{"section": "review", "problem": "Reviewer did not return JSON.", "suggestion": "Re-run report review."}], "approved": False}
        issues = data.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        normalized = []
        for issue in issues:
            if isinstance(issue, dict):
                normalized.append({
                    "section": str(issue.get("section", "General")),
                    "problem": str(issue.get("problem", "")),
                    "suggestion": str(issue.get("suggestion", "")),
                })
            else:
                normalized.append({
                    "section": "General",
                    "problem": str(issue),
                    "suggestion": "",
                })
        return {"issues": normalized, "approved": bool(data.get("approved", not normalized))}

    def _build_metadata_appendix(self, data: dict[str, Any]) -> str:
        artifacts = data.get("artifacts", [])
        tasks = data.get("task_records", [])
        passed = sum(1 for item in tasks if item.get("verification", {}).get("pass"))
        return "\n".join([
            "---",
            "",
            "## 附录：KaggleForge 执行记录",
            "",
            f"- Session ID: `{data.get('session_id')}`",
            f"- Source: {data.get('source')}",
            f"- Task count: {len(tasks)}",
            f"- Verified pass count: {passed}",
            f"- Artifact count: {len(artifacts)}",
            "",
            "### 关键文件",
            "- `task.md`: Kaggle 任务说明",
            "- `calibration.md`: 原子操作边界",
            "- `strategy.md`: research 策略",
            "- `plan_list.json`: 任务 DAG",
            "- `tasks/`: execute agent 输出",
            "- `verifications/`: verify agent 审查结果",
            "- `artifacts/`: research 阶段持久产物",
            "- `report_context.json`: report 阶段事实包",
            "- `paper.md`: report 初稿",
            "- `report_review.json`: report 审查结果",
            "- `paper_polished.md`: 最终报告",
        ])

    @staticmethod
    def _format_jsonish(value: Any) -> str:
        return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```"


WriteStage = ReportStage
