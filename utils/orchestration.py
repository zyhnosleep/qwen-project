"""Conversation orchestration for reflective, memory-aware agentic RAG."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

from smolagents import ToolCallingAgent

from utils.retrieval import SemanticRetriever


LOGGER = logging.getLogger(__name__)

AGENT_EXECUTION_INSTRUCTIONS = """
你是一个高准确性的图书馆问答助手。

始终遵守以下规则：
1. 结合“对话记忆”理解指代、省略和追问。
2. 如果任务里已经给了本地证据标签 [1]、[2] 等，回答时必须复用这些标签，不能自创新的本地证据编号。
3. 关键事实后要加内联引用，例如：该书由刘慈欣创作[1]。
4. 结尾必须输出“引用来源”小节。对本地文档证据，逐行使用给定的标准格式，例如：[1] 来源于《三体》。
5. 如果调用了数据库工具，请使用 [DB1]、[DB2] 之类的标签，并在“引用来源”中写明它来源于馆藏库存数据库。
6. 如果调用了网页搜索，请使用 [W1]、[W2] 之类的标签，并在“引用来源”中给出网页标题或 URL。
7. 没有证据支持的内容不要强答，应明确说明“不确定”或“当前证据不足”。
"""

QUERY_PLANNER_SYSTEM_PROMPT = """
你是一个查询路由器，负责为图书馆问答系统决定后续执行路径。

请根据“当前用户问题”和“对话记忆”返回一个 JSON 对象，字段必须完整：
{
  "route": "direct" | "local_grounded" | "agent_tooling",
  "retrieval_needed": true,
  "rewritten_queries": ["..."],
  "reason": "..."
}

字段含义：
- direct: 不需要检索和工具，直接回答即可。适用于打招呼、感谢、简单寒暄、关于助手自身的简单问题。
- local_grounded: 主要依赖本地知识库检索来回答，适用于课程资料、图书内容、概念解释、引用敏感的事实问答。
- agent_tooling: 需要进一步工具调用或多步推理，适用于实时网页信息、数据库库存信息、混合型复杂问题。

改写要求：
- rewritten_queries 最多 3 条。
- 如果用户问题里有“它/这本书/刚才那个作者”等指代，必须结合对话记忆把它展开成明确查询。
- 至少保留一个贴近原问题的查询。
- 如果 route 为 direct，也返回一个包含原问题的 rewritten_queries，便于日志一致。
"""

DIRECT_RESPONSE_SYSTEM_PROMPT = """
你是一个有对话记忆的图书馆助手。
当前问题不需要检索，请直接回答。

要求：
- 结合对话记忆理解省略和上下文。
- 保持自然、简洁。
- 不要杜撰外部事实。
- 如果只是寒暄或感谢，不需要引用。
"""

MEMORY_SUMMARY_SYSTEM_PROMPT = """
你负责压缩多轮对话记忆，供后续问答使用。

请把历史对话摘要成 4 到 6 条中文要点，重点保留：
- 用户正在做什么任务
- 用户明确提到的书名、作者、主题、编号
- 还未解决的问题或待继续追问的点
- 对回答格式的偏好，例如是否需要引用

不要展开解释，不要编造新信息。
"""

GROUNDED_SYNTHESIS_SYSTEM_PROMPT = """
你是一个图书馆问答助手，必须基于给定证据回答。

要求：
- 只能使用“本地证据”中的内容支撑事实性回答。
- 关键事实后必须加内联引用 [1]、[2]。
- 最后必须输出“引用来源”小节，并逐行照抄给定证据里的标准引用行。
- 如果证据不足，明确写“当前本地资料不足以确认”，但可以给出有限结论，并引用对应证据。
- 使用中文回答。
"""

REFLECTION_SYSTEM_PROMPT = """
你是最终审校器，要对草稿答案做反思与修订。

检查并修正以下问题：
- 是否正确利用了对话记忆，尤其是追问中的指代。
- 如果给了本地证据，关键事实是否带有 [1]、[2] 这样的内联引用。
- 是否存在超出证据的断言。
- 是否包含“引用来源”小节，且格式规范。

输出要求：
- 只输出最终修订后的答案。
- 若草稿已经合格，可做轻微润色，但不要删掉有效引用。
"""


@dataclass(frozen=True)
class ConversationTurn:
    """A single conversation turn."""

    user: str
    assistant: str


@dataclass(frozen=True)
class QueryPlan:
    """Planner output for one user query."""

    route: str
    retrieval_needed: bool
    rewritten_queries: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class EvidenceCard:
    """A citation-ready evidence item."""

    label: str
    title: str
    page_number: int | None
    source_type: str
    snippet: str
    citation_line: str

    def to_prompt_block(self) -> str:
        """Formats the evidence card for synthesis prompts."""
        page_text = f"第{self.page_number}页" if self.page_number else "页码未知"
        return (
            f"{self.label}\n"
            f"标题: 《{self.title}》\n"
            f"页码: {page_text}\n"
            f"来源类型: {self.source_type}\n"
            f"标准引用: {self.citation_line}\n"
            f"证据摘录: {self.snippet}"
        )


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _trim_text(text: str, max_chars: int = 480) -> str:
    text = _safe_text(text)
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars].rstrip()}..."


def _normalize_queries(queries: Sequence[str], fallback_query: str, limit: int = 3) -> tuple[str, ...]:
    normalized: list[str] = []
    for query in (fallback_query, *queries):
        cleaned = re.sub(r"\s+", " ", _safe_text(query))
        if cleaned and cleaned not in normalized:
            normalized.append(cleaned)
        if len(normalized) >= limit:
            break
    return tuple(normalized or (_safe_text(fallback_query),))


def _extract_json_object(text: str) -> dict[str, Any]:
    """Extracts the first JSON object from model output."""
    text = _safe_text(text)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _looks_like_small_talk(query: str) -> bool:
    """Fast path for trivial conversational turns."""
    normalized = re.sub(r"\s+", "", _safe_text(query).lower())
    if not normalized:
        return True
    small_talk_patterns = (
        "你好",
        "您好",
        "hi",
        "hello",
        "thanks",
        "thankyou",
        "谢谢",
        "再见",
        "拜拜",
        "你是谁",
        "早上好",
        "晚上好",
    )
    if normalized in small_talk_patterns:
        return True
    return len(normalized) <= 8 and any(token in normalized for token in small_talk_patterns)


class ConversationMemory:
    """Hybrid window-buffer plus rolling-summary memory."""

    def __init__(
        self,
        *,
        model: Any,
        window_size: int = 4,
        summary_trigger: int = 2,
    ) -> None:
        self._model = model
        self._window_size = max(window_size, 1)
        self._summary_trigger = max(summary_trigger, 1)
        self._summary = ""
        self._summary_buffer: list[ConversationTurn] = []
        self._recent_turns: list[ConversationTurn] = []

    def add_turn(self, user: str, assistant: str) -> None:
        """Adds a new turn and updates the rolling summary when needed."""
        self._recent_turns.append(ConversationTurn(user=_safe_text(user), assistant=_safe_text(assistant)))
        while len(self._recent_turns) > self._window_size:
            self._summary_buffer.append(self._recent_turns.pop(0))
        if len(self._summary_buffer) >= self._summary_trigger:
            self._flush_summary_buffer()

    def render_context(self) -> str:
        """Renders the current memory view for prompts."""
        sections: list[str] = []
        if self._summary:
            sections.append(f"历史摘要:\n{self._summary}")
        if self._summary_buffer:
            sections.append(f"较早但未压缩的对话:\n{self._format_turns(self._summary_buffer)}")
        if self._recent_turns:
            sections.append(f"最近对话:\n{self._format_turns(self._recent_turns)}")
        return "\n\n".join(sections).strip()

    @staticmethod
    def _format_turns(turns: Sequence[ConversationTurn]) -> str:
        lines: list[str] = []
        for index, turn in enumerate(turns, start=1):
            lines.append(f"第{index}轮用户: {turn.user}")
            lines.append(f"第{index}轮助手: {turn.assistant}")
        return "\n".join(lines)

    def _flush_summary_buffer(self) -> None:
        turn_block = self._format_turns(self._summary_buffer)
        prompt = (
            f"已有摘要:\n{self._summary or '无'}\n\n"
            f"新增历史对话:\n{turn_block}\n\n"
            "请输出新的压缩摘要。"
        )
        updated_summary = self._generate_text(MEMORY_SUMMARY_SYSTEM_PROMPT, prompt)
        if updated_summary:
            self._summary = updated_summary
        else:
            fallback = _trim_text(turn_block, max_chars=800)
            self._summary = f"{self._summary}\n{fallback}".strip()
        self._summary_buffer.clear()

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._model.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            LOGGER.warning("Conversation memory summarization failed: %s", exc)
            return ""
        return _safe_text(getattr(response, "content", ""))


class ReflectiveConversationAgent:
    """Adds planning, memory, and reflection around a tool-calling agent."""

    def __init__(
        self,
        *,
        model: Any,
        agent: ToolCallingAgent,
        retriever_tool: SemanticRetriever,
        memory: ConversationMemory | None = None,
        evidence_limit: int = 6,
        agent_max_steps: int = 8,
    ) -> None:
        self._model = model
        self._agent = agent
        self._retriever_tool = retriever_tool
        self._memory = memory or ConversationMemory(model=model)
        self._evidence_limit = max(evidence_limit, 1)
        self._agent_max_steps = max(agent_max_steps, 1)
        self._retriever_ready = False

    def run(self, query: str) -> str:
        """Runs one end-to-end turn with planning, memory, and reflection."""
        query = _safe_text(query)
        memory_context = self._memory.render_context()
        plan = self._plan_query(query, memory_context)
        LOGGER.info(
            "Query plan route=%s retrieval_needed=%s rewritten_queries=%s reason=%s",
            plan.route,
            plan.retrieval_needed,
            plan.rewritten_queries,
            plan.reason,
        )

        evidence_cards: list[EvidenceCard] = []
        if plan.retrieval_needed:
            evidence_cards = self._retrieve_evidence(plan.rewritten_queries)

        if plan.route == "direct":
            answer = self._answer_direct(query, memory_context)
        elif plan.route == "local_grounded":
            answer = self._answer_from_local_evidence(query, memory_context, plan, evidence_cards)
            answer = self._reflect_and_revise(query, memory_context, evidence_cards, answer)
        else:
            answer = self._answer_with_agent(query, memory_context, plan, evidence_cards)
            answer = self._reflect_and_revise(query, memory_context, evidence_cards, answer)

        self._memory.add_turn(query, answer)
        return answer

    def _plan_query(self, query: str, memory_context: str) -> QueryPlan:
        if _looks_like_small_talk(query):
            return QueryPlan(
                route="direct",
                retrieval_needed=False,
                rewritten_queries=(query,),
                reason="Heuristic small-talk fast path.",
            )

        planner_prompt = (
            f"对话记忆:\n{memory_context or '无'}\n\n"
            f"当前用户问题:\n{query}\n\n"
            "请只返回 JSON。"
        )
        payload = self._generate_json(QUERY_PLANNER_SYSTEM_PROMPT, planner_prompt)
        route = _safe_text(payload.get("route"))
        if route not in {"direct", "local_grounded", "agent_tooling"}:
            route = "local_grounded"
        retrieval_needed = bool(payload.get("retrieval_needed", route != "direct"))
        rewritten_queries = _normalize_queries(
            queries=payload.get("rewritten_queries") or (),
            fallback_query=query,
        )
        reason = _safe_text(payload.get("reason")) or "Planner fallback."
        return QueryPlan(
            route=route,
            retrieval_needed=retrieval_needed,
            rewritten_queries=rewritten_queries,
            reason=reason,
        )

    def _answer_direct(self, query: str, memory_context: str) -> str:
        prompt = (
            f"对话记忆:\n{memory_context or '无'}\n\n"
            f"当前用户问题:\n{query}"
        )
        answer = self._generate_text(DIRECT_RESPONSE_SYSTEM_PROMPT, prompt)
        return answer or "你好，我在。"

    def _answer_from_local_evidence(
        self,
        query: str,
        memory_context: str,
        plan: QueryPlan,
        evidence_cards: Sequence[EvidenceCard],
    ) -> str:
        if not evidence_cards:
            return "当前本地知识库中没有检索到足够证据，暂时无法给出带引用的可靠回答。"

        evidence_block = "\n\n".join(card.to_prompt_block() for card in evidence_cards)
        prompt = (
            f"对话记忆:\n{memory_context or '无'}\n\n"
            f"原始问题:\n{query}\n\n"
            f"改写后的检索查询:\n- " + "\n- ".join(plan.rewritten_queries) + "\n\n"
            f"本地证据:\n{evidence_block}\n\n"
            "请基于证据回答。"
        )
        answer = self._generate_text(GROUNDED_SYNTHESIS_SYSTEM_PROMPT, prompt)
        if answer:
            return answer
        return "当前本地资料不足，无法生成带引用的回答。"

    def _answer_with_agent(
        self,
        query: str,
        memory_context: str,
        plan: QueryPlan,
        evidence_cards: Sequence[EvidenceCard],
    ) -> str:
        evidence_block = "无"
        if evidence_cards:
            evidence_block = "\n\n".join(card.to_prompt_block() for card in evidence_cards)

        task = (
            f"用户问题:\n{query}\n\n"
            f"对话记忆:\n{memory_context or '无'}\n\n"
            f"查询改写:\n- " + "\n- ".join(plan.rewritten_queries) + "\n\n"
            f"可优先使用的本地证据:\n{evidence_block}\n\n"
            "要求：\n"
            "1. 如果给定的本地证据已经足够，请优先直接回答，不要重复检索。\n"
            "2. 如果仍然缺信息，再调用合适的工具。\n"
            "3. 最终回答必须尽量带引用；对本地证据使用现成的 [1]、[2] 标签。\n"
            "4. 结尾必须包含“引用来源”小节。\n"
            "5. 对不确定的内容要明确说明证据不足。"
        )
        try:
            answer = self._agent.run(task, reset=True, max_steps=self._agent_max_steps)
        except Exception as exc:
            LOGGER.warning("Tool-calling agent execution failed: %s", exc)
            answer = ""
        return _safe_text(answer) or "当前未能完成工具链求解，请稍后重试。"

    def _reflect_and_revise(
        self,
        query: str,
        memory_context: str,
        evidence_cards: Sequence[EvidenceCard],
        draft_answer: str,
    ) -> str:
        evidence_block = "无本地证据。"
        if evidence_cards:
            evidence_block = "\n\n".join(card.to_prompt_block() for card in evidence_cards)

        prompt = (
            f"对话记忆:\n{memory_context or '无'}\n\n"
            f"当前用户问题:\n{query}\n\n"
            f"可用本地证据:\n{evidence_block}\n\n"
            f"草稿答案:\n{draft_answer}\n\n"
            "请输出修订后的最终答案。"
        )
        revised = self._generate_text(REFLECTION_SYSTEM_PROMPT, prompt)
        return revised or draft_answer

    def _retrieve_evidence(self, queries: Sequence[str]) -> list[EvidenceCard]:
        if not self._retriever_ready:
            self._retriever_tool.setup()
            self._retriever_ready = True

        best_by_chunk: dict[str, dict[str, Any]] = {}
        for query in queries:
            try:
                result = self._retriever_tool.forward(query)
            except Exception as exc:
                LOGGER.warning("Local retrieval failed for query=%r: %s", query, exc)
                continue
            for item in result.get("results", []):
                chunk_id = _safe_text(item.get("chunk_id"))
                score = float(item.get("relevance_score", 0.0))
                if not chunk_id:
                    continue
                if chunk_id not in best_by_chunk or score > float(best_by_chunk[chunk_id].get("relevance_score", 0.0)):
                    best_by_chunk[chunk_id] = item

        ranked_items = sorted(
            best_by_chunk.values(),
            key=lambda item: float(item.get("relevance_score", 0.0)),
            reverse=True,
        )[: self._evidence_limit]

        evidence_cards: list[EvidenceCard] = []
        for index, item in enumerate(ranked_items, start=1):
            source = item.get("source", {})
            title = _safe_text(source.get("title")) or _safe_text(source.get("source")) or "未命名资料"
            page_number = source.get("page_number")
            if not isinstance(page_number, int):
                page_number = None
            source_type = _safe_text(source.get("source_type")) or "unknown"
            label = f"[{index}]"
            if page_number is None:
                citation_line = f"{label} 来源于《{title}》。"
            else:
                citation_line = f"{label} 来源于《{title}》第{page_number}页。"
            evidence_cards.append(
                EvidenceCard(
                    label=label,
                    title=title,
                    page_number=page_number,
                    source_type=source_type,
                    snippet=_trim_text(item.get("content", ""), max_chars=420),
                    citation_line=citation_line,
                )
            )
        LOGGER.info("Collected %d local evidence cards for rewritten_queries=%s", len(evidence_cards), queries)
        return evidence_cards

    def _generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        try:
            response = self._model.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            payload = _extract_json_object(getattr(response, "content", ""))
            if payload:
                return payload
        except Exception as exc:
            LOGGER.warning("Structured generation failed, retrying without response_format: %s", exc)

        try:
            response = self._model.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            LOGGER.warning("Fallback JSON generation failed: %s", exc)
            return {}
        return _extract_json_object(getattr(response, "content", ""))

    def _generate_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            response = self._model.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            LOGGER.warning("Text generation failed: %s", exc)
            return ""
        return _safe_text(getattr(response, "content", ""))
