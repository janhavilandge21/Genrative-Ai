"""
agent.py — LangGraph-powered Agentic Research Assistant.

Graph structure:
  START → research_planner → tool_executor (loop) → report_generator → END

The agent:
  1. Breaks the query into a research plan
  2. Executes tools iteratively (search → fetch → summarize)
  3. Synthesizes findings into a structured report
"""

from __future__ import annotations

import os
import json
from typing import Annotated, TypedDict, Sequence

from langchain_groq import ChatGroq
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
)
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import ALL_TOOLS


# ── Agent State ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    research_notes: list[str]
    final_report: str
    step_count: int


# ── LLM Setup ─────────────────────────────────────────────────────────────────
def get_llm_with_tools() -> ChatGroq:
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.1,
        api_key=os.environ["GROQ_API_KEY"],
    )
    return llm.bind_tools(ALL_TOOLS)


SYSTEM_PROMPT = """You are an expert AI Research Assistant with access to these tools:

1. **web_search(query)** — Search the web for recent information on any topic
2. **fetch_article(url)** — Fetch and read the full content of a specific URL
3. **summarize_text(text)** — Summarize long articles into key bullet points
4. **generate_report(research_data)** — Generate a structured research report from all collected data

## Your Research Workflow:
1. Start by searching for the main topic using web_search
2. Identify 2-3 most relevant URLs from search results
3. Fetch and read those articles using fetch_article
4. Summarize each article using summarize_text
5. Collect all summaries, then call generate_report with ALL findings combined
6. The generate_report call should be your FINAL tool call

## Important Rules:
- Always search before fetching articles
- Summarize each article after fetching it
- Combine ALL summaries before generating the final report
- Do not stop until you have a complete research report
- Be systematic and thorough
"""


# ── Node: Research Agent ──────────────────────────────────────────────────────
def research_agent_node(state: AgentState) -> AgentState:
    """Main agent node — decides which tool to call next."""
    llm_with_tools = get_llm_with_tools()

    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response = llm_with_tools.invoke(messages)

    return {
        **state,
        "messages": [response],
        "step_count": state.get("step_count", 0) + 1,
    }


# ── Node: Tool Executor ───────────────────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)


# ── Node: Collect Research Notes ─────────────────────────────────────────────
def collect_notes_node(state: AgentState) -> AgentState:
    """Extract tool results and accumulate research notes."""
    notes = list(state.get("research_notes", []))
    final_report = state.get("final_report", "")

    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            # Detect final report (generate_report output)
            if any(keyword in content for keyword in ["## ", "**Executive Summary**", "**Key Findings**", "# Research Report"]):
                final_report = content
            elif len(content) > 50:
                notes.append(content)
            break  # Only process the latest tool message

    return {
        **state,
        "research_notes": notes,
        "final_report": final_report,
    }


# ── Edge: Should Continue or End ─────────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    """Decide whether to continue tool execution or end."""
    messages = state["messages"]
    step_count = state.get("step_count", 0)

    # Safety limit
    if step_count > 15:
        return "end"

    # If final report is generated, end
    if state.get("final_report"):
        return "end"

    # Check if the last AI message has tool calls
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                return "continue"
            return "end"

    return "end"


# ── Build the Graph ───────────────────────────────────────────────────────────
def build_research_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("agent", research_agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("collect_notes", collect_notes_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "tools", "end": END},
    )
    graph.add_edge("tools", "collect_notes")
    graph.add_edge("collect_notes", "agent")

    return graph.compile()


# ── Public API ────────────────────────────────────────────────────────────────
class ResearchAssistant:
    def __init__(self):
        self.graph = build_research_graph()

    def research(self, query: str, callbacks=None) -> dict:
        """
        Run the research agent on a query.
        Returns dict with 'report', 'notes', 'messages', 'steps'.
        """
        initial_state: AgentState = {
            "messages": [HumanMessage(content=f"Research this topic thoroughly: {query}")],
            "query": query,
            "research_notes": [],
            "final_report": "",
            "step_count": 0,
        }

        config = {}
        if callbacks:
            config["callbacks"] = callbacks

        final_state = self.graph.invoke(initial_state, config=config)

        report = final_state.get("final_report", "")
        if not report:
            # Fallback: last AI message content
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    report = msg.content
                    break

        return {
            "report": report,
            "notes": final_state.get("research_notes", []),
            "steps": final_state.get("step_count", 0),
            "messages": final_state["messages"],
        }
