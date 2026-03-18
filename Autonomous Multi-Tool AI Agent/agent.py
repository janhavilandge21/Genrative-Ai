"""
agent.py — Autonomous Multi-Tool AI Agent using LangChain + Groq.

Uses LangChain's create_tool_calling_agent with a ReAct-style loop.
The agent autonomously selects tools based on the user's query.
"""

from __future__ import annotations

import os
from typing import Iterator

from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.callbacks import BaseCallbackHandler

from tools import ALL_TOOLS


# ─────────────────────────────────────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an autonomous AI agent with access to powerful tools.
You automatically select the best tool(s) for each task without asking for permission.

## Your Tools
- **calculator** → Any math: arithmetic, algebra, trigonometry, exponents
- **file_reader** → Read any local text file by path
- **file_writer** → Write or save content to a file (format: filepath|||content)
- **web_search** → Search the internet for real-time facts, news, data
- **python_repl** → Run Python code for data processing, algorithms, generation
- **text_analyzer** → Get statistics on any block of text
- **json_formatter** → Validate and pretty-print JSON strings

## Decision Rules
1. Math question → always use `calculator`
2. Need current info → always use `web_search`
3. File mentioned → use `file_reader` or `file_writer`
4. Complex computation / data task → use `python_repl`
5. Text stats requested → use `text_analyzer`
6. JSON provided → use `json_formatter`
7. Multi-step task → chain tools in sequence (search → analyze → write, etc.)

## Response Format
- Show your reasoning briefly before each tool call
- After all tool calls, provide a clear, well-formatted final answer
- Use markdown for structure when helpful
- Be concise but complete
"""


# ─────────────────────────────────────────────────────────────────────────────
# Callback for streaming tool events to Streamlit
# ─────────────────────────────────────────────────────────────────────────────
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Streams agent thoughts and tool calls into a Streamlit container."""

    def __init__(self, container):
        self.container = container
        self.log_lines: list[str] = []

    def _update(self, line: str):
        self.log_lines.append(line)
        self.container.markdown("\n".join(self.log_lines))

    def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get("name", "tool")
        self._update(f"🔧 **Calling:** `{tool_name}` with `{str(input_str)[:120]}`")

    def on_tool_end(self, output, **kwargs):
        preview = str(output)[:200].replace("\n", " ")
        self._update(f"   ↳ Result: {preview}{'...' if len(str(output)) > 200 else ''}\n")

    def on_agent_action(self, action, **kwargs):
        pass

    def on_agent_finish(self, finish, **kwargs):
        self._update("✅ **Agent finished.**")


# ─────────────────────────────────────────────────────────────────────────────
# Agent Builder
# ─────────────────────────────────────────────────────────────────────────────
def build_agent(model: str = "llama3-70b-8192", temperature: float = 0.1) -> AgentExecutor:
    """Construct and return the LangChain AgentExecutor."""
    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=os.environ["GROQ_API_KEY"],
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=True,
        max_iterations=10,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Run Agent
# ─────────────────────────────────────────────────────────────────────────────
def run_agent(
    query: str,
    chat_history: list | None = None,
    model: str = "llama3-70b-8192",
    temperature: float = 0.1,
    streamlit_container=None,
) -> dict:
    """
    Run the agent on a query and return results.

    Returns:
        {
          "output": str,
          "steps": list[tuple],   # (AgentAction, tool_output)
          "tools_used": list[str],
        }
    """
    executor = build_agent(model=model, temperature=temperature)

    callbacks = []
    if streamlit_container is not None:
        callbacks.append(StreamlitCallbackHandler(streamlit_container))

    history = []
    if chat_history:
        for msg in chat_history:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))

    result = executor.invoke(
        {"input": query, "chat_history": history},
        config={"callbacks": callbacks} if callbacks else {},
    )

    steps = result.get("intermediate_steps", [])
    tools_used = [step[0].tool for step in steps if hasattr(step[0], "tool")]

    return {
        "output": result.get("output", ""),
        "steps": steps,
        "tools_used": tools_used,
    }
