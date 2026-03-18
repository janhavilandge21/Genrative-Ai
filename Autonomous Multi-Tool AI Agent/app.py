"""
app.py — Streamlit UI for the Autonomous Multi-Tool AI Agent.
"""

import os
import sys
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Tool AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0a0c10;
    --surface: #13151c;
    --surface2: #1a1d27;
    --accent: #00d4aa;
    --accent2: #7c6af7;
    --orange: #f5a623;
    --red: #e06c75;
    --text: #e2e8f0;
    --muted: #64748b;
    --border: #1e2233;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
.stChatMessage {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    padding: 4px !important;
}
.stChatInputContainer textarea {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #0a0c10 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.tool-badge {
    display: inline-block;
    background: #1a1d27;
    border: 1px solid #2a2d3e;
    color: #00d4aa;
    border-radius: 6px;
    padding: 2px 9px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    margin: 2px 3px;
}
.step-box {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent2);
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--muted);
}
.hero {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4aa, #7c6af7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.env-warning {
    background: #2d1a1a;
    border: 1px solid #5c2d2d;
    border-radius: 8px;
    padding: 12px 16px;
    color: #e06c75;
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Example Prompts
# ─────────────────────────────────────────────────────────────────────────────
EXAMPLE_PROMPTS = {
    "🧮 Math": [
        "What is 2^32 and what is its square root?",
        "Calculate compound interest: $5000 at 7.5% for 10 years",
        "What is sin(45°) + cos(60°) × tan(30°)?",
    ],
    "🌐 Web Search": [
        "What are the latest features in Python 3.13?",
        "Search for the current price of gold per ounce",
        "Find the top 5 AI tools released in 2024",
    ],
    "🐍 Python Code": [
        "Write and run Python code to generate the first 15 Fibonacci numbers",
        "Run code to find all prime numbers between 1 and 100",
        "Generate a random password of 16 characters with symbols",
    ],
    "📊 Text Analysis": [
        "Analyze this text: 'Artificial intelligence is transforming every industry. Machine learning models are becoming more powerful and accessible to developers worldwide.'",
        "Count the words and sentences in: 'The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.'",
    ],
    "🔧 Multi-Tool": [
        "Search for Python best practices, then write a Python script that prints 10 tips",
        "Calculate the area of a circle with radius 7, then write the result to a file called circle_area.txt",
        "Generate 5 random numbers using Python code, then analyze the text output",
    ],
    "🔧 JSON": [
        'Format this JSON: {"name":"Alice","age":30,"skills":["python","ml","langchain"]}',
        'Validate and pretty-print: [{"id":1,"task":"buy groceries"},{"id":2,"task":"code project"}]',
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_tool_calls" not in st.session_state:
    st.session_state.total_tool_calls = 0


# ─────────────────────────────────────────────────────────────────────────────
# Environment Check
# ─────────────────────────────────────────────────────────────────────────────
def check_env() -> tuple[bool, list[str]]:
    missing = []
    if not os.environ.get("GROQ_API_KEY"):
        missing.append("GROQ_API_KEY")
    if not os.environ.get("TAVILY_API_KEY"):
        missing.append("TAVILY_API_KEY (optional — needed for web_search)")
    groq_ok = "GROQ_API_KEY" not in missing
    return groq_ok, missing


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="hero">🤖 AI Agent</p>', unsafe_allow_html=True)
    st.markdown("**Autonomous Multi-Tool Assistant**")
    st.divider()

    # API Key inputs
    st.markdown("### 🔑 API Keys")
    groq_input = st.text_input(
        "Groq API Key",
        value=os.environ.get("GROQ_API_KEY", ""),
        type="password",
        placeholder="gsk_...",
    )
    if groq_input:
        os.environ["GROQ_API_KEY"] = groq_input

    tavily_input = st.text_input(
        "Tavily API Key (for web search)",
        value=os.environ.get("TAVILY_API_KEY", ""),
        type="password",
        placeholder="tvly_...",
    )
    if tavily_input:
        os.environ["TAVILY_API_KEY"] = tavily_input

    st.divider()

    # Model settings
    st.markdown("### ⚙️ Model Settings")
    model = st.selectbox(
        "Groq Model",
        ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
        index=0,
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.1, 0.05)

    st.divider()

    # Tools overview
    st.markdown("### 🛠️ Available Tools")
    from tools import TOOL_DESCRIPTIONS
    for tool_name, desc in TOOL_DESCRIPTIONS.items():
        st.markdown(f"<span class='tool-badge'>{tool_name}</span> {desc}", unsafe_allow_html=True)

    st.divider()

    # Stats
    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("Queries", st.session_state.total_queries)
    col2.metric("Tool Calls", st.session_state.total_tool_calls)

    st.divider()
    if st.button("🗑 Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_tool_calls = 0
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Main Area
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero">Autonomous Multi-Tool AI Agent</h1>', unsafe_allow_html=True)
st.markdown("Ask anything — the agent automatically selects the right tools to answer.")

# Env warnings
groq_ok, missing = check_env()
if not groq_ok:
    st.markdown(
        '<div class="env-warning">⚠️ <b>GROQ_API_KEY not set.</b> Enter it in the sidebar or add to your .env file.<br>'
        'Get a free key at <a href="https://console.groq.com" style="color:#e06c75">console.groq.com</a></div>',
        unsafe_allow_html=True
    )

# Example prompts
with st.expander("💡 Example Prompts — Click to try", expanded=not st.session_state.messages):
    for category, prompts in EXAMPLE_PROMPTS.items():
        st.markdown(f"**{category}**")
        cols = st.columns(len(prompts))
        for col, prompt in zip(cols, prompts):
            if col.button(prompt[:55] + ("…" if len(prompt) > 55 else ""), key=f"ex_{prompt[:20]}"):
                st.session_state["pending_prompt"] = prompt
                st.rerun()

st.divider()

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("tools_used"):
            st.markdown(
                "**Tools used:** " + " ".join(f"<span class='tool-badge'>{t}</span>" for t in msg["tools_used"]),
                unsafe_allow_html=True
            )
        if msg.get("steps"):
            with st.expander(f"🔍 Agent steps ({len(msg['steps'])})"):
                for action, output in msg["steps"]:
                    st.markdown(
                        f'<div class="step-box">🔧 <b>{action.tool}</b>({str(action.tool_input)[:80]})<br>'
                        f'↳ {str(output)[:200]}{"..." if len(str(output)) > 200 else ""}</div>',
                        unsafe_allow_html=True
                    )

# Handle example prompt injection
pending = st.session_state.pop("pending_prompt", None)

# Chat input
prompt = st.chat_input("Ask me anything — math, web search, code, files, JSON...") or pending

if prompt:
    if not groq_ok:
        st.warning("Please enter your Groq API key in the sidebar first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Agent response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            log_container = st.container()
            thinking_placeholder.markdown("🤔 _Agent is thinking..._")

            try:
                from agent import run_agent
                result = run_agent(
                    query=prompt,
                    chat_history=st.session_state.messages[:-1],
                    model=model,
                    temperature=temperature,
                    streamlit_container=log_container,
                )
                thinking_placeholder.empty()

                answer = result["output"]
                tools_used = result["tools_used"]
                steps = result["steps"]

                st.markdown(answer)
                if tools_used:
                    st.markdown(
                        "**Tools used:** " + " ".join(f"<span class='tool-badge'>{t}</span>" for t in tools_used),
                        unsafe_allow_html=True
                    )
                if steps:
                    with st.expander(f"🔍 Agent steps ({len(steps)})"):
                        for action, output in steps:
                            st.markdown(
                                f'<div class="step-box">🔧 <b>{action.tool}</b>({str(action.tool_input)[:80]})<br>'
                                f'↳ {str(output)[:200]}{"..." if len(str(output)) > 200 else ""}</div>',
                                unsafe_allow_html=True
                            )

                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "tools_used": tools_used,
                    "steps": steps,
                })
                st.session_state.total_queries += 1
                st.session_state.total_tool_calls += len(steps)

            except Exception as e:
                thinking_placeholder.empty()
                err_msg = str(e)
                st.error(f"❌ Agent error: {err_msg}")
                if "GROQ_API_KEY" in err_msg or "Authentication" in err_msg:
                    st.info("💡 Check your Groq API key in the sidebar.")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error: {err_msg}",
                    "tools_used": [],
                    "steps": [],
                })
