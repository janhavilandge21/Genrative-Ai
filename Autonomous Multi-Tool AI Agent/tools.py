"""
tools.py — All LangChain tools for the Autonomous Multi-Tool AI Agent.

Tools implemented:
  1. calculator          — Safe math expression evaluator
  2. file_reader         — Read local text/code files
  3. file_writer         — Write content to local files
  4. web_search          — Real-time web search via Tavily
  5. python_repl         — Execute Python code safely
  6. text_analyzer       — Count words, chars, lines in text
  7. json_formatter      — Pretty-print and validate JSON
"""

from __future__ import annotations

import os
import json
import math
import ast
import traceback
import textwrap
from pathlib import Path
from typing import Any

from langchain.tools import tool


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 1 — Calculator
# ─────────────────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, abs, round, pi, e.
    Example inputs: "2 ** 10", "sqrt(144)", "sin(pi/2)", "(100 * 1.08) ** 3"
    Always use this for ANY math calculation — never compute manually.
    """
    try:
        # Safe math namespace
        safe_ns = {
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "log2": math.log2, "abs": abs, "round": round,
            "pow": pow, "exp": math.exp, "floor": math.floor,
            "ceil": math.ceil, "pi": math.pi, "e": math.e,
            "inf": math.inf,
        }
        # Parse and validate AST — only allow safe nodes
        tree = ast.parse(expression.strip(), mode="eval")
        allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
            ast.Constant, ast.Name, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.UAdd, ast.USub,
            ast.Load,
        )
        for node in ast.walk(tree):
            if not isinstance(node, allowed):
                return f"❌ Unsafe expression. Only math operations allowed."
        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, safe_ns)
        return f"✅ Result: {expression} = {result}"
    except ZeroDivisionError:
        return "❌ Error: Division by zero."
    except Exception as e:
        return f"❌ Calculation error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 2 — File Reader
# ─────────────────────────────────────────────────────────────────────────────
@tool
def file_reader(file_path: str) -> str:
    """
    Read and return the contents of a local text file.
    Supports any text-based file: .txt, .py, .json, .csv, .md, .html, .yaml, etc.
    Input: the full or relative file path as a string.
    Example: file_reader("data/notes.txt") or file_reader("C:/Users/me/report.md")
    """
    try:
        path = Path(file_path.strip().strip('"').strip("'"))
        if not path.exists():
            return f"❌ File not found: {path.resolve()}"
        if not path.is_file():
            return f"❌ Path is not a file: {path.resolve()}"
        size = path.stat().st_size
        if size > 500_000:  # 500 KB limit
            return f"❌ File too large ({size // 1024} KB). Max 500 KB."
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + 1
        preview = content[:5000]
        truncated = " [truncated at 5000 chars]" if len(content) > 5000 else ""
        return (
            f"📄 File: {path.name} | Size: {size} bytes | Lines: {lines}\n"
            f"{'─'*50}\n{preview}{truncated}"
        )
    except PermissionError:
        return f"❌ Permission denied: {file_path}"
    except Exception as e:
        return f"❌ Error reading file: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 3 — File Writer
# ─────────────────────────────────────────────────────────────────────────────
@tool
def file_writer(input: str) -> str:
    """
    Write content to a local file. Input format: "filepath|||content"
    The separator is exactly three pipe characters: |||
    Example: file_writer("output/notes.txt|||Hello, this is my content here.")
    Creates parent directories automatically. Will overwrite existing files.
    """
    try:
        if "|||" not in input:
            return "❌ Invalid format. Use: filepath|||content"
        file_path, content = input.split("|||", 1)
        path = Path(file_path.strip())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"✅ Saved {len(content)} characters to: {path.resolve()}"
    except Exception as e:
        return f"❌ Error writing file: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 4 — Web Search (Tavily)
# ─────────────────────────────────────────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the web for real-time information using Tavily Search API.
    Use this for: current events, facts, recent data, URLs, anything needing internet.
    Input: a natural language search query string.
    Example: web_search("latest Python 3.13 features") or web_search("SpaceX launch today")
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return "❌ TAVILY_API_KEY not set. Add it to your .env file."
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=5,
            include_answer=True,
        )
        parts = []
        if response.get("answer"):
            parts.append(f"🔍 Quick Answer: {response['answer']}\n")
        for i, r in enumerate(response.get("results", []), 1):
            parts.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('url', '')}\n"
                f"    {r.get('content', '')[:350]}\n"
            )
        return "\n".join(parts) if parts else "No results found."
    except Exception as e:
        return f"❌ Search error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 5 — Python Code Executor (safe REPL)
# ─────────────────────────────────────────────────────────────────────────────
@tool
def python_repl(code: str) -> str:
    """
    Execute Python code and return the output.
    Use this to: process data, run algorithms, generate text, do analysis.
    The code runs in a restricted environment. Use print() to see output.
    Input: a valid Python code string.
    Example: python_repl("import random\\nprint([random.randint(1,100) for _ in range(5)])")
    Note: No file system or network access inside the executor.
    """
    # Strip markdown code fences if present
    code = code.strip()
    for fence in ["```python", "```py", "```", "`"]:
        if code.startswith(fence):
            code = code[len(fence):]
        if code.endswith(fence[::-1] if fence == "`" else fence):
            code = code[: -len(fence)]
    code = textwrap.dedent(code).strip()

    import io, sys
    output_buffer = io.StringIO()
    error_buffer  = io.StringIO()

    # Restricted builtins — no open(), __import__, etc.
    safe_builtins = {
        k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
        for k in [
            "print", "len", "range", "enumerate", "zip", "map", "filter",
            "list", "dict", "set", "tuple", "str", "int", "float", "bool",
            "sum", "min", "max", "abs", "round", "sorted", "reversed",
            "isinstance", "type", "repr", "any", "all", "chr", "ord",
            "hex", "oct", "bin", "format", "hash", "id",
        ]
        if (isinstance(__builtins__, dict) and k in __builtins__)
        or (not isinstance(__builtins__, dict) and hasattr(__builtins__, k))
    }
    # Allow common safe imports
    import math, random, datetime, collections, itertools, functools, string, re, json as _json
    exec_globals = {
        "__builtins__": safe_builtins,
        "math": math, "random": random, "datetime": datetime,
        "collections": collections, "itertools": itertools,
        "functools": functools, "string": string, "re": re, "json": _json,
    }

    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = output_buffer, error_buffer
    try:
        exec(code, exec_globals)
        output = output_buffer.getvalue()
        errors = error_buffer.getvalue()
        result = ""
        if output:
            result += f"📤 Output:\n{output}"
        if errors:
            result += f"\n⚠️ Stderr:\n{errors}"
        return result.strip() or "✅ Code executed successfully (no output)."
    except Exception:
        return f"❌ Execution error:\n{traceback.format_exc()}"
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 6 — Text Analyzer
# ─────────────────────────────────────────────────────────────────────────────
@tool
def text_analyzer(text: str) -> str:
    """
    Analyze a block of text and return statistics: word count, character count,
    line count, sentence count, average word length, most common words.
    Use this when asked to analyze, count, or get statistics about text.
    Input: the text to analyze.
    """
    if not text.strip():
        return "❌ No text provided."
    import re as _re
    from collections import Counter
    words   = _re.findall(r'\b\w+\b', text.lower())
    lines   = text.splitlines()
    sentences = _re.split(r'[.!?]+', text)
    sentences = [s for s in sentences if s.strip()]
    avg_len = sum(len(w) for w in words) / len(words) if words else 0
    top_words = Counter(w for w in words if len(w) > 3).most_common(10)
    top_str = ", ".join(f'"{w}" ({c})' for w, c in top_words)
    return (
        f"📊 Text Analysis Results:\n"
        f"  • Characters (total):  {len(text)}\n"
        f"  • Characters (no spaces): {len(text.replace(' ', ''))}\n"
        f"  • Words:               {len(words)}\n"
        f"  • Lines:               {len(lines)}\n"
        f"  • Sentences:           {len(sentences)}\n"
        f"  • Avg word length:     {avg_len:.1f} chars\n"
        f"  • Unique words:        {len(set(words))}\n"
        f"  • Top words:           {top_str}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TOOL 7 — JSON Formatter
# ─────────────────────────────────────────────────────────────────────────────
@tool
def json_formatter(input_json: str) -> str:
    """
    Parse, validate, and pretty-print a JSON string.
    Use when the user provides JSON that needs formatting, validation, or inspection.
    Also shows key structure information about the JSON.
    Input: a JSON string (can be minified or messy).
    """
    try:
        data = json.loads(input_json.strip())
        pretty = json.dumps(data, indent=2, ensure_ascii=False)
        # Structure summary
        if isinstance(data, dict):
            keys = list(data.keys())
            summary = f"Object with {len(keys)} keys: {', '.join(str(k) for k in keys[:10])}"
        elif isinstance(data, list):
            summary = f"Array with {len(data)} items"
        else:
            summary = f"Scalar value: {type(data).__name__}"
        return f"✅ Valid JSON — {summary}\n\n{pretty}"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# All tools for export
# ─────────────────────────────────────────────────────────────────────────────
ALL_TOOLS = [
    calculator,
    file_reader,
    file_writer,
    web_search,
    python_repl,
    text_analyzer,
    json_formatter,
]

TOOL_DESCRIPTIONS = {
    "calculator":     "🧮 Safe math expression evaluator",
    "file_reader":    "📄 Read local text/code files",
    "file_writer":    "💾 Write content to local files",
    "web_search":     "🌐 Real-time web search (Tavily)",
    "python_repl":    "🐍 Execute Python code safely",
    "text_analyzer":  "📊 Analyze text statistics",
    "json_formatter": "🔧 Validate & pretty-print JSON",
}
