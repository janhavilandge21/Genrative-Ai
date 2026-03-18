"""
tools.py — LangChain tools for web search, article fetching, and summarization.
"""

from __future__ import annotations

import os
import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_groq import ChatGroq


# ── Shared LLM (used inside tools for summarization) ─────────────────────────
def _get_llm() -> ChatGroq:
    return ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        api_key=os.environ["GROQ_API_KEY"],
    )


# ── Tool 1: Web Search (via Tavily) ──────────────────────────────────────────
@tool
def web_search(query: str) -> str:
    """
    Search the web for a given query using Tavily Search API.
    Returns a list of relevant results with titles, URLs, and snippets.
    Use this to find recent articles and information on any topic.
    """
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(
            query=query,
            max_results=5,
            include_answer=True,
            include_raw_content=False,
        )
        results = []
        if response.get("answer"):
            results.append(f"Quick Answer: {response['answer']}\n")

        for i, r in enumerate(response.get("results", []), 1):
            results.append(
                f"[{i}] {r.get('title', 'No title')}\n"
                f"    URL: {r.get('url', '')}\n"
                f"    Snippet: {r.get('content', '')[:400]}\n"
            )
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# ── Tool 2: Fetch & Extract Article Text ─────────────────────────────────────
@tool
def fetch_article(url: str) -> str:
    """
    Fetch and extract the main text content from a given URL.
    Use this after web_search to read the full content of a specific article.
    Input must be a valid URL string.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (research-assistant-bot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract text from article/main content areas
        for selector in ["article", "main", '[role="main"]', ".content", "#content"]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(separator="\n", strip=True)
                if len(text) > 200:
                    return text[:4000]

        # Fallback: body text
        text = soup.body.get_text(separator="\n", strip=True) if soup.body else ""
        return text[:4000] if text else "Could not extract article text."
    except Exception as e:
        return f"Fetch error for {url}: {e}"


# ── Tool 3: Summarize Text ───────────────────────────────────────────────────
@tool
def summarize_text(text: str) -> str:
    """
    Summarize a long piece of text into key points.
    Use this after fetching an article to condense its content.
    Input should be the raw article text.
    """
    if not text or len(text.strip()) < 100:
        return "Text too short to summarize."
    try:
        llm = _get_llm()
        prompt = (
            "You are a research assistant. Summarize the following text into "
            "5-7 clear bullet points capturing the most important facts, findings, "
            "and insights. Be concise and factual.\n\n"
            f"TEXT:\n{text[:3000]}\n\nSUMMARY:"
        )
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Summarization error: {e}"


# ── Tool 4: Generate Research Report ─────────────────────────────────────────
@tool
def generate_report(research_data: str) -> str:
    """
    Generate a structured research report from collected research data.
    Input should be a string containing all gathered summaries and findings.
    Use this as the FINAL step after collecting and summarizing information.
    Returns a well-formatted markdown research report.
    """
    try:
        llm = _get_llm()
        prompt = (
            "You are a professional research analyst. Based on the following research data, "
            "generate a comprehensive, well-structured research report in Markdown format.\n\n"
            "The report must include:\n"
            "1. **Executive Summary** (2-3 sentences)\n"
            "2. **Key Findings** (bullet points)\n"
            "3. **Detailed Analysis** (3-5 paragraphs)\n"
            "4. **Trends & Insights**\n"
            "5. **Conclusion & Recommendations**\n\n"
            f"RESEARCH DATA:\n{research_data}\n\n"
            "REPORT:"
        )
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Report generation error: {e}"


# ── All tools exported ────────────────────────────────────────────────────────
ALL_TOOLS = [web_search, fetch_article, summarize_text, generate_report]
