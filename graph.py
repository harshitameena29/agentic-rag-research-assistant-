import json
import os
import re
from typing import Any, Dict, List, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END

load_dotenv()

llm = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

tavily = TavilySearch(
    max_results=5,
    topic="general",
)

class ResearchState(TypedDict, total=False):
    question: str
    sub_questions: List[str]
    current_index: int
    findings: List[Dict[str, Any]]
    final_report: str
    retriever: Any
    route: str

def extract_json(text: str):
    text = text.strip()
    text = re.sub(r"\`\`\`json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\`\`\`", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"**\\{**.\***\\}**", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    match = re.search(r"**\\[**.\***\\]**", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return None

def planner_node(state: ResearchState):
    question = state["question"]

    prompt = f"""
You are the planning agent of an Agentic RAG research assistant.

User question:

{question}

Break this question into 2 to 4 smaller research sub-questions.

The sub-questions should collectively provide enough information
to answer the original question.

Return ONLY valid JSON in this exact format:

{{
    "sub_questions": [
        "sub-question 1",
        "sub-question 2",
        "sub-question 3"
    ]
}}
"""

    response = llm.invoke(prompt)
    parsed = extract_json(response.content)

    if parsed and isinstance(parsed, dict):
        sub_questions = parsed.get("sub_questions", [])
    else:
        sub_questions = [question]

    if not sub_questions:
        sub_questions = [question]

    return {
        "sub_questions": sub_questions,
        "current_index": 0,
        "findings": [],
    }

def route_node(state: ResearchState):
    retriever = state.get("retriever")

    if retriever is None:
        return {"route": "web"}

    current_question = state["sub_questions"][state["current_index"]]

    prompt = f"""
You are a routing agent in an Agentic RAG system.

A user uploaded a PDF document.

Research sub-question:

{current_question}

Decide where this question should be answered from.

Choose:

- local = use the uploaded PDF
- web = use live web search
- both = use both the PDF and web

Return ONLY one word:

local
web
both
"""

    response = llm.invoke(prompt)
    route = response.content.strip().lower()

    if route not in ["local", "web", "both"]:
        route = "local"

    return {
        "route": route
    }

def local_research_node(state: ResearchState):
    retriever = state.get("retriever")

    if retriever is None:
        return {}

    current_question = state["sub_questions"][state["current_index"]]
    documents = retriever.invoke(current_question)

    findings = state.get("findings", []).copy()
    retrieved_text = []

    for doc in documents:
        page_number = doc.metadata.get("page")

        if page_number is not None:
            source = f"PDF page {page_number + 1}"
        else:
            source = "Uploaded PDF"

        retrieved_text.append(
            f"[Source: {source}]\n{doc.page_content}"
        )

    findings.append(
        {
            "sub_question": current_question,
            "source_type": "pdf",
            "content": "\n\n".join(retrieved_text),
        }
    )

    return {
        "findings": findings
    }

def web_research_node(state: ResearchState):
    current_question = state["sub_questions"][state["current_index"]]

    try:
        result = tavily.invoke(
            {
                "query": current_question
            }
        )

        findings = state.get("findings", []).copy()

        findings.append(
            {
                "sub_question": current_question,
                "source_type": "web",
                "content": str(result),
            }
        )

        return {
            "findings": findings
        }

    except Exception as e:
        findings = state.get("findings", []).copy()

        findings.append(
            {
                "sub_question": current_question,
                "source_type": "web",
                "content": f"Web search failed: {str(e)}",
            }
        )

        return {
            "findings": findings
        }

def both_research_node(state: ResearchState):
    state_after_local = local_research_node(state)

    updated_state = {
        **state,
        **state_after_local
    }

    state_after_web = web_research_node(updated_state)

    return {
        "findings": state_after_web.get(
            "findings",
            updated_state.get("findings", [])
        )
    }

def continue_research(state: ResearchState):
    current_index = state["current_index"]
    total_questions = len(state["sub_questions"])
    next_index = current_index + 1

    if next_index >= total_questions:
        return "finish"

    return "continue"

def increment_question(state: ResearchState):
    return {
        "current_index": state["current_index"] + 1
    }

def writer_node(state: ResearchState):
    question = state["question"]
    findings = state.get("findings", [])

    research_text = ""

    for i, finding in enumerate(findings, 1):
        research_text += f"""
==================================================
RESEARCH RESULT {i}
==================================================

Sub-question:
{finding.get("sub_question", "")}

Source type:
{finding.get("source_type", "")}

Information:
{finding.get("content", "")}
"""

    prompt = f"""
You are the final answer writer in an Agentic RAG system.

Original user question:

{question}

Below are research results collected by the agent:

{research_text}

Write a clear, accurate and well-structured answer to the
original question.

IMPORTANT RULES:

1. Base the answer on the provided research.
2. Do not invent facts.
3. If information came from the uploaded PDF, mention the
   relevant PDF page when possible.
4. If information came from web search, clearly identify it
   as web research.
5. If the uploaded PDF does not contain enough information,
   say so instead of pretending it does.
6. Use headings and bullet points where useful.
7. Give a concise but sufficiently detailed answer.
"""

    response = llm.invoke(prompt)

    return {
        "final_report": response.content
    }

def build_graph(retriever=None):
    workflow = StateGraph(ResearchState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("router", route_node)
    workflow.add_node("local_research", local_research_node)
    workflow.add_node("web_research", web_research_node)
    workflow.add_node("both_research", both_research_node)
    workflow.add_node("increment", increment_question)
    workflow.add_node("writer", writer_node)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "router")

    workflow.add_conditional_edges(
        "router",
        lambda state: state.get("route", "local"),
        {
            "local": "local_research",
            "web": "web_research",
            "both": "both_research",
        },
    )

    workflow.add_edge("local_research", "increment")
    workflow.add_edge("web_research", "increment")
    workflow.add_edge("both_research", "increment")

    workflow.add_conditional_edges(
        "increment",
        continue_research,
        {
            "continue": "router",
            "finish": "writer",
        },
    )

    workflow.add_edge("writer", END)

    return workflow.compile()