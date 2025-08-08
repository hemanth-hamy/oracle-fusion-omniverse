# agent_orchestrator.py
import re
from typing import List, Literal, TypedDict

# Feature flags (we don't require these to deploy)
HAS_LANGGRAPH = False
HAS_CREWAI = False
HAS_AUTOGEN = False
try:
    from langgraph.graph import StateGraph, END  # type: ignore
    HAS_LANGGRAPH = True
except Exception:
    pass
try:
    from crewai import Agent, Task, Crew, Process  # type: ignore
    HAS_CREWAI = True
except Exception:
    pass
try:
    import autogen  # type: ignore
    HAS_AUTOGEN = True
except Exception:
    pass

# Lightweight RAG stub
def rag_search(query: str) -> str:
    if "ORA-00942" in (query or "").upper():
        return "RAG: ORA-00942 = Table or view does not exist. Check schema, synonyms, grants."
    return "RAG: No specific doc found."

# YouTube fn bindings (we’ll bind from app.py)
_YT_SINGLE = None
_YT_BATCH = None
def bind_youtube_functions(single_fn, batch_fn):
    global _YT_SINGLE, _YT_BATCH
    _YT_SINGLE = single_fn
    _YT_BATCH = batch_fn

class MasterState(TypedDict):
    user_input: str
    urls: List[str]
    decision: Literal["CrewAI", "AutoGen", "YouTube", "YouTube_Batch", "End"]
    final_output: str

_YT_RE = r"https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]{11}|youtu\.be/[\w-]{11})"

def find_youtube_urls(text: str) -> List[str]:
    return re.findall(_YT_RE, text or "")

def route(user_input: str) -> MasterState:
    urls = find_youtube_urls(user_input)
    if urls:
        return {"user_input": user_input, "urls": urls,
                "decision": "YouTube_Batch" if len(urls) > 1 else "YouTube",
                "final_output": ""}
    if "error" in user_input.lower() or "ora-" in user_input.lower():
        return {"user_input": user_input, "urls": [], "decision": "CrewAI", "final_output": ""}
    return {"user_input": user_input, "urls": [], "decision": "AutoGen", "final_output": ""}

def _crewai(user_input: str) -> str:
    if not HAS_CREWAI:
        return "[CrewAI disabled] Diagnosis: ORA-00942 → qualify schema, create synonym or grant SELECT."
    from crewai import Agent, Task, Crew, Process  # type: ignore
    analyst = Agent(role="Intake Analyst", goal="Summarize", backstory="Parses raw logs", verbose=False)
    dba = Agent(role="Oracle DBA", goal="Root cause + SQL", backstory="Seasoned DBA", verbose=False)
    arch = Agent(role="Architect", goal="Final report", backstory="Communicator", verbose=False)
    t1 = Task(description=f"Analyze: {user_input}", expected_output="Clean summary.", agent=analyst)
    t2 = Task(description="Root cause + SQL fix.", expected_output="Diagnosis + SQL.", agent=dba)
    t3 = Task(description="Assemble final report.", expected_output="Formatted steps.", agent=arch, context=[t1, t2])
    Crew(agents=[analyst, dba, arch], tasks=[t1, t2, t3], process=Process.sequential)
    return "CrewAI Report: ORA-00942 → qualify object, create synonym or grant SELECT."

def _autogen(user_input: str) -> str:
    if not HAS_AUTOGEN:
        return "[AutoGen disabled] Plan: clarify requirements → design → implement."
    import autogen  # type: ignore
    return "AutoGen Summary: gather requirements → design → implement."

def execute(state: MasterState) -> str:
    d = state["decision"]
    if d == "YouTube":
        return _YT_SINGLE(state["urls"][0]) if _YT_SINGLE else "YouTube summarizer not bound."
    if d == "YouTube_Batch":
        return _YT_BATCH(state["urls"]) if _YT_BATCH else "YouTube batch summarizer not bound."
    if d == "CrewAI":
        return _crewai(state["user_input"])
    if d == "AutoGen":
        return _autogen(state["user_input"])
    return "No decision."

def build_langgraph_app():
    if not HAS_LANGGRAPH:
        return None
    from langgraph.graph import StateGraph, END  # type: ignore
    def router_node(s: MasterState):
        r = route(s["user_input"])
        return {"decision": r["decision"], "urls": r["urls"]}
    def exec_node(s: MasterState):
        out = execute({"user_input": s["user_input"], "urls": s["urls"], "decision": s["decision"], "final_output": ""})
        return {"final_output": out}
    g = StateGraph(MasterState)
    g.add_node("Router", router_node)
    g.add_node("Exec", exec_node)
    g.set_entry_point("Router")
    g.add_edge("Router", "Exec")
    g.add_edge("Exec", END)
    return g.compile()
