"""
3_agent.py

WHAT THIS FILE DOES:
Builds a ReAct agent with 3 tools:
  - search_logs          → searches ChromaDB semantically
  - check_metric_spike   → checks memory/CPU/traffic metrics
  - get_deployment_history → checks recent deployments

The agent decides which tools to use and in what order.
It loops Thought → Action → Observation until it has enough
information to write a complete Root Cause Analysis.

REQUIRES:
  - data/chroma_db/ must exist (run 1_ingest_logs.py first)
  - GROQ_API_KEY in .env file
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain import hub
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ── Configuration ──────────────────────────────
CHROMA_DIR  = "data/chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama-3.3-70b-versatile"

# Global vectorstore — loaded once, shared by all tools
_vectorstore = None


def get_vectorstore():
    """
    Loads ChromaDB once and reuses it.
    This is called a singleton pattern —
    we don't want to reload the database on every tool call.
    """
    global _vectorstore

    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name    = EMBED_MODEL,
            model_kwargs  = {"device": "cpu"},
            encode_kwargs = {"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            persist_directory  = CHROMA_DIR,
            embedding_function = embeddings,
            collection_name    = "system_logs",
        )

    return _vectorstore

@tool
def search_logs(query: str) -> str:
    """
    Search through system logs using semantic similarity.
    Use this to find error messages, crash events, stack traces,
    OOMKilled events, database errors, deployment events,
    traffic spikes, or any system activity.

    Input: a natural language description of what you are looking for.
    Output: the most relevant log chunks found.

    Examples of good inputs:
    - "payment service OOMKilled memory crash"
    - "database connection pool exhausted error"
    - "deployment version update rollout"
    - "traffic spike high request rate"
    """
    try:
        vs   = get_vectorstore()
        docs = vs.similarity_search(query, k=5)

        if not docs:
            return "No relevant log entries found for this query."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown").split("/")[-1]
            results.append(
                f"[Chunk {i} | Source: {source}]\\n{doc.page_content}"
            )

        return "\\n\\n".join(results)

    except Exception as e:
        return f"Error searching logs: {e}"
    
@tool
def check_metric_spike(query: str) -> str:
    """
    Check for metric anomalies or spikes for a specific service.
    Use this when you suspect a resource problem or want to
    quantify how severe a memory, CPU, or traffic issue was.

    Input: a single string combining service name and metric type.
    Format: "service_name metric_type"

    Examples of valid inputs:
    - "payment-service memory"
    - "payment-service connections"
    - "payment-service throughput"
    - "payment-service error_rate"
    - "payment-service latency"
    - "postgres-primary connections"
    - "nginx-ingress error_rate"
    """
    METRICS = {
        "payment-service": {
            "memory": (
                "Peak: 910Mi/1Gi (91% of limit). "
                "Baseline was 490Mi. "
                "Grew 82% in 15 minutes (01:44–02:00 UTC). "
                "OOMKilled at 02:00 UTC. "
                "Cause: unbounded ResponseCache introduced in v2.4.0"
            ),
            "throughput": (
                "Peak: 1240 req/s at 01:44 UTC. "
                "Baseline: 380-410 req/s. "
                "That is a 3.1x spike. "
                "Returned to normal after pod restart."
            ),
            "connections": (
                "Peak: 50/50 pool connections used + 112 requests queued. "
                "Pool exhausted at 01:58 UTC. "
                "Cause: traffic spike drove connection demand beyond pool limit."
            ),
            "error_rate": (
                "Peak: 12.4% error rate at 02:00 UTC. "
                "Baseline: under 0.1%. "
                "Caused by OOMKill and 503s from nginx."
            ),
            "latency": (
                "Peak p99: 2340ms at 01:58 UTC. "
                "Threshold: 500ms. "
                "Caused by DB connection pool exhaustion."
            ),
        },
        "postgres-primary": {
            "connections": (
                "Peak: 50/50 hard limit hit at 01:57 UTC. "
                "New connections rejected with FATAL error. "
                "Dropped back to 14 after pod restart at 02:00 UTC."
            ),
        },
        "nginx-ingress": {
            "error_rate": (
                "100% 503 errors from 02:00 to 02:31 UTC. "
                "Circuit breaker opened at 02:00:30. "
                "Closed at 02:31 when service recovered."
            ),
        },
    }

    # Parse the single string into service and metric
    # Handle formats like:
    #   "payment-service memory"
    #   "payment-service, memory"
    #   "payment-service,memory"
    query   = query.replace(",", " ").strip()
    parts   = query.split()

    # Find service name — could be hyphenated like "payment-service"
    # so we join first two words if needed
    svc    = None
    metric = None

    for known_svc in METRICS.keys():
        if known_svc in query.lower():
            svc    = known_svc
            # Whatever is left after removing the service name is the metric
            metric = query.lower().replace(known_svc, "").strip()
            break

    # If still not found, try first word as service, last word as metric
    if not svc and len(parts) >= 2:
        svc    = parts[0].lower()
        metric = parts[-1].lower()

    if not svc or svc not in METRICS:
        available = list(METRICS.keys())
        return (
            f"Service not found. Available services: {available}. "
            f"Format: 'service_name metric_type'"
        )

    if not metric or metric not in METRICS[svc]:
        available = list(METRICS[svc].keys())
        return (
            f"Metric '{metric}' not found for '{svc}'. "
            f"Available metrics: {available}"
        )

    return f"📊 {svc} | {metric}: {METRICS[svc][metric]}"

@tool
def get_deployment_history(service_name: str) -> str:
    """
    Get recent deployment history for a service.
    Use this when you want to check if a recent code change
    or version update may have caused or contributed to an incident.

    Deployments within 6 hours of an incident are high priority suspects.

    Input: service name as a string
    Examples: "payment-service", "postgres-primary"

    Output: recent deployments with versions, timestamps, and change notes.
    """
    # IN PRODUCTION: query ArgoCD, Spinnaker, or Flux:
    #   argocd.get_app_history("payment-service")
    # Or query Kubernetes directly:
    #   kubectl rollout history deployment/payment-service

    HISTORY = {
        "payment-service": [
            {
                "timestamp":    "2024-01-14 23:10 UTC",
                "from_version": "v2.3.1",
                "to_version":   "v2.4.0",
                "status":       "SUCCESS",
                "hours_before_crash": 2.83,
                "changed_files": ["ResponseCache.java", "ConnectionPool config"],
                "notes": (
                    "PR #1847: Added in-memory response caching to improve "
                    "latency. WARNING: Cache max-size was not set — "
                    "unbounded growth possible under high load."
                ),
            }
        ],
        "postgres-primary": [],
    }

    svc = service_name.lower().strip()

    # Handle variations like "payment service" or "payment_service"
    if "payment" in svc:
        svc = "payment-service"
    elif "postgres" in svc or "database" in svc or "db" in svc:
        svc = "postgres-primary"

    if svc not in HISTORY:
        return f"No deployment history found for '{service_name}'."

    deployments = HISTORY[svc]

    if not deployments:
        return f"No recent deployments for '{service_name}'. Database was not changed recently."

    lines = [f"🚀 DEPLOYMENT HISTORY: {service_name}"]
    lines.append("─" * 40)

    for d in deployments:
        lines.append(f"Time:          {d['timestamp']}")
        lines.append(f"Version:       {d['from_version']} → {d['to_version']}")
        lines.append(f"Status:        {d['status']}")
        lines.append(f"Hours before incident: {d['hours_before_crash']}")
        lines.append(f"Changed files: {', '.join(d['changed_files'])}")
        lines.append(f"Notes:         {d['notes']}")

    return "\\n".join(lines)

def build_agent():
    """
    Creates the ReAct agent and wraps it in an AgentExecutor.

    Three components needed:
    1. LLM        — the brain that decides what to do
    2. Tools      — the actions it can take
    3. Prompt     — teaches it the Thought/Action/Observation format
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    # The brain
    llm = ChatGroq(
        model       = LLM_MODEL,
        temperature = 0,
        api_key     = api_key,
    )

    # The tools
    tools = [search_logs, check_metric_spike, get_deployment_history]

    # The ReAct prompt — teaches Thought/Action/Observation format
    # Pulled from LangChain Hub (cached after first download)
    print("📥 Pulling ReAct prompt from LangChain Hub...")
    react_prompt = hub.pull("hwchase17/react")
    print("  ✅ Prompt ready.")

    # Wire LLM + tools + prompt into a ReAct agent
    agent = create_react_agent(
        llm    = llm,
        tools  = tools,
        prompt = react_prompt,
    )

    # Wrap in AgentExecutor — this runs the actual loop
    agent_executor = AgentExecutor(
        agent                     = agent,
        tools                     = tools,
        verbose                   = True,   # prints every Thought/Action/Observation
        max_iterations            = 8,      # safety limit — stops infinite loops
        handle_parsing_errors     = True,   # recovers gracefully from LLM format mistakes
        return_intermediate_steps = True,   # gives us the full reasoning trace
    )

    return agent_executor

def run_agent(question: str, agent_executor) -> dict:
    """
    Runs the agent on a question and returns structured output.
    """
    print(f"\\n{'─' * 50}")
    print(f"Question: {question}")
    print(f"{'─' * 50}\\n")

    result = agent_executor.invoke({"input": question})

    # Extract tool calls from intermediate steps
    steps = []
    for action, observation in result.get("intermediate_steps", []):
        steps.append({
            "tool":        action.tool,
            "input":       action.tool_input,
            "observation": str(observation),
        })

    return {
        "question":     question,
        "final_answer": result["output"],
        "steps":        steps,
        "num_steps":    len(steps),
        "tools_used":   [s["tool"] for s in steps],
    }

def main():
    print("=" * 50)
    print("  ReAct Agent — Multi-Step RCA")
    print("=" * 50)

    print("\\nBuilding agent...")
    agent = build_agent()

    # A complex multi-part question that benefits from multiple tool calls
    question = (
        "Why did the payment service fail at 2am on January 15th? "
        "Check if there was a recent deployment, look at the memory metrics, "
        "and give me a complete root cause analysis with remediation steps."
    )

    result = run_agent(question, agent)

    print("\\n" + "=" * 50)
    print("📊 AGENT SUMMARY")
    print("=" * 50)
    print(f"Tool calls made: {result['num_steps']}")
    print(f"Tools used:      {result['tools_used']}")

    print("\\n" + "=" * 50)
    print("🤖 FINAL ROOT CAUSE ANALYSIS:")
    print("=" * 50)
    print(result["final_answer"])
    print("=" * 50)


if __name__ == "__main__":
    main()