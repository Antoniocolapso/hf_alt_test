import os
import json
import re
import datetime
from dataclasses import dataclass, field
from typing import List, TypedDict, Annotated

import streamlit as st
import pandas as pd
from openai import OpenAI

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Union Mobile AI Support",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 100%);
    color: white; padding: 20px 30px; border-radius: 10px; margin-bottom: 20px;
}
.badge-verified   { background:#4CAF50; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; font-size:.85em; }
.badge-unverified { background:#f44336; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; font-size:.85em; }
.injection-warning { background:#fff3e0; border-left:4px solid #ff6f00; padding:10px 15px; border-radius:4px; margin:10px 0; }
.output-warning    { background:#fce4ec; border-left:4px solid #c62828; padding:10px 15px; border-radius:4px; margin:10px 0; }
</style>
""", unsafe_allow_html=True)

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MEMORY_FILE = "customer_memory.json"
PLACEHOLDER_NAMES = {"anonymous", "guest", "unknown", "user", "customer", ""}
LARGE_REFUND_THRESHOLD = 50

INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?(instructions|prompts|rules)",
    r"you are now|pretend (you are|to be)|act as (if you are|a)",
    r"system prompt|reveal (your|the) (prompt|instructions|system)",
    r"jailbreak|dan mode|developer mode|unrestricted mode",
    r"forget (everything|all|prior|previous)",
    r"disregard (all |your |previous )?(instructions|rules|guidelines)",
    r"new persona|override (your|all) (rules|instructions|safety)",
    r"\[system\]|<\|system\|>|##SYSTEM|\{\{system\}\}",
    r"print (your|the) (instructions|prompt|system message)",
    r"bypass (safety|content|filter|restriction)",
]

OUTPUT_SAFETY_PATTERNS = [
    r"(confidential|internal|proprietary) (data|information|details)",
    r"(competitor|rival) (is better|outperforms|superior)",
    r"guaranteed|100% (certain|sure|accurate|correct)",
    r"(sue|lawsuit|legal action) (union mobile|the company)",
    r"(free|no charge|complimentary).{0,30}(forever|permanently|always)",
    r"(your data|customer data|account data) (has been|is being) (sold|shared|leaked)",
]

MANAGER_ONLY_OPERATIONS = [
    "suspend", "cancel", "terminate", "delete account",
    "reset pin", "change owner", "transfer ownership"
]


# ─── HELPERS ──────────────────────────────────────────────────────────────────
def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def get_greeting(customer_name: str) -> str:
    """Return personalised greeting or neutral fallback for placeholder names."""
    if customer_name.strip().lower() in PLACEHOLDER_NAMES:
        return "Hello! How can I assist you today?"
    return f"Hello {customer_name}!"

def scan_for_injection(text: str) -> tuple:
    for p in INJECTION_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True, p
    return False, None

def scan_output_safety(text: str) -> tuple:
    for p in OUTPUT_SAFETY_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            return True, p
    return False, None

def detect_billing_tier(query: str) -> str:
    q = query.lower()
    if any(kw in q for kw in ["large refund","full refund","waive all","cancel charges","credit entire"]):
        return "manager_only"
    for amt in re.findall(r'\$([\d,]+)', query):
        try:
            if int(amt.replace(',','')) > LARGE_REFUND_THRESHOLD:
                return "manager_only"
        except ValueError:
            pass
    return "standard"


# ─── OPENAI CLIENT (from Streamlit secrets) ───────────────────────────────────
@st.cache_resource
def get_openai_client():
    """
    Build OpenAI client loading credentials from Streamlit secrets.
    Set OPENAI_API_KEY (required) and OPENAI_API_BASE (optional) in your
    Hugging Face Space secrets or .streamlit/secrets.toml locally.
    """
    return OpenAI()

oai_client = get_openai_client()


# ─── PERSISTENT MEMORY (file-backed, keyed by customer_account_id) ────────────
def load_memory_store() -> dict:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def get_customer_memory(customer_account_id: str, intent_filter: str = None) -> List[dict]:
    """Load memory by stable customer_account_id (NOT conversation_id)."""
    store = load_memory_store()
    all_interactions = store.get(customer_account_id, [])
    if not all_interactions:
        return []
    if intent_filter:
        matching = [i for i in all_interactions if i.get('intent') == intent_filter]
        other    = [i for i in all_interactions if i.get('intent') != intent_filter]
        return (matching[-3:] + other[-2:])[-5:]
    return all_interactions[-5:]

def append_customer_memory(customer_account_id: str, interaction: dict) -> None:
    """Append interaction keyed by stable customer_account_id."""
    store = load_memory_store()
    if customer_account_id not in store:
        store[customer_account_id] = []
    store[customer_account_id].append(interaction)
    with open(MEMORY_FILE, 'w') as f:
        json.dump(store, f, indent=2)

def format_memory_context(memory: List[dict]) -> str:
    if not memory:
        return "No previous interactions on record."
    lines = ["=== Previous Interactions ==="]
    for m in memory:
        lines.append(
            f"[{m.get('timestamp','')[:10]}] {m.get('intent','').upper()} — "
            f"{m.get('agent_used','')} — {m.get('resolution_type','')}\n"
            f"  Q: {m.get('query','')[:80]}\n"
            f"  A: {m.get('response_summary','')[:120]}"
        )
    return "\n".join(lines)


# ─── IN-MEMORY CONVERSATION STORE ─────────────────────────────────────────────
# Keyed by customer_account_id (or session fallback). Persists for the lifetime
# of the Streamlit server process — future turns in the same session automatically
# receive full conversation context without re-reading from disk.

def _conv_store() -> dict:
    """Return (and lazily create) the process-level conversation store."""
    if "conv_store" not in st.session_state:
        st.session_state["conv_store"] = {}
    return st.session_state["conv_store"]

def get_conv_history(session_key: str) -> List[dict]:
    """Return list of {role, content} dicts for this session."""
    return _conv_store().get(session_key, [])

def append_conv_turn(session_key: str, role: str, content: str) -> None:
    """Append a single turn to the in-memory conversation history."""
    store = _conv_store()
    if session_key not in store:
        store[session_key] = []
    store[session_key].append({"role": role, "content": content})

def clear_conv_history(session_key: str) -> None:
    _conv_store().pop(session_key, None)

def _session_key() -> str:
    """Use stable customer_account_id when verified, else fallback session id."""
    acct = st.session_state.get("customer_account_id", "")
    return acct if acct else st.session_state.get("_fallback_session_id", "anon")


# ─── DATASET ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset() -> pd.DataFrame:
    """
    Load the enriched dataset saved by the notebook.
    Contains real customer names, PINs, account IDs from curated conversations.
    """
    if os.path.exists("df_enriched.csv"):
        df = pd.read_csv("df_enriched.csv")
        return df
    else:
        st.error("⚠️ df_enriched.csv not found. Please run the Jupyter notebook first.")
        st.stop()


# ─── CONTEXT RETRIEVAL ────────────────────────────────────────────────────────
def retrieve_context(query: str, intent: str, df: pd.DataFrame, n: int = 2) -> str:
    """Per-agent domain-specific RAG retrieval (Feedback #4)."""
    matches = df[df['intent_category'] == intent].copy()
    if matches.empty:
        return "No relevant examples found."
    query_words = set(query.lower().split())
    matches['relevance'] = matches['full_text'].apply(
        lambda t: len(query_words & set(str(t).lower().split()))
    )
    top = matches.nlargest(n, 'relevance')
    parts = [f"[{intent}/{r['resolution_type']}]\n{str(r['full_text'])[:400]}"
             for _, r in top.iterrows()]
    return "\n\n---\n\n".join(parts)


# ─── SESSION STATE INIT ───────────────────────────────────────────────────────
def init_session():
    import uuid
    defaults = {
        "messages": [],
        "verified": False,
        "customer_name": "",
        "conversation_id": "",
        "customer_account_id": "",
        "account_pin_confirmed": False,
        "decision_log": [],
        "injection_warned": False,
        "output_warned": False,
        # Stable fallback key for anonymous sessions
        "_fallback_session_id": str(uuid.uuid4()),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()
df = load_dataset()


# ─── LANGGRAPH STATE ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # Inputs
    query: str
    customer_name: str
    ver_status: str
    acct_id: str
    pin_confirmed: bool
    greeting: str
    history_str: str       # Multi-turn conversation history as formatted string
    memory_ctx: str        # Formatted past interactions from persistent store
    context: str           # RAG context
    intent: str

    # Outputs (written by nodes)
    response_text: str
    agent_display: str
    resolution: str
    injection_flag: bool
    output_flagged: bool
    matched_injection: str
    matched_output: str

    # Audit side-effects (accumulated across nodes)
    extra_log_entries: List[dict]


# ─── LANGGRAPH NODES ──────────────────────────────────────────────────────────

def input_guardrail_node(state: AgentState) -> AgentState:
    """Node 1 — detect prompt injection."""
    flagged, matched = scan_for_injection(state["query"])
    state["injection_flag"] = flagged
    state["matched_injection"] = matched or ""
    if flagged:
        state["response_text"] = (
            "Your request has been flagged for security review. "
            "A human agent will assist you shortly."
        )
        state["agent_display"]  = "🛡️ Security System"
        state["resolution"]     = "blocked"
        state["output_flagged"] = False
    return state


def supervisor_node(state: AgentState) -> AgentState:
    """Node 2 — classify intent."""
    if state["injection_flag"]:
        state["intent"] = "guardrail"
        return state

    # Build a short representation of recent history for intent classification
    recent_history = state.get("history_str", "")
    last_lines = "\n".join(recent_history.split("\n")[-6:]) if recent_history else ""

    intent_prompt = f"""Classify into one word: network, billing, account, or escalation.
Customer: {state['customer_name']} | Status: {state['ver_status']}
Recent chat: {last_lines[:200]}
Query: {state['query']}
Reply with ONE word only."""

    resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0, max_tokens=10
    )
    intent = resp.choices[0].message.content.strip().lower()
    state["intent"] = intent if intent in {"network","billing","account","escalation"} else "network"
    state["extra_log_entries"].append({
        "timestamp": utc_now(), "node": "SupervisorAgent",
        "customer_name": state["customer_name"],
        "verification_status": state["ver_status"],
        "query": state["query"][:100], "intent_category": state["intent"],
        "agent_selected": f"{state['intent'].capitalize()}Agent",
        "injection_flag": False, "resolution_type": "routing",
        "response_summary": f"Routed to {state['intent']}"
    })
    return state


def network_agent_node(state: AgentState) -> AgentState:
    """Node 3a — network troubleshooting specialist."""
    prompt = f"""You are the Network Support Specialist at Union Mobile.
Begin with: "{state['greeting']}"
Provide specific troubleshooting steps.

Conversation history:
{state['history_str']}
Customer memory: {state['memory_ctx']}
Current query: {state['query']}
Knowledge base: {state['context']}"""

    r = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3, max_tokens=400
    )
    state["response_text"] = r.choices[0].message.content.strip()
    state["agent_display"]  = "🔧 Network Agent"
    state["resolution"]     = "troubleshoot"
    return state


def billing_agent_node(state: AgentState) -> AgentState:
    """Node 3b — billing specialist with tier-based RBAC."""
    greeting   = state["greeting"]
    ver_status = state["ver_status"]

    if ver_status != "verified":
        state["response_text"] = (
            f"{greeting} Billing information requires identity verification. "
            "Please log in with your name and account PIN in the sidebar."
        )
        state["resolution"] = "blocked"
    else:
        billing_tier = detect_billing_tier(state["query"])
        if billing_tier == "manager_only":
            state["response_text"] = (
                f"{greeting} This refund request exceeds the standard agent limit. "
                "Large refunds require senior billing manager approval. "
                "Your case is being escalated — a manager will contact you within 24 hours."
            )
            state["resolution"] = "escalate"
            state["extra_log_entries"].append({
                "timestamp": utc_now(), "node": "BillingAgent",
                "customer_name": state["customer_name"],
                "verification_status": ver_status,
                "query": state["query"][:100], "intent_category": "billing",
                "agent_selected": "BillingAgent", "injection_flag": False,
                "resolution_type": "escalate",
                "response_summary": "Large refund escalated to manager",
                "AUDIT_FLAG": "LARGE_REFUND_ESCALATED"
            })
        else:
            prompt = f"""You are the Billing Specialist at Union Mobile.
Begin with: "{greeting}"
Customer is verified. Answer billing question precisely.

Conversation history:
{state['history_str']}
Customer memory: {state['memory_ctx']}
Current query: {state['query']}
Knowledge base: {state['context']}"""
            r = oai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=400
            )
            state["response_text"] = r.choices[0].message.content.strip()
            state["resolution"] = (
                "refund"
                if any(w in state["response_text"].lower() for w in ["refund","credit","reimburse"])
                else "inform"
            )

    state["agent_display"] = "💰 Billing Agent"
    return state


def account_agent_node(state: AgentState) -> AgentState:
    """Node 3c — account management specialist."""
    greeting   = state["greeting"]
    ver_status = state["ver_status"]

    if ver_status != "verified":
        state["response_text"] = f"{greeting} Account management requires identity verification first."
        state["resolution"]    = "blocked"
    elif any(op in state["query"].lower() for op in MANAGER_ONLY_OPERATIONS) and not state["pin_confirmed"]:
        state["response_text"] = (
            f"{greeting} This operation requires senior manager authorisation. "
            "I'm connecting you with a senior manager now."
        )
        state["resolution"] = "escalate"
    else:
        prompt = f"""You are the Account Management Specialist at Union Mobile.
Begin with: "{greeting}"
Customer is verified. Help with their account request.

Conversation history:
{state['history_str']}
Customer memory: {state['memory_ctx']}
Current query: {state['query']}
Knowledge base: {state['context']}"""
        r = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=400
        )
        state["response_text"] = r.choices[0].message.content.strip()
        state["resolution"]    = "inform"

    state["agent_display"] = "👤 Account Agent"
    return state


def escalation_agent_node(state: AgentState) -> AgentState:
    """Node 3d — escalation with handoff packet generation."""
    greeting = state["greeting"]
    acct_id  = state["acct_id"]

    summary_prompt = f"""Create escalation handoff (100 words): customer {state['customer_name']}, account {acct_id}.
Issue: {state['query']}
History: {state['history_str']}
Include CUSTOMER, ISSUE, ATTEMPTS, REASON, URGENCY."""

    sum_r = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": summary_prompt}],
        temperature=0.2, max_tokens=200
    )
    esc_summary = sum_r.choices[0].message.content.strip()

    # Save escalation packet to persistent memory
    if acct_id:
        append_customer_memory(acct_id, {
            "timestamp": utc_now(), "query": state["query"][:200], "intent": "escalation",
            "agent_used": "EscalationAgent", "resolution_type": "escalate",
            "response_summary": esc_summary[:300], "escalation_packet": esc_summary
        })

    state["response_text"] = (
        f"{greeting} I understand your frustration. I'm escalating your case "
        "to a senior specialist with full context. You'll receive a follow-up within 24 hours."
    )
    state["agent_display"] = "🚨 Escalation Team"
    state["resolution"]    = "escalate"

    state["extra_log_entries"].append({
        "timestamp": utc_now(), "node": "EscalationAgent_HANDOFF",
        "customer_name": state["customer_name"],
        "verification_status": state["ver_status"],
        "query": state["query"][:100], "intent_category": "escalation",
        "agent_selected": "EscalationAgent", "injection_flag": False,
        "resolution_type": "escalate",
        "response_summary": state["response_text"][:100],
        "ESCALATION_SUMMARY": esc_summary
    })
    return state


def output_guardrail_node(state: AgentState) -> AgentState:
    """Node 4 — output safety scan."""
    if state["injection_flag"]:
        state["output_flagged"]   = False
        state["matched_output"]   = ""
        return state

    out_flagged, out_pattern = scan_output_safety(state["response_text"])
    state["output_flagged"] = out_flagged
    state["matched_output"] = out_pattern or ""

    if out_flagged:
        state["response_text"] = (
            "I appreciate your patience. Let me connect you with a specialist "
            "who can provide accurate information for your request."
        )
        state["extra_log_entries"].append({
            "timestamp": utc_now(), "node": "OutputGuardrailNode",
            "customer_name": state["customer_name"],
            "verification_status": state["ver_status"],
            "query": state["query"][:100], "intent_category": state["intent"],
            "agent_selected": "OutputGuardrailNode", "injection_flag": False,
            "resolution_type": "blocked",
            "response_summary": f"Output violation: {out_pattern}"
        })
    return state


# ─── ROUTING FUNCTIONS ────────────────────────────────────────────────────────

def route_after_guardrail(state: AgentState) -> str:
    return "end" if state["injection_flag"] else "supervisor"

def route_after_supervisor(state: AgentState) -> str:
    intent_map = {
        "network":    "network_agent",
        "billing":    "billing_agent",
        "account":    "account_agent",
        "escalation": "escalation_agent",
    }
    return intent_map.get(state["intent"], "network_agent")


# ─── BUILD LANGGRAPH ──────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("input_guardrail",   input_guardrail_node)
    g.add_node("supervisor",        supervisor_node)
    g.add_node("network_agent",     network_agent_node)
    g.add_node("billing_agent",     billing_agent_node)
    g.add_node("account_agent",     account_agent_node)
    g.add_node("escalation_agent",  escalation_agent_node)
    g.add_node("output_guardrail",  output_guardrail_node)

    g.set_entry_point("input_guardrail")

    g.add_conditional_edges("input_guardrail", route_after_guardrail, {
        "end":        "output_guardrail",
        "supervisor": "supervisor",
    })
    g.add_conditional_edges("supervisor", route_after_supervisor, {
        "network_agent":    "network_agent",
        "billing_agent":    "billing_agent",
        "account_agent":    "account_agent",
        "escalation_agent": "escalation_agent",
    })
    for specialist in ["network_agent","billing_agent","account_agent","escalation_agent"]:
        g.add_edge(specialist, "output_guardrail")
    g.add_edge("output_guardrail", END)

    return g.compile()

agent_graph = build_graph()


# ─── CORE PIPELINE ────────────────────────────────────────────────────────────

def process_message(query: str) -> dict:
    """
    Prepare state, run LangGraph, persist memory, update in-memory conv history.
    """
    timestamp     = utc_now()
    customer_name = st.session_state.customer_name or "Guest"
    ver_status    = "verified" if st.session_state.verified else "unverified"
    acct_id       = st.session_state.customer_account_id
    pin_confirmed = st.session_state.account_pin_confirmed
    greeting      = get_greeting(customer_name)
    sess_key      = _session_key()

    # ── Load in-memory conversation history ──────────────────────────────────
    conv_history = get_conv_history(sess_key)   # [{role, content}, ...]
    history_str = ""
    for turn in conv_history[-6:]:
        history_str += f"{turn['role'].upper()}: {turn['content']}\n"

    # ── Intent-filtered persistent memory (pre-fetch; supervisor refines) ────
    # We do a quick intent guess here only to pre-filter memory; supervisor
    # will re-classify with full context and overwrite state["intent"].
    memory     = get_customer_memory(acct_id) if acct_id else []
    memory_ctx = format_memory_context(memory)

    # ── RAG context (intent refined after supervisor runs inside graph) ──────
    # We pass a generic context here; per-agent RAG is applied in each node
    # if you want to keep it truly per-agent, move retrieve_context inside each
    # agent node. Here we fetch after supervisor classification to stay consistent
    # with the original design.
    # For graph simplicity we retrieve once in process_message after the fact;
    # the nodes receive this context via state.
    # (A separate pre-supervisor fetch would require two graph passes.)
    context = ""   # Nodes will receive "" on first pass; see post-supervisor hook below.

    # Build initial state
    initial_state: AgentState = {
        "query":           query,
        "customer_name":   customer_name,
        "ver_status":      ver_status,
        "acct_id":         acct_id,
        "pin_confirmed":   pin_confirmed,
        "greeting":        greeting,
        "history_str":     history_str,
        "memory_ctx":      memory_ctx,
        "context":         context,
        "intent":          "",
        "response_text":   "",
        "agent_display":   "🤖 Support Agent",
        "resolution":      "inform",
        "injection_flag":  False,
        "output_flagged":  False,
        "matched_injection": "",
        "matched_output":  "",
        "extra_log_entries": [],
    }

    # ── Run LangGraph ────────────────────────────────────────────────────────
    # We use a two-step approach: run supervisor first to get intent, then
    # inject RAG context before specialist node runs.
    # LangGraph executes the full graph in one call; we enrich context by
    # running a lightweight intent-only pass first, then the full graph.

    # Step 1: get intent (lightweight)
    intent_prompt = f"""Classify into one word: network, billing, account, or escalation.
Customer: {customer_name} | Status: {ver_status}
Recent chat: {history_str[-200:]}
Query: {query}
Reply with ONE word only."""
    intent_resp = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": intent_prompt}],
        temperature=0, max_tokens=10
    )
    pre_intent = intent_resp.choices[0].message.content.strip().lower()
    pre_intent = pre_intent if pre_intent in {"network","billing","account","escalation"} else "network"

    # Step 2: fetch RAG context and intent-filtered memory now that we know intent
    context    = retrieve_context(query, pre_intent, df)
    memory     = get_customer_memory(acct_id, intent_filter=pre_intent) if acct_id else []
    memory_ctx = format_memory_context(memory)

    initial_state["context"]    = context
    initial_state["memory_ctx"] = memory_ctx

    # Step 3: run full LangGraph (supervisor will re-classify; result is authoritative)
    final_state = agent_graph.invoke(initial_state)

    intent       = final_state["intent"]
    response_txt = final_state["response_text"]
    agent_name   = final_state["agent_display"]
    resolution   = final_state["resolution"]
    inj_flag     = final_state["injection_flag"]
    out_flagged  = final_state["output_flagged"]

    # ── Flush extra log entries produced inside nodes ────────────────────────
    for entry in final_state.get("extra_log_entries", []):
        st.session_state.decision_log.append(entry)

    # ── Log final decision ───────────────────────────────────────────────────
    st.session_state.decision_log.append({
        "timestamp": timestamp,
        "node": f"{intent.capitalize()}Agent" if not inj_flag else "GuardrailNode",
        "customer_name": customer_name,
        "verification_status": ver_status,
        "query": query[:100],
        "intent_category": intent,
        "agent_selected": agent_name,
        "injection_flag": inj_flag,
        "output_flagged": out_flagged,
        "resolution_type": resolution,
        "response_summary": response_txt[:100],
    })

    # ── Update in-memory conversation history ────────────────────────────────
    append_conv_turn(sess_key, "user", query)
    append_conv_turn(sess_key, "assistant", response_txt)

    # ── Save to persistent customer memory ──────────────────────────────────
    if acct_id and resolution not in ["blocked"] and intent not in ["escalation", "guardrail"]:
        append_customer_memory(acct_id, {
            "timestamp": timestamp, "query": query[:200], "intent": intent,
            "agent_used": agent_name, "resolution_type": resolution,
            "response_summary": response_txt[:200]
        })

    # ── Set warning flags ────────────────────────────────────────────────────
    if inj_flag:
        st.session_state.injection_warned = True
    if out_flagged:
        st.session_state.output_warned = True

    return {
        "response": response_txt,
        "agent_name": agent_name,
        "intent": intent,
        "resolution_type": resolution,
        "injection_flag": inj_flag,
        "output_flagged": out_flagged,
    }


# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1 style="margin:0;font-size:1.8em;">📱 Union Mobile AI Customer Support</h1>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.divider()
    st.markdown("## 🔐 Customer Login")

    conv_options = [""] + list(df['conversation_id'].unique())
    sel_conv = st.selectbox("Select Account ID", conv_options)
    if sel_conv:
        st.session_state.conversation_id = sel_conv

    inp_name = st.text_input("Customer Name", placeholder="Enter your full name")
    inp_pin  = st.text_input("Account PIN", type="password", placeholder="4-digit PIN")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Verify Identity", type="primary", use_container_width=True):
            if not sel_conv:
                st.error("Select an Account ID")
            elif not inp_name or not inp_pin:
                st.error("Enter name and PIN")
            else:
                rec = df[df['conversation_id'] == sel_conv]
                if rec.empty:
                    st.error("Account not found")
                else:
                    r = rec.iloc[0]
                    name_ok = inp_name.strip().lower() == str(r['customer_name']).strip().lower()
                    pin_ok  = inp_pin.strip() == str(r['account_pin'])
                    if name_ok and pin_ok:
                        prev_acct = st.session_state.get("customer_account_id", "")
                        st.session_state.verified             = True
                        st.session_state.customer_name        = r['customer_name']
                        st.session_state.customer_account_id  = str(r['customer_account_id'])
                        st.session_state.account_pin_confirmed = True
                        # Migrate anonymous conv history to the verified account key
                        anon_key = st.session_state.get("_fallback_session_id", "")
                        if anon_key and anon_key in _conv_store():
                            existing = _conv_store().pop(anon_key, [])
                            acct_key = str(r['customer_account_id'])
                            _conv_store().setdefault(acct_key, [])
                            _conv_store()[acct_key] = existing + _conv_store()[acct_key]
                        st.success("✅ Verified!")
                        st.rerun()
                    else:
                        st.error("Name or PIN incorrect")

    with col2:
        if st.button("🔓 Guest Mode", use_container_width=True):
            st.session_state.verified             = False
            st.session_state.customer_name        = inp_name or "Guest"
            st.session_state.customer_account_id  = ""
            st.session_state.account_pin_confirmed = False
            st.info("Guest mode")
            st.rerun()

    # Status badge
    st.markdown("### Status")
    if st.session_state.verified:
        st.markdown(f'<span class="badge-verified">✅ VERIFIED — {st.session_state.customer_name}</span>',
                    unsafe_allow_html=True)
        st.caption(f"Account ID: {st.session_state.customer_account_id}")
    else:
        st.markdown('<span class="badge-unverified">❌ NOT VERIFIED</span>', unsafe_allow_html=True)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        if st.button("🔄 Reset Session", use_container_width=True):
            sess_key = _session_key()
            clear_conv_history(sess_key)
            for k in ["messages","verified","customer_name","conversation_id",
                      "customer_account_id","account_pin_confirmed","decision_log",
                      "injection_warned","output_warned","conv_store"]:
                st.session_state.pop(k, None)
            init_session()
            st.rerun()
    with col4:
        if st.session_state.customer_account_id:
            if st.button("🗑️ Clear History", use_container_width=True):
                store = load_memory_store()
                store.pop(st.session_state.customer_account_id, None)
                with open(MEMORY_FILE,'w') as f:
                    json.dump(store, f, indent=2)
                clear_conv_history(st.session_state.customer_account_id)
                st.success("Cleared")

    # Test credentials
    st.markdown("### 💡 Test Credentials")
    if not df.empty:
        s = df.iloc[0]
        st.code(f"ID:   {s['conversation_id']}\nName: {s['customer_name']}\nPIN:  {s['account_pin']}")


# ── MAIN AREA ─────────────────────────────────────────────────────────────────
col_chat, col_info = st.columns([2, 1])

with col_chat:
    # Security warning banners
    if st.session_state.injection_warned:
        st.markdown('<div class="injection-warning">⚠️ <b>Input Security Alert:</b> A prompt injection attempt was detected and blocked.</div>',
                    unsafe_allow_html=True)
    if st.session_state.output_warned:
        st.markdown('<div class="output-warning">🔴 <b>Output Safety Alert:</b> A generated response was intercepted and replaced for policy compliance.</div>',
                    unsafe_allow_html=True)

    st.markdown("### 💬 Chat")

    # Welcome message
    if not st.session_state.messages:
        if st.session_state.verified:
            welcome = (f"Hello {st.session_state.customer_name}! I'm your Union Mobile AI support assistant. "
                       "I can help with network issues, billing questions, and account management.")
        else:
            welcome = ("Welcome to Union Mobile support! I can help with general queries. "
                       "For billing and account access, please verify your identity using the login panel.")
        st.session_state.messages.append({
            "role": "assistant", "content": welcome, "agent": "🤖 Support Assistant",
            "timestamp": utc_now()
        })

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("agent"):
                st.caption(msg["agent"])
            st.write(msg["content"])

    # Input
    if user_input := st.chat_input("Type your message here..."):
        st.session_state.messages.append({
            "role": "user", "content": user_input, "timestamp": utc_now()
        })
        with st.spinner("Connecting to the right agent..."):
            try:
                result = process_message(user_input)
                st.session_state.messages.append({
                    "role": "assistant", "content": result["response"],
                    "agent": result["agent_name"], "timestamp": utc_now()
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Technical issue. Please try again. ({str(e)[:80]})",
                    "agent": "⚠️ System", "timestamp": utc_now()
                })
        st.rerun()


with col_info:
    # ── INTERACTION HISTORY ──
    st.markdown("### 📋 Interaction History")
    with st.expander("View Past Interactions", expanded=False):
        acct = st.session_state.customer_account_id
        if acct:
            memory = get_customer_memory(acct)
            if memory:
                for m in reversed(memory):
                    ts = m.get('timestamp','')[:10]
                    if m.get('escalation_packet'):
                        st.markdown(f"**{ts}** — 🚨 ESCALATION")
                        st.text_area("Handoff Packet", m['escalation_packet'], height=100, disabled=True)
                    else:
                        st.markdown(f"**{ts}** — _{m.get('intent','').upper()}_\n"
                                    f"- 🤖 {m.get('agent_used','')}\n"
                                    f"- ✓ {m.get('resolution_type','')}\n"
                                    f"- 💬 _{m.get('query','')[:60]}..._\n---")
            else:
                st.info("No past interactions found.")
        else:
            st.info("Verify your identity to view history.")

    # ── ACCOUNT INFO ──
    if st.session_state.verified and st.session_state.conversation_id:
        st.markdown("### 👤 Account")
        rec = df[df['conversation_id'] == st.session_state.conversation_id]
        if not rec.empty:
            r = rec.iloc[0]
            with st.expander("Details", expanded=True):
                st.write(f"**Name:** {r['customer_name']}")
                st.write(f"**Account ID:** {r.get('customer_account_id','N/A')}")
                st.write(f"**Access Level:** {r['access_level']}")

    # ── QUICK ACTIONS ──
    st.markdown("### ⚡ Quick Actions")
    actions = {
        "📶 Signal issue":   "My signal keeps dropping. What troubleshooting steps can I take?",
        "💳 Check bill":     "Can you explain why my bill is higher than usual this month?",
        "📱 Change plan":    "I'd like to upgrade to a plan with more data.",
        "🆘 Get help":       "I've had this issue unresolved for three weeks now."
    }
    for label, msg in actions.items():
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({"role":"user","content":msg,"timestamp":utc_now()})
            with st.spinner("Processing..."):
                try:
                    result = process_message(msg)
                    st.session_state.messages.append({
                        "role": "assistant", "content": result["response"],
                        "agent": result["agent_name"], "timestamp": utc_now()
                    })
                except Exception as e:
                    st.error(str(e)[:80])
            st.rerun()

    # ── CONVERSATION CONTEXT INDICATOR ──
    sess_key = _session_key()
    turns = len(get_conv_history(sess_key)) // 2
    if turns > 0:
        st.markdown("### 🔄 Conversation")
        st.metric("Turns in session", turns)
        st.caption("Full conversation history carried forward each turn via in-memory store.")


# ── DECISION LOG ──────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📊 Decision Log (Audit Trail)")
with st.expander("View Full Decision Log", expanded=False):
    if st.session_state.decision_log:
        COLORS = {
            "network":"#E3F2FD","billing":"#E8F5E9","account":"#FFF3E0",
            "escalation":"#FFEBEE","guardrail":"#F3E5F5",
            "identity_check":"#E0F2F1","routing":"#F5F5F5"
        }
        df_log = pd.DataFrame(st.session_state.decision_log)
        ca, cb, cc, cd = st.columns(4)
        ca.metric("Total Nodes", len(df_log))
        cb.metric("Injections Blocked", int(df_log.get('injection_flag', pd.Series([False]*len(df_log))).sum()))
        cc.metric("Output Intercepts",  int(df_log.get('output_flagged', pd.Series([False]*len(df_log))).sum()))
        cd.metric("Escalations",
                  len(df_log[df_log.get('resolution_type', pd.Series()) == 'escalate']) if 'resolution_type' in df_log else 0)

        dcols = ['timestamp','node','customer_name','verification_status',
                 'intent_category','injection_flag','resolution_type','response_summary']
        show = [c for c in dcols if c in df_log.columns]

        def color_row(row):
            bg = COLORS.get(row.get('intent_category',''), '#FFFFFF')
            return [f'background-color:{bg}'] * len(row)

        st.dataframe(df_log[show].style.apply(color_row, axis=1),
                     use_container_width=True, height=300)
        csv = df_log[show].to_csv(index=False)
        st.download_button("⬇️ Download Audit Log (CSV)", data=csv,
                           file_name=f"audit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
        st.markdown("**Colors:** 🔵 Network · 🟢 Billing · 🟠 Account · 🔴 Escalation · 🟣 Guardrail")
    else:
        st.info("No log entries yet. Start a conversation.")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#888;font-size:.8em;padding:10px">
    Union Mobile AI Support — MLS-4 v3 | LangGraph | Streamlit Secrets | In-Memory Conversation Store
    <br>⚠️ Demonstration system only.
</div>
""", unsafe_allow_html=True)
