"""
Orchestrator + Web server. Multi-level routing: agents include finance subcategories.
No intent classifier in this flow — UX decides the agent.
"""

import os
import uuid
from dotenv import load_dotenv

load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from agent_1 import answer_general
from agent_2 import answer_finance
from agent_3 import answer_app
from finance_rag import warmup
from finance_agents import (
    answer_indicators_signals,
    answer_particular_stock,
    answer_market_basics,
    answer_trading_strategies,
    answer_risk_management,
    answer_news_impact,
    answer_investment_options,
    answer_ask_your_own,
)

app = Flask(__name__, static_folder="static", static_url_path="")

# In-memory session history: session_id -> { agent_name -> list of {role, content} }
# Last 6 turns (12 messages) per agent. Wiped when the app process stops.
SESSION_HISTORY: dict[str, dict[str, list[dict]]] = {}
MAX_TURNS = 6

# Finance subcategories
FINANCE_SUBCATEGORIES = {
    "finance_indicators_signals": "Indicators & Signals",
    "finance_particular_stock": "Particular Stock",
    "finance_market_basics": "Market Basics",
    "finance_trading_strategies": "Trading Strategies",
    "finance_risk_management": "Risk Management",
    "finance_news_impact": "News Impact",
    "finance_investment_options": "Investment Options",
    "finance_ask_your_own": "Ask Your Own",
}

# Changes on every server start. Client sends last run_id; if it doesn't match, we use empty history (effective clear after restart).
SERVER_RUN_ID = str(uuid.uuid4())


def _get_history(session_id: str, agent: str) -> list[dict]:
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = {}
    return SESSION_HISTORY[session_id].get(agent, [])


def _append_turn(session_id: str, agent: str, user_message: str, response: str) -> None:
    if session_id not in SESSION_HISTORY:
        SESSION_HISTORY[session_id] = {}
    if agent not in SESSION_HISTORY[session_id]:
        SESSION_HISTORY[session_id][agent] = []
    hist = SESSION_HISTORY[session_id][agent]
    hist.append({"role": "user", "content": user_message})
    hist.append({"role": "assistant", "content": response})
    # Keep only last MAX_TURNS turns (2 messages per turn)
    SESSION_HISTORY[session_id][agent] = hist[-(MAX_TURNS * 2) :]




def run_by_agent(user_message: str, active_agent: str, history: list[dict] | None = None) -> tuple[str, str]:
    """
    Route to the selected agent. Returns (agent_label, response).
    active_agent can be: finance_* (subcategories), app, or general.
    """
    history = history or []
    
    # Finance subcategories routing
    if active_agent == "finance_indicators_signals":
        return "Indicators & Signals Expert", answer_finance(user_message, history=history)
    elif active_agent == "finance_particular_stock":
        return "Stock Analyst", answer_finance(user_message, history=history)
    elif active_agent == "finance_market_basics":
        return "Market Basics Educator", answer_finance(user_message, history=history)
    elif active_agent == "finance_trading_strategies":
        return "Trading Strategies Expert", answer_finance(user_message, history=history)
    elif active_agent == "finance_risk_management":
        return "Risk Management Advisor", answer_finance(user_message, history=history)
    elif active_agent == "finance_news_impact":
        return "News & Impact Analyst", answer_finance(user_message, history=history)
    elif active_agent == "finance_investment_options":
        return "Investment Options Guide", answer_finance(user_message, history=history)
    elif active_agent == "finance_ask_your_own":
        return "Finance Expert", answer_finance(user_message, history=history)
    # Legacy routing
    elif active_agent == "finance":
        return "Finance Expert", answer_finance(user_message, history=history)
    elif active_agent == "app":
        return "App Expert", answer_app(user_message, history=history)
    else:
        return "General", answer_general(user_message, history=history)


@app.route("/")
def index():
    """Serve the chatbot page."""
    return send_from_directory("static", "index.html")


@app.route("/logo.png")
def logo():
    """Serve MoneyChoice logo from project root."""
    root = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(root, "logo.png")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Body: { "message": "...", "active_agent": "finance_*"|"app"|"general", "session_id": "...", "run_id": "..." (optional) }
    Returns: { "response": "...", "agent": "...", "session_id": "...", "run_id": "..." }
    If client's run_id does not match server (e.g. after restart), history is cleared for this request.
    """
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    active_agent = (data.get("active_agent") or "general").lower().strip()
    
    # Validate agent
    valid_agents = list(FINANCE_SUBCATEGORIES.keys()) + ["finance", "app", "general"]
    if active_agent not in valid_agents:
        active_agent = "general"
    
    session_id = (data.get("session_id") or "").strip()
    if not session_id:
        session_id = str(uuid.uuid4())
    client_run_id = (data.get("run_id") or "").strip()
    # After server restart, run_id changes; use empty history so we don't carry over from before restart
    use_history = client_run_id == SERVER_RUN_ID
    history = _get_history(session_id, active_agent) if use_history else []
    if not message:
        return jsonify({"error": "Message is required"}), 400
    try:
        agent_label, response = run_by_agent(message, active_agent, history=history)
        if use_history:
            _append_turn(session_id, active_agent, message, response)
        else:
            # New run: store this turn so next message has context
            _append_turn(session_id, active_agent, message, response)
        return jsonify({
            "response": response,
            "agent": agent_label,
            "session_id": session_id,
            "run_id": SERVER_RUN_ID,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


import os

def main():
    key = os.environ.get("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        raise RuntimeError("GROQ_API_KEY not set")

    print("Warming Finance RAG...")
    warmup()
    print("Finance RAG ready.")

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

