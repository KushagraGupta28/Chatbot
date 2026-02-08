"""
Agent 3 – App / Product RAG Agent. Explains features, issues, what to do next.
Does NOT explain finance concepts. Uses app-only knowledge.
"""

import os
from groq import Groq


# App-only knowledge base. Feature docs, common issues, UI, errors. No finance content.
APP_KNOWLEDGE = [

    {
        "text": (
            "Add portfolio: Use the 'Add portfolio' button. If the button does not respond immediately, "
            "wait 10–15 seconds — syncing can be slow under load. Refresh the page once. "
            "If the portfolio still doesn’t appear, restart the app. "
            "If the issue continues, log out and log back in, then try again."
        ),
        "keywords": "add portfolio, portfolio, not working, button, sync, restart, login"
    },

    {
        "text": (
            "Portfolio sync delay: Portfolios can take 10–15 seconds to appear after adding. "
            "Refresh the page once. If it still doesn’t show, restart the app to reinitialize sync. "
            "As a final step, log out and log back in — this often resolves stale session issues."
        ),
        "keywords": "portfolio, sync, delay, refresh, not showing, restart, logout"
    },

    {
        "text": (
            "Predictions not showing: Prediction data may fail to load due to temporary sync or session issues. "
            "First, refresh the page and wait a few seconds. If predictions still don’t appear, "
            "restart the app. If the problem persists, log out and log back in, then re-open the portfolio."
        ),
        "keywords": "prediction, predictions not showing, signal missing, forecast, restart, login"
    },

    {
        "text": (
            "Data not updating or stuck: Updates are near real-time but may lag during peak usage. "
            "Refresh once and wait 10 seconds. If data remains unchanged, restart the app. "
            "Logging out and logging back in usually clears cached data and restores updates."
        ),
        "keywords": "data not updating, stuck, refresh, lag, restart, logout"
    },

    {
        "text": (
            "Button not responding or UI frozen: This can happen due to a temporary frontend or network issue. "
            "Refresh the page once. If buttons are still unresponsive, restart the app. "
            "If the UI remains frozen, log out and log back in to reset the session."
        ),
        "keywords": "button not responding, ui frozen, click not working, restart, logout"
    },

    {
        "text": (
            "Error messages: If you see an error message, note the exact text and retry once. "
            "If the error persists, restart the app. Logging out and logging back in resolves most session-related errors. "
            "If the same error continues after that, it may be a backend or broker-side issue."
        ),
        "keywords": "error, error message, retry, restart, login, logout"
    },

    {
        "text": (
            "Known limitations: Some brokers enforce rate limits, and we throttle requests to stay compliant. "
            "Large portfolios may take longer to sync. During off-market hours or high traffic, "
            "data and predictions can be delayed. Restarting the app or logging out and back in can help."
        ),
        "keywords": "limit, slow, delay, sync, broker, rate limit"
    },

    {
        "text": (
            "UI clarification: Buy/Sell/Hold labels come from our signal logic. "
            "If you’re confused why a label appears, ask about the signal — that’s a finance question. "
            "If you’re asking where a feature is or how to perform an action, that’s an app usage question."
        ),
        "keywords": "ui, label, buy, sell, hold, where, how"
    },

    {
        "text": (
            "General troubleshooting flow: "
            "1) Refresh once and wait 10–15 seconds. "
            "2) Restart the app if the issue continues. "
            "3) Log out and log back in as a final step. "
            "Most issues resolve by step 3."
        ),
        "keywords": "restart app, logout login, troubleshooting, fix issue"
    }

]



def _retrieve(query: str, top_k: int = 4) -> list[str]:
    q = set(query.lower().split())
    scored = []

    for chunk in APP_KNOWLEDGE:
        keys = set(k.strip() for k in chunk["keywords"].lower().split(","))
        score = len(q & keys) + 0.1 * len(keys)
        scored.append((score, chunk["text"]))

    scored.sort(key=lambda x: -x[0])
    return [text for _, text in scored[:top_k]]



def _last_user_message(history: list[dict]) -> str:
    """Get the most recent user message from history for RAG when current message is short."""
    for m in reversed(history):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


FOLLOW_UP_PHRASES = (
    "yes", "no", "y", "n", "and then?", "what else?", "explain me further", "explain further",
    "tell me more", "more", "go deeper", "continue", "elaborate", "more detail",
)


def _is_follow_up(user_message: str, has_history: bool) -> bool:
    if not has_history:
        return False
    msg = user_message.strip().lower()
    return len(msg) <= 40 and (msg in FOLLOW_UP_PHRASES or any(p in msg for p in ("explain further", "tell me more", "more detail")))


def answer_app(user_message: str, history: list[dict] | None = None) -> str:
    """
    Uses conversation history + current message. For follow-ups like "explain me further"
    we do NOT inject RAG so the model expands on the same topic.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = history or []
    has_history = len(history) > 0
    is_follow_up = _is_follow_up(user_message, has_history)

    if is_follow_up:
        user_content = f"""The user wants you to expand on what you just explained. Stay on the SAME topic (same feature or issue). Do not switch to a different topic.\n\nCurrent message from user: {user_message}"""
    else:
        query_for_rag = user_message
        if len(user_message.strip()) < 25 or user_message.strip().lower() in FOLLOW_UP_PHRASES:
            last_user = _last_user_message(history)
            if last_user:
                query_for_rag = last_user
        chunks = _retrieve(query_for_rag, top_k=4)
        context = "\n\n".join(chunks) if chunks else "No specific context; answer from general product support knowledge."
        if has_history:
            user_content = f"""Relevant context from our app docs (use if needed):\n{context}\n\nConversation so far is above. Current message from user: {user_message}"""
        else:
            user_content = f"""Relevant context from our app docs (use if needed):\n{context}\n\nUser message: {user_message}"""

    system = """You are a product support agent. You explain how features work, why something didn't work, and what the user should do next.
- Be specific, calm, and actionable. Give one clear next step. Avoid generic 'contact support' unless truly needed.
- Do NOT explain finance concepts, indicators, or market logic — only app/product behavior and fixes."""
    if has_history:
        system += """
- Use the conversation history above. If the user says "explain further", "tell me more", etc., they want MORE detail on what YOU JUST said. Continue that same topic. Do not introduce a new one."""
    else:
        system += """
- This is the first message in the conversation. Do NOT refer to "previous conversation", "earlier", or "as we discussed" — there is no prior context."""

    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": user_content},
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.4,
        max_completion_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()
