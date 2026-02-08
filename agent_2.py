"""
Agent 2 – Finance RAG Agent. Explains indicators, signals, market logic.
Uses finance_rag for context (PDF + fallback keyword knowledge).
"""

import os

from groq import Groq

from finance_rag import retrieve as retrieve_finance_context


def _last_user_message(history: list[dict]) -> str:
    """Get the most recent user message from history for RAG when current message is short."""
    for m in reversed(history):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


# Phrases that mean "expand on what you just said" — for these we do NOT inject RAG, so the model stays on the same topic.
FOLLOW_UP_PHRASES = (
    "yes", "no", "y", "n", "go deeper", "deeper", "example", "please",
    "explain me further", "explain further", "tell me more", "more", "expand",
    "and?", "what else?", "continue", "go on", "elaborate", "break it down",
    "can you explain more", "explain more", "further", "in more detail",
    "more detail", "clarify", "could you elaborate",
)


def _is_follow_up(user_message: str, has_history: bool) -> bool:
    """True if the user is asking to expand on the previous answer (do not inject unrelated RAG)."""
    if not has_history:
        return False
    msg = user_message.strip().lower()
    if len(msg) > 40:
        return False
    return msg in FOLLOW_UP_PHRASES or any(phrase in msg for phrase in ("explain further", "tell me more", "go deeper", "more detail", "elaborate"))


def answer_finance(user_message: str, history: list[dict] | None = None) -> str:
    """
    Uses conversation history + current message. For follow-ups like "explain me further"
    we do NOT inject RAG (so the model expands on the same topic). For new questions we use RAG.
    """
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = history or []
    has_history = len(history) > 0
    is_follow_up = _is_follow_up(user_message, has_history)

    if is_follow_up:
        # Do NOT add RAG context — it often pulls in a different topic (e.g. RSI) and the model switches. Rely only on conversation above.
        user_content = f"""The user is asking you to expand on what you just explained. Stay on the SAME topic you were discussing. Do not switch to a different concept (e.g. if you were explaining long vs short, give more detail on long vs short; do not start talking about RSI or MACD).

Current message from user: {user_message}"""
    else:
        # New or specific question: use RAG. For short messages use last user question for retrieval.
        query_for_rag = user_message
        if len(user_message.strip()) < 25 or user_message.strip().lower() in FOLLOW_UP_PHRASES:
            last_user = _last_user_message(history)
            if last_user:
                query_for_rag = last_user
        chunks = retrieve_finance_context(query_for_rag, top_k=4)
        context = "\n\n".join(chunks) if chunks else "No specific context found; answer from general finance knowledge."
        if has_history:
            user_content = f"""Relevant context from our finance docs (use if needed):\n{context}\n\nConversation so far is above. Current message from user: {user_message}"""
        else:
            user_content = f"""Relevant context from our finance docs (use if needed):\n{context}\n\nUser message: {user_message}"""

    system = """You are a finance explainer, not a trader.

    RULES FOR ANSWERS:
    - Answer in **3 to 4 short lines maximum**
    - Each line must be **one sentence only**
    - Do NOT write long paragraphs
    - Prefer bullet points or clearly separated lines
    - Be clear, direct, and beginner-friendly
    - Explain *why* briefly, not in depth

    CONTENT RULES:
    - Explain indicators, signals, and market logic only
    - When "Relevant context" is provided, stay grounded in it
    - Do NOT invent facts
    - Do NOT discuss apps, UI, buttons, bugs, or navigation

    FOLLOW-UP RULE:
    - If the user asks "explain further", "tell me more", "go deeper", etc.,
    continue the SAME topic only.
    - Add clarity or an example, but still stay within 3–4 lines.
    """

    if has_history:
        system += """
- Use the conversation history above. If the user says "explain further", "tell me more", "go deeper", or similar, they want MORE detail on the topic YOU JUST EXPLAINED. Continue that same topic. Do NOT introduce a new topic (e.g. do not talk about RSI/MACD if you were explaining long vs short)."""
    else:
        system += """
- This is the first message in the conversation. Do NOT refer to "previous conversation", "earlier", "as we discussed", or "based on our previous chat" — there is no prior context."""

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
