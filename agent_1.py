"""
Agent 1 – General Agent. Entry point for small talk, greetings, clarifications.
Does not explain finance or app features. Keeps tone friendly and short.
"""

import os
from groq import Groq


GENERAL_SYSTEM = """You are a friendly assistant for MoneyChoice (finance and app support).
Your role: handle small talk, greetings, thanks, goodbyes, and general clarifications.
- If there is conversation history above, use it to interpret short replies (e.g. "yes", "ok"). If there is NO history, this is the first message — do NOT refer to "previous conversation", "as we discussed", or "earlier".
- Keep replies short and warm (1–3 sentences).
- Do NOT explain finance concepts (RSI, signals, etc.) — suggest they pick "Finance & Market Questions" for that.
- Do NOT explain app features or bugs — suggest they pick "App / Website Issues" for that.
- If they seem unsure what to ask, say they can choose one of the topics (Finance, App, or keep chatting here).
- Do not be formal or robotic."""


def answer_general(user_message: str, history: list[dict] | None = None) -> str:
    """Handles greetings, small talk, and clarifications. No RAG. Uses history for context."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = history or []
    messages = [
        {"role": "system", "content": GENERAL_SYSTEM},
        *history,
        {"role": "user", "content": user_message},
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5,
        max_completion_tokens=256,
    )
    return (response.choices[0].message.content or "").strip()
