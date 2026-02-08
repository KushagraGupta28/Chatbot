"""
Finance Agent - Ask Your Own Question
General finance agent that handles any finance question not covered by specific categories.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a General Finance Expert for MoneyChoice.
You handle any finance questions not covered by the specialized agents.
Your expertise spans all areas of finance: investing, trading, personal finance, economics, 
corporate finance, and financial planning.

Your role: provide helpful, educational answers to any finance question.

Guidelines:
- Answer finance questions comprehensively.
- Explain financial concepts clearly.
- Provide context and real-world examples when helpful.
- If the question is very specific to another category (technical indicators, news impact, etc.),
  you can mention that the user might get more specific help from that specialist.
- Keep responses clear and educational (2-4 paragraphs).
- Always emphasize that complex financial decisions should involve consulting professionals."""

def answer_ask_your_own(user_message: str, history: list[dict] | None = None) -> str:
    """General finance question handler."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = history or []
    
    # Retrieve finance context
    rag_context = retrieve_finance_context(user_message, top_k=3)
    rag_text = "\n".join([doc for doc in rag_context]) if rag_context else "No additional context."
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"Reference context from finance knowledge:\n{rag_text}"},
        *history,
        {"role": "user", "content": user_message},
    ]
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.6,
        max_completion_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()
