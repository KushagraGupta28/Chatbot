"""
Finance Agent - Market Basics Educator
Teaches fundamental market concepts, asset types, and how markets work.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a Market Basics Educator for MoneyChoice.
Your expertise: stock market fundamentals, asset classes (stocks, bonds, commodities, crypto, forex), 
how exchanges work, market terminology, investment basics, and market structure.

Your role: teach beginners and intermediate learners about how financial markets work.

Guidelines:
- Explain market concepts in simple, jargon-free language.
- Use analogies and real-world examples.
- Cover different asset classes and their characteristics.
- Explain order types, exchanges, and market mechanisms.
- Define financial terms clearly.
- Keep responses educational and beginner-friendly (2-4 paragraphs).
- If user asks about specific stocks, redirect to Particular Stock topic.
- If user asks about trading tactics, redirect to Trading Strategies topic."""

def answer_market_basics(user_message: str, history: list[dict] | None = None) -> str:
    """Market basics educator."""
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
