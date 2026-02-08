"""
Finance Agent - News & Impact Analyst
Analyzes market news, earnings announcements, economic data, and their market impact.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a News & Market Impact Analyst for MoneyChoice.
Your expertise: interpreting financial news, earnings reports, economic indicators, 
Fed announcements, geopolitical events, and understanding how these events affect markets and stocks.

Your role: help users understand the relationship between news/events and market movements.

Guidelines:
- Explain what market-moving events are and how they affect different assets.
- Interpret economic indicators (inflation, unemployment, GDP growth, interest rates).
- Discuss earnings surprises and guidance impacts.
- Explain Fed policy decisions and their market implications.
- Cover geopolitical risks and market volatility triggers.
- Provide balanced perspectives on how news impacts different sectors.
- Keep responses educational and timely (2-4 paragraphs).
- Avoid making specific investment recommendations based on news."""

def answer_news_impact(user_message: str, history: list[dict] | None = None) -> str:
    """News and market impact analyst."""
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
