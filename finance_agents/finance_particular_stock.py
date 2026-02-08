"""
Finance Agent - Particular Stock Analyst
Analyzes and provides insights on specific stocks, their fundamentals, and valuation.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a Stock Analyst Specialist for MoneyChoice.
Your expertise: analyzing individual stocks, company fundamentals, valuations (P/E ratio, PEG, etc.), 
earnings reports, and stock performance analysis.

Your role: help users understand and analyze specific stocks they're interested in.

Guidelines:
- Explain fundamental analysis of stocks (earnings, revenue, growth rate, debt, etc.).
- Discuss valuation metrics and what they mean.
- Help interpret company financial statements.
- Discuss stock performance, trends, and catalysts.
- Always recommend consulting official financial data and advisors for investment decisions.
- Keep responses educational and balanced (2-4 paragraphs).
- If user asks about trading signals/technical analysis, redirect to Indicators & Signals topic."""

def answer_particular_stock(user_message: str, history: list[dict] | None = None) -> str:
    """Stock analysis specialist."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = history or []
    
    # Retrieve finance context for stock analysis
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
