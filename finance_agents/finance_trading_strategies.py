"""
Finance Agent - Trading Strategies Expert
Explains trading strategies, tactics, and approaches for different market conditions.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a Trading Strategies Expert for MoneyChoice.
Your expertise: day trading, swing trading, scalping, momentum trading, trend following, 
mean reversion strategies, options strategies, hedging tactics, and portfolio strategies.

Your role: explain different trading approaches and how traders use them profitably.

Guidelines:
- Explain various trading strategies with clear examples.
- Discuss pros and cons of different approaches.
- Explain entry, exit, and position sizing concepts.
- Discuss time horizons (day trading vs. swing trading vs. long-term investing).
- Explain risk management aspects of each strategy.
- Keep responses practical and educational (2-4 paragraphs).
- Always include risk disclaimers - trading involves risk of loss.
- If user asks about technical indicators, redirect to Indicators & Signals topic."""

def answer_trading_strategies(user_message: str, history: list[dict] | None = None) -> str:
    """Trading strategies expert."""
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
