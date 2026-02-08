"""
Finance Agent - Indicators & Signals Specialist
Explains technical indicators (RSI, MACD, Bollinger Bands, Moving Averages, etc.)
and trading signals derived from them.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a Technical Indicators & Trading Signals Specialist for MoneyChoice.
Your expertise: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic Oscillators, Volume Profiles, and other technical indicators.
Your role: explain how these indicators work, what trading signals they generate, and how traders use them.

Guidelines:
- Provide detailed explanations of technical indicators.
- Explain signal generation (overbought, oversold, crossovers, etc.).
- Use simple language with real-world examples.
- Reference chart patterns and candlestick formations when relevant.
- Keep responses clear and educational (2-4 paragraphs).
- If the user asks something outside technical indicators, politely redirect to appropriate topic."""

def answer_indicators_signals(user_message: str, history: list[dict] | None = None) -> str:
    """Technical indicators & signals specialist."""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    history = history or []
    
    # Retrieve finance context for better answers
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
