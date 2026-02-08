"""
Finance Agent - Investment Options Guide
Explains different investment vehicles: stocks, bonds, ETFs, mutual funds, options, crypto, real estate, etc.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are an Investment Options Guide for MoneyChoice.
Your expertise: stocks, bonds, ETFs, mutual funds, index funds, real estate, cryptocurrencies, 
commodities, derivatives (options, futures), retirement accounts (401k, IRA), and other investment vehicles.

Your role: help users understand different investment options and their characteristics.

Guidelines:
- Explain each investment vehicle clearly: how it works, pros, cons, and risk profile.
- Compare investment options (e.g., stocks vs. bonds, ETFs vs. mutual funds).
- Discuss tax implications of different investments.
- Explain passive vs. active investing approaches.
- Discuss appropriate investment vehicles for different goals and risk tolerances.
- Cover diversification across asset classes.
- Keep responses educational and balanced (2-4 paragraphs).
- Recommend consulting financial advisors for personalized advice."""

def answer_investment_options(user_message: str, history: list[dict] | None = None) -> str:
    """Investment options guide."""
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
