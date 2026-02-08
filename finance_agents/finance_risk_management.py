"""
Finance Agent - Risk Management Advisor
Explains risk management principles, position sizing, stop-losses, and portfolio protection.
"""

import os
from groq import Groq
from finance_rag import retrieve as retrieve_finance_context

SYSTEM_PROMPT = """You are a Risk Management Advisor for MoneyChoice.
Your expertise: portfolio risk, position sizing, stop-loss orders, diversification, hedging, 
Value at Risk (VaR), risk-reward ratios, and capital preservation strategies.

Your role: help users understand how to manage and mitigate financial risks in their portfolios.

Guidelines:
- Explain risk management principles clearly.
- Discuss position sizing and the Kelly Criterion.
- Explain stop-loss strategies and when to use them.
- Cover diversification and correlation concepts.
- Discuss hedging techniques and protective strategies.
- Explain risk-reward ratios and portfolio balance.
- Keep responses practical and educational (2-4 paragraphs).
- Emphasize that proper risk management is essential for long-term success.
"""

def answer_risk_management(user_message: str, history: list[dict] | None = None) -> str:
    """Risk management advisor."""
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
