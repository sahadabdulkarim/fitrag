"""Recommendation Agent: Generates final structured answer from retrieved context."""

import os
import json

from groq import Groq
from dotenv import load_dotenv

from app.agents.state import FitRAGState

load_dotenv()

RECOMMENDATION_SYSTEM_PROMPT = """You are FitRAG, an evidence-based fitness and nutrition advisor.

You generate structured, actionable recommendations grounded in the provided research context.

RULES:
1. ONLY use information from the provided context documents
2. Cite sources explicitly (e.g., "According to Source 1...")
3. If there are safety flags, address them FIRST before any recommendations
4. Be specific with numbers (sets, reps, grams, percentages) when the context supports it
5. Structure your response clearly with headers
6. If the context doesn't fully answer the question, explicitly state what's missing
7. Never recommend anything dangerous without appropriate disclaimers

RESPONSE FORMAT:
Use clear markdown with these sections as relevant:
## Answer
[Main answer grounded in sources]

## Safety Considerations (if applicable)
[Any warnings or contraindications]

## Specific Recommendations
[Actionable advice with numbers]

## Sources
[Which source documents you referenced]
"""


def recommendation_agent(state: FitRAGState) -> FitRAGState:
    """Generate final recommendation using retrieved context + safety info.

    This is the final agent in the pipeline. It takes all accumulated
    context (parsed query, safety flags, retrieved documents) and
    generates a comprehensive, grounded response.
    """
    parsed_query = state.get("parsed_query", {})
    safety_flags = state.get("safety_flags", [])
    retrieval_context = state.get("retrieval_context", "")
    injury_context = state.get("injury_context", [])

    # Build the user prompt with all context
    user_prompt_parts = []

    # Add the specific question
    question = parsed_query.get("specific_question", state["raw_input"])
    user_prompt_parts.append(f"USER QUESTION: {question}")

    # Add user context (if available)
    user_context_items = []
    if parsed_query.get("goal"):
        user_context_items.append(f"Goal: {parsed_query['goal']}")
    if parsed_query.get("experience_level") and parsed_query["experience_level"] != "unknown":
        user_context_items.append(f"Experience: {parsed_query['experience_level']}")
    if parsed_query.get("dietary_restrictions"):
        user_context_items.append(f"Diet: {', '.join(parsed_query['dietary_restrictions'])}")
    if parsed_query.get("training_frequency"):
        user_context_items.append(f"Training: {parsed_query['training_frequency']}x/week")
    if parsed_query.get("body_stats"):
        stats = ", ".join(f"{k}: {v}" for k, v in parsed_query["body_stats"].items())
        user_context_items.append(f"Stats: {stats}")

    if user_context_items:
        user_prompt_parts.append(f"\nUSER PROFILE:\n" + "\n".join(f"- {item}" for item in user_context_items))

    # Add safety flags
    if safety_flags:
        flags_text = "\n".join(
            f"⚠️ [{f['severity'].upper()}] {f['concern']}: {f['recommendation']}"
            for f in safety_flags
        )
        user_prompt_parts.append(f"\nSAFETY FLAGS:\n{flags_text}")

    # Add injury-specific context
    if injury_context:
        injury_text = "\n\n".join(
            f"[Injury Source: {ctx['source']}]\n{ctx['content']}"
            for ctx in injury_context
        )
        user_prompt_parts.append(f"\nINJURY-SPECIFIC CONTEXT:\n{injury_text}")

    # Add main retrieval context
    user_prompt_parts.append(f"\nRETRIEVED CONTEXT:\n{retrieval_context}")

    user_prompt_parts.append("\nBased on ALL the above context, provide your recommendation.")

    full_prompt = "\n\n".join(user_prompt_parts)

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        recommendation = response.choices[0].message.content

        # Extract cited sources from retrieved docs
        sources_cited = list(set(
            doc.get("source", "unknown")
            for doc in state.get("retrieved_docs", [])
        ))

        return {
            **state,
            "recommendation": recommendation,
            "sources_cited": sources_cited,
            "workflow_path": state.get("workflow_path", []) + ["recommendation_agent"],
        }

    except Exception as e:
        return {
            **state,
            "recommendation": f"Error generating recommendation: {str(e)}\n\nRetrieved context is available above.",
            "sources_cited": [],
            "workflow_path": state.get("workflow_path", []) + ["recommendation_agent (error)"],
            "error": str(e),
        }
