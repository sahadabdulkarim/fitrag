"""Intake Agent: Parses raw user input into structured fitness query."""

import json
import os

from groq import Groq
from dotenv import load_dotenv

from app.agents.state import FitRAGState, ParsedQuery

load_dotenv()

INTAKE_SYSTEM_PROMPT = """You are a fitness query parser. Your job is to extract structured information 
from a user's natural language fitness/nutrition question.

You MUST respond with ONLY valid JSON (no markdown, no explanation, no code fences).

Extract these fields:
- goal: one of "muscle_gain", "fat_loss", "injury_rehab", "strength", "endurance", "flexibility", "general_info"
- specific_question: the core question being asked (rephrased clearly)
- dietary_restrictions: list of dietary restrictions mentioned (empty list if none)
- injuries: list of body parts with injuries/pain (empty list if none)
- training_frequency: number of training days per week mentioned (null if not mentioned)
- experience_level: one of "beginner", "intermediate", "advanced", "unknown"
- body_stats: dict with any mentioned stats like weight_kg, height_cm, age, body_fat_pct (empty dict if none)
- urgency: "low" for general questions, "medium" for specific program requests, "high" for injury/pain questions

Example input: "I'm 28, vegan, want to lose 10kg. I train 4x per week but have a bad knee"
Example output: {"goal": "fat_loss", "specific_question": "How to lose 10kg as a vegan with a knee injury training 4 days per week", "dietary_restrictions": ["vegan"], "injuries": ["knee"], "training_frequency": 4, "experience_level": "unknown", "body_stats": {"age": 28}, "urgency": "medium"}
"""


def intake_agent(state: FitRAGState) -> FitRAGState:
    """Parse raw user input into structured query using LLM.

    This agent takes the raw user message and extracts structured
    information: goals, constraints, injuries, experience level, etc.
    """
    raw_input = state["raw_input"]

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": INTAKE_SYSTEM_PROMPT},
                {"role": "user", "content": raw_input},
            ],
            temperature=0.1,
            max_tokens=512,
        )

        result_text = response.choices[0].message.content.strip()

        # Parse JSON response
        # Handle potential markdown code fences
        if result_text.startswith("```"):
            result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0]

        parsed = json.loads(result_text)

        # Build ParsedQuery with defaults
        parsed_query: ParsedQuery = {
            "goal": parsed.get("goal", "general_info"),
            "specific_question": parsed.get("specific_question", raw_input),
            "dietary_restrictions": parsed.get("dietary_restrictions", []),
            "injuries": parsed.get("injuries", []),
            "training_frequency": parsed.get("training_frequency"),
            "experience_level": parsed.get("experience_level", "unknown"),
            "body_stats": parsed.get("body_stats", {}),
            "urgency": parsed.get("urgency", "low"),
        }

        return {
            **state,
            "parsed_query": parsed_query,
            "has_injury": len(parsed_query["injuries"]) > 0,
            "workflow_path": state.get("workflow_path", []) + ["intake_agent"],
        }

    except (json.JSONDecodeError, KeyError) as e:
        # Fallback: create a basic parsed query without LLM
        return {
            **state,
            "parsed_query": {
                "goal": "general_info",
                "specific_question": raw_input,
                "dietary_restrictions": [],
                "injuries": [],
                "training_frequency": None,
                "experience_level": "unknown",
                "body_stats": {},
                "urgency": "low",
            },
            "has_injury": False,
            "workflow_path": state.get("workflow_path", []) + ["intake_agent (fallback)"],
            "error": f"Intake parsing error: {str(e)}",
        }
    except Exception as e:
        return {
            **state,
            "parsed_query": {
                "goal": "general_info",
                "specific_question": raw_input,
                "dietary_restrictions": [],
                "injuries": [],
                "training_frequency": None,
                "experience_level": "unknown",
                "body_stats": {},
                "urgency": "low",
            },
            "has_injury": False,
            "workflow_path": state.get("workflow_path", []) + ["intake_agent (error)"],
            "error": f"Intake agent error: {str(e)}",
        }
