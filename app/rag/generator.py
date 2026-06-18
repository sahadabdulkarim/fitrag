"""LLM generation module using Groq (free tier, Llama 3.1 70B)."""

import os
from dataclasses import dataclass

from groq import Groq

from app.core.prompts import SYSTEM_PROMPT, RAG_QUERY_PROMPT


@dataclass
class GenerationConfig:
    """Configuration for Groq LLM generation."""

    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.3
    max_tokens: int = 1024


class GroqGenerator:
    """Generate answers using Groq's free API with Llama 3.1."""

    def __init__(self, config: GenerationConfig | None = None):
        self.config = config or GenerationConfig()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file or environment.\n"
                "Get a free key at: https://console.groq.com"
            )
        self.client = Groq(api_key=api_key)

    def generate(self, question: str, context: str) -> str:
        """Generate an answer given a question and retrieved context."""
        user_prompt = RAG_QUERY_PROMPT.format(
            context=context,
            question=question,
        )

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return response.choices[0].message.content

    def generate_streaming(self, question: str, context: str):
        """Generate an answer with streaming output."""
        user_prompt = RAG_QUERY_PROMPT.format(
            context=context,
            question=question,
        )

        stream = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
