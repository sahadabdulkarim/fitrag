# RAG Training Agent

A production-grade RAG system using LangGraph multi-agent orchestration 
and hybrid retrieval (Dense + BM25 + RRF) for evidence-based recommendations.

## Architecture
- 4-node LangGraph agent (Intake → Safety → Retrieval → Recommendation)
- Hybrid retrieval: FAISS dense + BM25 sparse + Reciprocal Rank Fusion
- FastAPI backend with full audit trail in SQLite
- Complete explainability: every decision traceable to source documents

## Stack
Python · LangChain · LangGraph · FAISS · FastAPI · SQLite · Groq API

## Setup
\`\`\`bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # Add your API key
python scripts/ingest.py
python app/query.py
\`\`\`
