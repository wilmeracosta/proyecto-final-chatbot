
"""
conomad_api.py  — Backend FastAPI para el chatbot CoNomad
Ejecutar: uvicorn conomad_api:app --reload --port 8000
"""

import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# ── Configuración ──
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-proj-XXXXXXXX")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o"
TOP_K           = 5
CACHE_DIR       = Path("conomad_cache")

# ── Carga del índice ──
faiss_index = faiss.read_index(str(CACHE_DIR / "faiss_index.bin"))
with open(CACHE_DIR / "chunks.pkl", "rb") as f:
    data = pickle.load(f)
all_chunks  = data["chunks"]
all_sources = data["sources"]

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
Eres el asistente virtual oficial de CoNomad, un espacio de coworking y coliving
para nómadas digitales ubicado en Corralejo, Fuerteventura (Islas Canarias).

Tu objetivo es responder con amabilidad y precisión las preguntas de los usuarios
sobre los servicios, precios, alojamientos y experiencias de CoNomad.

INSTRUCCIONES:
- Responde SIEMPRE en el mismo idioma que el usuario (español o inglés).
- Usa ÚNICAMENTE la información proporcionada en el contexto.
- Si la respuesta no está en el contexto, dí amablemente que no tienes esa
  información y sugiere contactar vía WhatsApp o la web.
- Sé conciso, cálido y profesional.
- Cuando menciones precios, sé exacto con los datos del contexto.
- No inventes información.
""".strip()

# ── FastAPI app ──
app = FastAPI(title="CoNomad Chatbot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://conomad.es", "http://localhost"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

# ── Modelos Pydantic ──
class ChatRequest(BaseModel):
    query   : str
    history : List[Dict] = []

class ChatResponse(BaseModel):
    answer  : str
    history : List[Dict]

# ── Helpers ──
def embed_query(query: str) -> np.ndarray:
    res = client.embeddings.create(input=[query.replace("\n", " ")], model=EMBEDDING_MODEL)
    vec = np.array(res.data[0].embedding, dtype="float32")
    return (vec / np.linalg.norm(vec)).reshape(1, -1)

def retrieve(query: str) -> List[Dict]:
    q    = embed_query(query)
    scores, idxs = faiss_index.search(q, TOP_K)
    return [
        {"text": all_chunks[i], "source": all_sources[i], "score": float(s)}
        for s, i in zip(scores[0], idxs[0]) if i >= 0
    ]

def rag_answer(query: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    ctx = retrieve(query)
    context_str = "\n\n".join(
        f"[{r['source']}]\n{r['text']}" for r in ctx
    )
    user_msg = f"CONTEXTO:\n{context_str}\n\nPREGUNTA:\n{query}"
    msgs     = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs    += history
    msgs    += [{"role": "user", "content": user_msg}]
    resp     = client.chat.completions.create(
        model=CHAT_MODEL, messages=msgs, temperature=0.3, max_tokens=800
    )
    answer   = resp.choices[0].message.content
    updated  = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated

# ── Endpoint ──
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, history = rag_answer(req.query, req.history)
    return ChatResponse(answer=answer, history=history)

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(all_chunks)}
