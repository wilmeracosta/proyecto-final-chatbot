# ============================================================
# IMPORTACIONES
# ============================================================

import os
import re
import json
import time
import pickle
import textwrap
from pathlib import Path
from typing import List, Tuple, Dict

import requests
import numpy as np
import faiss
import tiktoken
from bs4 import BeautifulSoup
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Modelos
EMBEDDING_MODEL = "text-embedding-3-small"   # 1536 dimensiones, económico
CHAT_MODEL      = "gpt-4o"                   # Cambia a gpt-4o-mini para ahorrar

# Parámetros de chunking
CHUNK_SIZE      = 500     # tokens por chunk
CHUNK_OVERLAP   = 50      # tokens de solapamiento entre chunks

# Número de chunks más relevantes a recuperar
TOP_K           = 5

# URLs del sitio a scrapear
BASE_URL = "https://conomad.es"
URLS_TO_SCRAPE = [
    "https://conomad.es/",
    "https://conomad.es/coworking/",
    "https://conomad.es/coliving/",
    "https://conomad.es/team-retreats/",
    "https://conomad.es/quienes-somos-2/",
    "https://conomad.es/contacto-2/",
    "https://conomad.es/destiny-club/",
    "https://conomad.es/coliving/bristol-sunset-beach/",
    "https://conomad.es/coliving/shambhala/",
    "https://conomad.es/coliving/holidays-home/",
    "https://conomad.es/fuerteventura/",
    # Tienda en linea
    "https://conomad.es/paquetes/",
    # Productos Open Space
    "https://conomad.es/producto/open-space-1-hour/",
    "https://conomad.es/producto/open-space-1-day/",
    "https://conomad.es/producto/open-space-1-week/",
    "https://conomad.es/producto/open-space-1-month/",
    "https://conomad.es/producto/open-space-10-passes/",
    # Productos Open Space + Gym
    "https://conomad.es/producto/open-space-open-gym-1-day/",
    "https://conomad.es/producto/open-space-open-gym-1-week/",
    "https://conomad.es/producto/open-space-open-gym-1-month/",
    # Productos Private Cabin
    "https://conomad.es/producto/private-cabin-1-day/",
    "https://conomad.es/producto/private-cabin-1-week/",
    "https://conomad.es/producto/private-cabin-1-month/",
]

# Rutas de almacenamiento local
CACHE_DIR        = Path("conomad_cache")
CACHE_DIR.mkdir(exist_ok=True)
FAISS_INDEX_PATH = CACHE_DIR / "faiss_index.bin"
CHUNKS_PATH      = CACHE_DIR / "chunks.pkl"

# Cliente OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Tokenizer para contar tokens (usa el del modelo de embedding)
enc = tiktoken.get_encoding("cl100k_base")

# ============================================================
# SCRAPER
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; CoNomadBot/1.0; "
        "+https://conomad.es)"
    )
}

TAGS_TO_REMOVE  = ["script", "style", "noscript", "header", "footer",
                   "nav", "aside", "form", "iframe", "svg"]
CLASSES_TO_SKIP = ["menu", "cookie", "gdpr", "popup", "modal",
                   "social", "share", "lang", "translatepress", "wpr-advanced-text"]


def fetch_page(url: str) -> str:
    """Descarga una URL y devuelve el HTML crudo."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    except requests.RequestException as e:
        print(f"  ⚠️  Error al descargar {url}: {e}")
        return ""


def clean_html(html: str, url: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for tag in TAGS_TO_REMOVE:
        for el in soup.find_all(tag):
            el.decompose()

    for class_name in CLASSES_TO_SKIP:
        for el in soup.find_all(class_=class_name):
            el.decompose()

    elementos = soup.find_all(["p", "h1", "h2", "h3", "h4", "li"])

    lines = []
    seen = set()
    for el in elementos:
        texto = el.get_text(separator=" ", strip=True)
        if texto and len(texto) > 3 and texto not in seen:
            seen.add(texto)
            lines.append(texto)

    if not lines:
        return ""

    text = "\n".join(lines)
    return f"[Fuente: {url}]\n{text}"


def scrape_site(urls: List[str]) -> List[Dict]:
    """Recorre todas las URLs, descarga y limpia el contenido."""
    documents = []
    for url in tqdm(urls, desc="Scraping"):
        html = fetch_page(url)
        if not html:
            continue
        text = clean_html(html, url)
        if text and len(text) > 100:
            documents.append({"url": url, "text": text})
        time.sleep(0.5)
    return documents


documents = scrape_site(URLS_TO_SCRAPE)

# ============================================================
# CHUNKING CON SOLAPAMIENTO
# ============================================================

def split_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int    = CHUNK_OVERLAP,
) -> List[str]:
    tokens = enc.encode(text)
    chunks = []
    start  = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text   = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_size - overlap

    return chunks


def build_chunk_store(documents: List[Dict]) -> Tuple[List[str], List[str]]:
    all_chunks  = []
    all_sources = []

    for doc in documents:
        chunks = split_into_chunks(doc["text"])
        all_chunks.extend(chunks)
        all_sources.extend([doc["url"]] * len(chunks))

    return all_chunks, all_sources


all_chunks, all_sources = build_chunk_store(documents)

with open(CHUNKS_PATH, "wb") as f:
    pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

# ============================================================
# EMBEDDINGS CON OPENAI
# ============================================================

EMBED_BATCH_SIZE = 100


def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> np.ndarray:
    all_embeddings = []

    for i in tqdm(range(0, len(texts), EMBED_BATCH_SIZE), desc="Generando embeddings"):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        batch_clean = [t.replace("\n", " ") for t in batch]

        response = client.embeddings.create(input=batch_clean, model=model)
        batch_vecs = [item.embedding for item in response.data]
        all_embeddings.extend(batch_vecs)
        time.sleep(0.2)

    return np.array(all_embeddings, dtype="float32")


embeddings = get_embeddings(all_chunks)

# ============================================================
# CONSTRUCCIÓN Y PERSISTENCIA DEL ÍNDICE FAISS
# ============================================================

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)

    norms      = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)
    index.add(normalized)

    return index, normalized


faiss_index, normalized_embeddings = build_faiss_index(embeddings)
faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))

# ============================================================
# FUNCIÓN DE CARGA DEL ÍNDICE
# ============================================================

def load_index_from_disk() -> Tuple:
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "No se encontró el índice. Ejecuta el pipeline completo primero."
        )
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "rb") as f:
        data = pickle.load(f)
    return index, data["chunks"], data["sources"]


# Descomenta para recargar en lugar de regenerar:
# faiss_index, all_chunks, all_sources = load_index_from_disk()

# ============================================================
# FUNCIÓN DE RETRIEVAL
# ============================================================

def embed_query(query: str, model: str = EMBEDDING_MODEL) -> np.ndarray:
    response = client.embeddings.create(
        input=[query.replace("\n", " ")],
        model=model,
    )
    vec  = np.array(response.data[0].embedding, dtype="float32")
    norm = np.linalg.norm(vec)
    return (vec / (norm + 1e-10)).reshape(1, -1)


def retrieve(
    query: str,
    index: faiss.IndexFlatIP,
    chunks: List[str],
    sources: List[str],
    top_k: int = TOP_K,
) -> List[Dict]:
    query_vec       = embed_query(query)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text"  : chunks[idx],
                "source": sources[idx],
                "score" : float(score),
            })
    return results

# ============================================================
# PIPELINE RAG COMPLETO
# ============================================================

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


def build_rag_prompt(query: str, context_results: List[Dict]) -> str:
    context_blocks = []
    for i, r in enumerate(context_results, 1):
        context_blocks.append(f"[Fragmento {i} — {r['source']}]\n{r['text']}")
    context_str = "\n\n".join(context_blocks)

    return (
        f"CONTEXTO RELEVANTE DE LA WEB DE CONOMAD:\n"
        f"{'='*60}\n"
        f"{context_str}\n"
        f"{'='*60}\n\n"
        f"PREGUNTA DEL USUARIO:\n{query}"
    )


def rag_answer(
    query: str,
    history: List[Dict] = None,
    top_k: int = TOP_K,
) -> Tuple[str, List[Dict]]:
    if history is None:
        history = []

    context_results = retrieve(query, faiss_index, all_chunks, all_sources, top_k)
    user_message    = build_rag_prompt(query, context_results)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=800,
    )

    answer = response.choices[0].message.content

    updated_history = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]

    return answer, updated_history

# ============================================================
# WIDGET DE CHAT CON ipywidgets
# ============================================================

chat_history = []

CSS = """
<style>
#conomad-chat-toggle {
  position: fixed; bottom: 20px; right: 20px;
  background: #000; color: #fff; border: none;
  border-radius: 999px; width: 60px; height: 60px;
  font-size: 22px; cursor: pointer; z-index: 9999;
}
#conomad-chat-window {
  position: fixed; bottom: 90px; right: 20px;
  width: 320px; height: 420px; background: #fff;
  border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  display: none; flex-direction: column; overflow: hidden;
  z-index: 9999; font-family: Arial, sans-serif;
}
.chat-header { background: #000; color: #fff; padding: 12px; font-weight: bold; }
#chat-messages { flex: 1; padding: 12px; overflow-y: auto; background: #f5f5f5; }
.bubble { margin-bottom: 10px; padding: 10px 12px; border-radius: 12px; max-width: 80%; font-size: 14px; }
.bubble.user { background: #000; color: #fff; margin-left: auto; }
.bubble.bot { background: #e5e5ea; color: #000; }
.chat-input-area { display: flex; border-top: 1px solid #ddd; }
#chat-input { flex: 1; border: none; padding: 10px; outline: none; }
#chat-send { border: none; background: #000; color: #fff; padding: 0 16px; cursor: pointer; }
</style>
"""

output_area  = widgets.Output()
text_input   = widgets.Text(
    placeholder="Escribe tu pregunta sobre CoNomad...",
    layout=widgets.Layout(width="580px"),
)
send_button  = widgets.Button(
    description="Enviar", button_style="primary",
    icon="paper-plane", layout=widgets.Layout(width="100px"),
)
clear_button = widgets.Button(
    description="Limpiar", button_style="warning",
    icon="trash", layout=widgets.Layout(width="90px"),
)
input_row = widgets.HBox([text_input, send_button, clear_button])


def render_messages(messages: List[Dict]):
    html = CSS + '<div style="max-width:700px; background:#f5f5f5; border-radius:12px; padding:16px;">'
    for msg in messages:
        if msg["role"] == "user":
            html += f'''
            <div style="text-align:right; margin: 8px 0;">
                <span style="font-size:12px; color:#888; display:block; margin-bottom:2px;">Tú</span>
                <span style="background:#000; color:#fff; padding:10px 14px;
                             border-radius:18px 18px 4px 18px; display:inline-block;
                             max-width:80%; font-size:14px;">{msg["content"]}</span>
            </div>'''
        else:
            content = msg["content"].replace("\n", "<br>")
            html += f'''
            <div style="text-align:left; margin: 8px 0;">
                <span style="font-size:12px; color:#888; display:block; margin-bottom:2px;">🌊 CoNomad Bot</span>
                <span style="background:#fff; color:#000; padding:10px 14px;
                             border-radius:18px 18px 18px 4px; display:inline-block;
                             max-width:80%; font-size:14px; border:1px solid #ddd;">{content}</span>
            </div>'''
    html += '</div>'
    return html


def on_send(b):
    global chat_history
    query = text_input.value.strip()
    if not query:
        return

    text_input.value = ""
    send_button.disabled = True
    send_button.description = "..."

    with output_area:
        clear_output(wait=True)
        temp_history = chat_history + [{"role": "user", "content": query}]
        display(HTML(render_messages(temp_history)))

    try:
        answer, chat_history = rag_answer(query, history=chat_history)
    except Exception as e:
        answer = f"⚠️ Error al generar respuesta: {e}"
        chat_history = chat_history + [
            {"role": "user",      "content": query},
            {"role": "assistant", "content": answer},
        ]

    with output_area:
        clear_output(wait=True)
        display(HTML(render_messages(chat_history)))

    send_button.disabled    = False
    send_button.description = "Enviar"


def on_clear(b):
    global chat_history
    chat_history = []
    with output_area:
        clear_output(wait=True)
        display(HTML(
            CSS +
            '<div style="max-width:700px; background:#f5f5f5; border-radius:12px; padding:16px;">'
            '<div style="text-align:left; margin: 8px 0;">'
            '<span style="font-size:12px; color:#888; display:block; margin-bottom:2px;">🌊 CoNomad Bot</span>'
            '<span style="background:#fff; color:#000; padding:10px 14px;'
            'border-radius:18px 18px 18px 4px; display:inline-block;'
            'max-width:80%; font-size:14px; border:1px solid #ddd;">'
            '¡Hola! Soy el asistente virtual de CoNomad. ¿En qué puedo ayudarte hoy?'
            '</span></div></div>'
        ))


send_button.on_click(on_send)
clear_button.on_click(on_clear)
text_input.on_submit(lambda x: on_send(None))

on_clear(None)
display(output_area, input_row)

# ============================================================
# GENERACIÓN DEL WIDGET HTML PARA EMBEBER EN LA WEB
# ============================================================

CHAT_WIDGET_HTML = '''
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>CoNomad Chat Widget</title>
  <style>
#conomad-chat-toggle {
  position: fixed; bottom: 20px; right: 20px;
  background: #000; color: #fff; border: none;
  border-radius: 999px; width: 60px; height: 60px;
  font-size: 22px; cursor: pointer; z-index: 9999;
}
#conomad-chat-window {
  position: fixed; bottom: 90px; right: 20px;
  width: 320px; height: 420px; background: #fff;
  border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  display: none; flex-direction: column; overflow: hidden;
  z-index: 9999; font-family: Arial, sans-serif;
}
.chat-header { background: #000; color: #fff; padding: 12px; font-weight: bold; }
#chat-messages { flex: 1; padding: 12px; overflow-y: auto; background: #f5f5f5; }
.bubble { margin-bottom: 10px; padding: 10px 12px; border-radius: 12px; max-width: 80%; font-size: 14px; }
.bubble.user { background: #000; color: #fff; margin-left: auto; }
.bubble.bot { background: #e5e5ea; color: #000; }
.chat-input-area { display: flex; border-top: 1px solid #ddd; }
#chat-input { flex: 1; border: none; padding: 10px; outline: none; }
#chat-send { border: none; background: #000; color: #fff; padding: 0 16px; cursor: pointer; }
  </style>
</head>
<body>

<button id="conomad-chat-toggle" title="Abrir chat">💬</button>

<div id="conomad-chat-window">
  <div class="chat-header"><span>🌊</span> CoNomad Assistant</div>
  <div id="chat-messages">
    <div class="bubble bot">¡Hola! Soy el asistente virtual de CoNomad. ¿En qué puedo ayudarte?</div>
  </div>
  <div class="chat-input-area">
    <input id="chat-input" type="text" placeholder="Escribe tu pregunta..." />
    <button id="chat-send">➤</button>
  </div>
</div>

<script>
  const API_ENDPOINT = "http://localhost:8000/chat"; // CAMBIAR por tu endpoint real

  const toggle  = document.getElementById("conomad-chat-toggle");
  const chatWin = document.getElementById("conomad-chat-window");
  const msgArea = document.getElementById("chat-messages");
  const input   = document.getElementById("chat-input");
  const sendBtn = document.getElementById("chat-send");

  let history = [];

  toggle.addEventListener("click", () => {
    chatWin.classList.toggle("open");
    toggle.textContent = chatWin.classList.contains("open") ? "✕" : "💬";
    if (chatWin.classList.contains("open")) input.focus();
  });

  function addBubble(text, role) {
    const div = document.createElement("div");
    div.classList.add("bubble", role);
    div.innerHTML = text.replace(/\n/g, "<br>");
    msgArea.appendChild(div);
    msgArea.scrollTop = msgArea.scrollHeight;
    return div;
  }

  async function sendMessage() {
    const query = input.value.trim();
    if (!query) return;
    input.value = "";
    sendBtn.disabled = true;

    addBubble(query, "user");
    const typingDiv = addBubble("Escribiendo…", "bot typing");

    try {
      const res = await fetch(API_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, history }),
      });
      const data = await res.json();
      typingDiv.remove();
      addBubble(data.answer, "bot");
      history = data.history;
    } catch (err) {
      typingDiv.remove();
      addBubble("⚠️ Error de conexión. Inténtalo de nuevo.", "bot");
    }
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") sendMessage(); });
</script>
</body>
</html>
'''

import os
os.makedirs("conomad_cache", exist_ok=True)

with open("conomad_chat_widget.html", "w", encoding="utf-8") as f:
    f.write(CHAT_WIDGET_HTML)

# ============================================================
# SERVIDOR FastAPI
# Para ejecutar: uvicorn conomad_api:app --reload --port 8000
# ============================================================

API_CODE = '''
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
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL      = "gpt-4o"
TOP_K           = 5
CACHE_DIR       = Path("conomad_cache")

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

app = FastAPI(title="CoNomad Chatbot API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://conomad.es", "http://localhost"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query   : str
    history : List[Dict] = []

class ChatResponse(BaseModel):
    answer  : str
    history : List[Dict]

def embed_query(query: str) -> np.ndarray:
    res = client.embeddings.create(input=[query.replace("\\n", " ")], model=EMBEDDING_MODEL)
    vec = np.array(res.data[0].embedding, dtype="float32")
    return (vec / np.linalg.norm(vec)).reshape(1, -1)

def retrieve(query: str) -> List[Dict]:
    q = embed_query(query)
    scores, idxs = faiss_index.search(q, TOP_K)
    return [
        {"text": all_chunks[i], "source": all_sources[i], "score": float(s)}
        for s, i in zip(scores[0], idxs[0]) if i >= 0
    ]

def rag_answer(query: str, history: List[Dict]) -> Tuple[str, List[Dict]]:
    ctx = retrieve(query)
    context_str = "\\n\\n".join(f"[{r[\'source\']}]\\n{r[\'text\']}" for r in ctx)
    user_msg = f"CONTEXTO:\\n{context_str}\\n\\nPREGUNTA:\\n{query}"
    msgs     = [{"role": "system", "content": SYSTEM_PROMPT}]
    msgs    += history
    msgs    += [{"role": "user", "content": user_msg}]
    resp     = client.chat.completions.create(
        model=CHAT_MODEL, messages=msgs, temperature=0.3, max_tokens=800
    )
    answer  = resp.choices[0].message.content
    updated = history + [
        {"role": "user",      "content": query},
        {"role": "assistant", "content": answer},
    ]
    return answer, updated

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer, history = rag_answer(req.query, req.history)
    return ChatResponse(answer=answer, history=history)

@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(all_chunks)}
'''

with open("conomad_api.py", "w", encoding="utf-8") as f:
    f.write(API_CODE)

# ============================================================
# REINDEXACIÓN SELECTIVA
# ============================================================

def update_index_with_new_urls(new_urls: List[str]):
    """Añade nuevas páginas al índice FAISS sin regenerar todo."""
    global all_chunks, all_sources, faiss_index

    new_docs = scrape_site(new_urls)
    if not new_docs:
        print("⚠️ No se encontró contenido en las nuevas URLs.")
        return

    new_chunks, new_sources = build_chunk_store(new_docs)
    new_embeddings = get_embeddings(new_chunks)

    norms      = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
    normalized = new_embeddings / (norms + 1e-10)
    faiss_index.add(normalized)

    all_chunks  = all_chunks  + new_chunks
    all_sources = all_sources + new_sources

    faiss.write_index(faiss_index, str(FAISS_INDEX_PATH))
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump({"chunks": all_chunks, "sources": all_sources}, f)

# Ejemplo de uso:
# update_index_with_new_urls(["https://conomad.es/blog/nuevo-articulo/"])
