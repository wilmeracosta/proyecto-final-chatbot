# 🤖 CoNomad Chatbot — RAG con OpenAI y FAISS

Chatbot conversacional para el sitio web de [CoNomad](https://conomad.es), un espacio de coworking y coliving en Corralejo, Fuerteventura. El bot responde preguntas sobre servicios, precios y alojamientos usando información extraída directamente de la web.

## 🧠 Arquitectura

```
conomad.es → BeautifulSoup → Chunks → text-embedding-3-small → FAISS → GPT-4o → Respuesta
```

El sistema usa **RAG (Retrieval-Augmented Generation)**: en lugar de entrenar un modelo, extrae el contenido de la web, lo convierte en vectores y los almacena en un índice FAISS. Cuando el usuario hace una pregunta, el sistema busca los fragmentos más relevantes y se los pasa a GPT-4o como contexto para generar la respuesta.

## 📁 Estructura del proyecto

```
├── main.py                  # Pipeline completo: scraping, embeddings, índice y widget
├── main_ipynb.ipynb         # Versión notebook para Google Colab
├── conomad_api.py           # Backend FastAPI (generado al ejecutar main.py)
├── conomad_chat_widget.html # Widget HTML para incrustar en WordPress
├── requirements.txt         # Dependencias del proyecto
├── .env.example             # Plantilla de variables de entorno
└── .gitignore               # Archivos excluidos de Git
```

## ⚙️ Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/Final-chatbot-conomad.git
cd Final-chatbot-conomad
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Crea el archivo `.env` con tu API key de OpenAI:
```bash
cp .env.example .env
```
Edita el `.env` y añade tu clave:
```
OPENAI_API_KEY=sk-proj-...
```

## 🚀 Uso

### 1. Generar el índice

Ejecuta `main.py` para hacer el scraping, generar los embeddings y construir el índice FAISS:

```bash
python main.py
```

Esto generará la carpeta `conomad_cache/` con los archivos del índice, y también el archivo `conomad_api.py`.

### 2. Arrancar el servidor

```bash
uvicorn conomad_api:app --host 0.0.0.0 --port 8000
```

El servidor quedará disponible en `http://localhost:8000`.

### 3. Incrustar el widget en WordPress

Abre `conomad_chat_widget.html`, actualiza la variable `API_ENDPOINT` con la URL de tu servidor desplegado y pega el contenido en un bloque HTML personalizado de WordPress.

## 🔄 Actualizar el índice

Si se publican nuevas páginas en la web, puedes añadirlas al índice sin regenerarlo completo usando la función `update_index_with_new_urls`:

```python
update_index_with_new_urls(["https://conomad.es/blog/nuevo-articulo/"])
```

## 🛠️ Tecnologías

- [OpenAI API](https://platform.openai.com/) — embeddings y generación de respuestas
- [FAISS](https://github.com/facebookresearch/faiss) — búsqueda vectorial
- [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) — scraping web
- [FastAPI](https://fastapi.tiangolo.com/) — servidor backend
- [tiktoken](https://github.com/openai/tiktoken) — tokenización

## 📄 Licencia

MIT
