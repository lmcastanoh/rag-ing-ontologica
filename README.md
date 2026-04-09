# RAG Ontologica — Sistema RAG Agentico para Fichas Tecnicas Vehiculares

Sistema RAG (Retrieval-Augmented Generation) agentico especializado en fichas tecnicas
de vehiculos. Usa una arquitectura **ReAct + Reflecting** con LangGraph, donde un agente
razona autonomamente sobre que herramientas usar (busqueda vectorial MMR, HyDE,
descomposicion de preguntas, comparacion, etc.), evalua su propia respuesta, y reintenta
con feedback hasta 3 veces antes de escalar a busqueda web.

## Stack Tecnologico

| Componente | Tecnologia |
|------------|-----------|
| Backend API | FastAPI + Uvicorn |
| Orquestacion | LangGraph (grafo de estados con loops) |
| LLM | OpenAI `gpt-5-nano` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Base vectorial | ChromaDB (chunking fijo o semantico) |
| Busqueda | MMR (Maximal Marginal Relevance) |
| Frontend | Streamlit |
| OCR | EasyOCR (paginas escaneadas) |
| Extraccion PDF | pdfplumber |
| Web Search | DuckDuckGo (fallback) |
| Tracing | LangSmith |
| Evaluacion | Metricas custom + LLM-as-Judge |

---

## Estructura del Proyecto

```
rag-ing-ontologica/
│
├── backend/
│   ├── app.py                # API FastAPI: endpoints /ingest, /chat/stream
│   ├── rag_graph.py          # Grafo LangGraph: ReAct + Reflecting
│   ├── rag_store.py          # ChromaDB: chunking fijo y semantico
│   ├── tools.py              # 9 tools del agente ReAct
│   ├── prompts.py            # System prompts (clasificador, ReAct, critico, judges)
│   ├── schemas.py            # Modelos Pydantic
│   ├── evaluation.py         # Metricas: Recall@k, Precision@k, MRR, nDCG, LLM-as-Judge
│   ├── eval_dataset.py       # Dataset de ground truth para evaluacion
│   ├── run_evaluation.py     # Script CLI para evaluacion batch
│   ├── kg_rag_agent.py       # Agente RAG sobre Knowledge Graph (ontologia)
│   ├── kg_retriever.py       # Retriever para SPARQL queries
│   ├── ontologia/            # Ontologia OWL/RDF de vehiculos
│   ├── data/                 # PDFs organizados por marca
│   ├── chroma_db/            # Vector store con chunking fijo
│   └── chroma_db_semantic/   # Vector store con chunking semantico
│
├── frontend/
│   └── streamlit_app.py      # Interfaz de chat con trazabilidad visual
│
├── .env                      # Variables de entorno
├── requeriments.txt          # Dependencias Python
└── README.md
```

---

## Datos Disponibles

- **7 marcas**: Toyota, Mazda, Volkswagen, Peugeot, Opel, MG Emotor, Seat
- **50 modelos** indexados
- **584 chunks** (fijo) / **476 chunks** (semantico) en ChromaDB
- Metadata por chunk: `source`, `page`, `marca`, `modelo`, `doc_id`, `chunk_id`, `ocr`, `chunking`

---

## Arquitectura: ReAct + Reflecting Agent

```
START
  │
  ▼
┌─────────────────────┐
│  classify_intent    │  Clasifica en Busqueda|Resumen|Comparacion|GENERAL
└──────────┬──────────┘
           │
    ┌──────┴──────────────────────┐
    │                             │
 GENERAL                    needs_retrieval
    │                             │
    ▼                             ▼
┌──────────────┐       ┌───────────────────┐
│ answer_general│       │   react_agent      │ ◄── Loop ReAct (max 7 iteraciones)
└──────┬───────┘       │                    │     Thought → Action → Observation
       │               │  9 tools disponibles│
       ▼               └─────────┬──────────┘
      END                        │
                                 ▼
                       ┌──────────────────┐
                       │ generate_grounded │ ◄────┐
                       └────────┬─────────┘      │
                                │                │
                                ▼                │ retry con feedback
                       ┌──────────────────┐      │ (max 3 reintentos)
                       │evaluate_grounding │      │
                       └────────┬─────────┘      │
                                │                │
                                ▼                │
                       ┌──────────────────┐      │
                       │evaluate_metrics   │      │
                       └────────┬─────────┘      │
                                │                │
                    ┌───────────┼────────────────┘
                    │           │
              aprobado     rechazado + 3 reintentos
                    │           │
                    ▼           ▼
                   END   ┌──────────────┐
                         │ web_fallback  │  Busqueda en internet
                         └──────┬───────┘
                                ▼
                               END
```

### Patrones implementados

| Patron | Implementacion |
|--------|----------------|
| **ReAct (Reasoning + Acting)** | El agente razona (Thought), elige tool (Action), observa resultado, decide si continuar. Loop de hasta 7 iteraciones. |
| **Reflecting** | El critico evalua la respuesta y da feedback. Si score < 0.5, reintenta con correccion (max 3 veces). |
| **Web Fallback** | Tras 3 fallos de reflexion, escala a busqueda web (DuckDuckGo). |
| **Memory conversacional** | `last_model`/`last_make` persisten entre turnos via `MemorySaver` + reducer `_keep_latest`. |

### Tools del agente ReAct

| Tool | Funcion | Busqueda |
|------|---------|----------|
| `buscar_vectorial` | Busqueda semantica en ChromaDB con filtros de metadata | MMR (lambda=0.7) |
| `buscar_hyde` | HyDE: genera doc hipotetico y busca similares | MMR (lambda=0.7) |
| `buscar_especificacion` | Dato tecnico puntual de un modelo | MMR (lambda=0.7) |
| `buscar_por_marca` | Todos los modelos de una marca | MMR (lambda=0.5) |
| `comparar_modelos` | Tabla comparativa entre 2 modelos | MMR (lambda=0.5) |
| `resumir_ficha` | Resumen estructurado de un modelo | MMR (lambda=0.5) |
| `descomponer_pregunta` | Divide preguntas complejas en sub-preguntas | — |
| `listar_modelos_disponibles` | Catalogo completo indexado | — |
| `buscar_web` | DuckDuckGo (solo en web_fallback) | — |

> **MMR (Maximal Marginal Relevance)**: balance entre relevancia y diversidad.
> `lambda=0.7` para busquedas puntuales, `lambda=0.5` para resumenes/comparaciones donde
> se necesita cubrir distintas secciones de la ficha tecnica sin chunks redundantes.

---

## Modulo de Evaluacion

El sistema incluye un modulo completo de evaluacion con:

### Metricas de Retrieval (con ground truth)
- **Recall@k**: fraccion de docs relevantes recuperados
- **Precision@k**: fraccion de docs recuperados que son relevantes
- **MRR (Mean Reciprocal Rank)**: posicion del primer doc relevante
- **nDCG@k**: relevancia ponderada por posicion

### LLM-as-Judge (sin ground truth)
- **Relevance**: la respuesta es relevante para la pregunta?
- **Faithfulness**: la respuesta es fiel al contexto (sin alucinacion)?

### Visualizacion en trazabilidad
Las metricas de retrieval se calculan en cada query y se muestran en el expander de
trazabilidad del frontend. Las metricas LLM-as-Judge solo se ejecutan en modo evaluacion
batch (`eval_mode=True`) para no agregar latencia al chat.

### Script de evaluacion batch

```powershell
cd backend
python run_evaluation.py
```

Ejecuta las 17 preguntas del dataset de ground truth, calcula todas las metricas, imprime
tabla resumen y guarda resultados en `eval_results.json`.

---

## Tracing con LangSmith

El sistema esta integrado con [LangSmith](https://smith.langchain.com/) para tracing
completo del grafo. Cada nodo tiene `RunnableLambda(...).with_config({run_name, tags, metadata})`,
asi que LangSmith captura automaticamente el arbol de ejecucion completo.

### Configurar LangSmith

1. Crear cuenta en https://smith.langchain.com/
2. Generar API key en **Settings → API Keys → Create API Key** (Personal Access Token)
3. Agregar al `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_TU_API_KEY_AQUI
LANGCHAIN_PROJECT=rag-ontologica
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

4. Reiniciar el backend. Al arrancar deberias ver:
   ```
   [LangSmith] Tracing ACTIVO — proyecto: rag-ontologica
   ```

5. Hacer una pregunta y ver el run completo en https://smith.langchain.com/

### Lo que vera en LangSmith

Cada query genera un trace con el arbol completo:
- `Intent Classifier` → `ReAct Agent` (con todas las iteraciones y tool calls)
- `Grounded Generator` → `Grounding Critic (Reflecting)` → `Metrics Evaluator`
- Si aplica: `Web Fallback`
- Por cada nodo: tiempo de ejecucion, tokens consumidos, prompts enviados, outputs

---

## Configuracion en Windows

### Requisitos previos

- **Python 3.12** — https://www.python.org/downloads/
- **Git** — https://git-scm.com/
- Cuenta de OpenAI con API key activa
- (Opcional) Cuenta de LangSmith para tracing

### 1. Clonar el repositorio

```powershell
git clone https://github.com/lmcastanoh/rag-ing-ontologica.git
cd rag-ing-ontologica
```

### 2. Crear entorno virtual

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
```

### 3. Instalar dependencias

```powershell
pip install -r requeriments.txt
```

### 4. Configurar variables de entorno

Crear archivo `.env` en la raiz del proyecto:

```env
# OpenAI
OPENAI_API_KEY=sk-tu-clave-aqui
HF_TOKEN=hf-tu-token-aqui

# Vector store: "fixed" (1000 chars) o "semantic" (SemanticChunker)
CHROMA_STORE=fixed

# LangSmith Tracing (opcional pero recomendado)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_tu_key_aqui
LANGCHAIN_PROJECT=rag-ontologica
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 5. Agregar documentos PDF

Colocar PDFs dentro de `backend/data/` organizados por marca:

```
backend/data/
├── Toyota/
│   ├── ficha-tecnica-hilux.pdf
│   └── ficha-tecnica-fortuner.pdf
├── Mazda/
│   └── ficha-tecnica-mazda-cx-5-2026.pdf
└── ...
```

### 6. Ejecutar el backend (Terminal 1)

```powershell
cd rag-ing-ontologica
.\.venv\Scripts\Activate
cd backend
uvicorn app:app --reload --port 8001
```

Verificar en: http://localhost:8001/docs

### 7. Ingestar documentos

Desde el frontend (boton "Ingestar") o con curl:

```powershell
# Chunking fijo (1000 chars)
curl -X POST http://localhost:8001/ingest -H "Content-Type: application/json" -d "{\"data_dir\": \"./data\"}"

# Chunking semantico (SemanticChunker)
curl -X POST http://localhost:8001/ingest/semantic -H "Content-Type: application/json" -d "{\"data_dir\": \"./data\"}"
```

### 8. Ejecutar el frontend (Terminal 2)

```powershell
cd rag-ing-ontologica
.\.venv\Scripts\Activate
cd frontend
streamlit run streamlit_app.py
```

Acceder en: http://localhost:8501

### 9. (Opcional) Ejecutar evaluacion batch

```powershell
cd backend
python run_evaluation.py
```

---

## Endpoints de la API

### `POST /ingest` — Ingesta con chunking fijo

```json
// Request
{"data_dir": "./data"}

// Response
{"files_dir": "./data", "raw_docs": 277, "chunks": 584, "ids_added": 584}
```

### `POST /ingest/semantic` — Ingesta con chunking semantico

```json
// Request
{"data_dir": "./data"}

// Response
{"files_dir": "./data", "raw_docs": 277, "chunks": 476, "ids_added": 476, "chunking_method": "semantic"}
```

### `POST /chat/stream` — Chat con streaming SSE

```json
// Request
{"question": "Cual es la potencia del Toyota Hilux?", "session_id": "sesion-1"}
```

Eventos SSE:
- `token` — respuesta final del RAG
- `trazabilidad` — JSON con la ruta completa del grafo, pasos ReAct, metricas, etc.
- `done` — fin del stream

---

## Trazabilidad en el Frontend

El expander "Trazabilidad de la respuesta" muestra:

- **Ruta del grafo**: nodos visitados (ej: `classify_intent → react_agent → generate_grounded → evaluate_grounding → evaluate_metrics`)
- **Clasificacion**: intent detectado y entidades extraidas
- **Pasos ReAct**: cada thought, action y action input del agente
- **Chunks recuperados**: doc_id, pagina y chunk_id de cada chunk usado
- **Verificacion**: score del critico, issues, reintentos
- **Metricas**: Recall@k, Precision@k, MRR, nDCG@k (siempre), Relevance + Faithfulness (en eval mode)
- **Fallback web**: indicador si se uso busqueda en internet

---

## Solucion de problemas

### Error 401 de LangSmith

Si ves `LangSmithAuthError: 401 Unauthorized` al enviar trazas:
1. Verifica que `LANGCHAIN_API_KEY` este correctamente seteada en `.env`
2. **Reinicia el backend** completamente (mata uvicorn y vuelvelo a arrancar) — el proceso debe leer las nuevas variables al iniciar

### Puerto en uso

```powershell
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

### Error de OpenAI 429 (cuota excedida)

Verificar creditos en: https://platform.openai.com/account/billing

### EasyOCR lento en primera ejecucion

Es normal: descarga modelos de ~100 MB la primera vez. Las ejecuciones siguientes usan cache.

### Ingesta semantica falla con "client closed"

Asegurate de tener la version actualizada de `rag_store.py` con singleton de embeddings.

---

## Comandos rapidos (Windows PowerShell)

```powershell
# Backend
cd rag-ing-ontologica && .\.venv\Scripts\Activate && cd backend && uvicorn app:app --reload --port 8001

# Frontend (otra terminal)
cd rag-ing-ontologica && .\.venv\Scripts\Activate && cd frontend && streamlit run streamlit_app.py

# Evaluacion batch
cd rag-ing-ontologica && .\.venv\Scripts\Activate && cd backend && python run_evaluation.py
```

---

## Licencia

Proyecto academico — RAG Ontologica 2026
