# RAG Ontologica — Sistema RAG Agentico para Fichas Tecnicas Vehiculares

Sistema RAG (Retrieval-Augmented Generation) **agentico** especializado en fichas tecnicas
de vehiculos. Combina:

- **Arquitectura ReAct + Reflecting** con LangGraph
- **Transformaciones automaticas de consulta** (HyDE, Query Decomposition)
- **Busqueda hibrida**: vectorial MMR + Knowledge Graph (SPARQL/OWL)
- **Auto-evaluacion** con LLM-as-Judge y metricas de retrieval
- **Fallback web** con retroalimentacion de la base de conocimiento
- **Tracing completo** con LangSmith

---

## Stack Tecnologico

| Componente | Tecnologia |
|------------|-----------|
| Backend API | FastAPI + Uvicorn |
| Orquestacion | LangGraph (grafo de estados con loops) |
| LLM | OpenAI `gpt-5-nano` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Base vectorial | ChromaDB (chunking fijo o semantico) |
| Knowledge Graph | GraphDB (SPARQL/OWL) |
| Busqueda semantica | MMR (Maximal Marginal Relevance) |
| Frontend | Streamlit |
| OCR | EasyOCR |
| Extraccion PDF | pdfplumber |
| Web Search | DuckDuckGo (fallback) |
| Tracing | LangSmith |
| Evaluacion | Recall@k, Precision@k, MRR, nDCG, LLM-as-Judge |

---

## Estructura del Proyecto

```
rag-ing-ontologica/
│
├── backend/
│   ├── app.py                  # API FastAPI: endpoints /ingest, /chat/stream
│   ├── rag_graph.py            # Grafo LangGraph: ReAct + Reflecting + Transformaciones
│   ├── rag_store.py            # ChromaDB: chunking fijo y semantico
│   ├── tools.py                # 10 tools del agente ReAct (incluye KG)
│   ├── prompts.py              # System prompts (clasificador, transformer, ReAct, critico, judges)
│   ├── schemas.py              # Modelos Pydantic
│   ├── evaluation.py           # Metricas: Recall@k, Precision@k, MRR, nDCG, LLM-as-Judge
│   ├── eval_dataset.py         # Dataset de ground truth
│   ├── run_evaluation.py       # Script CLI para evaluacion batch
│   ├── kg_retriever.py         # Funciones SPARQL para consultar el Knowledge Graph
│   ├── kg_rag_agent.py         # [DEPRECATED] Agente legacy KG-RAG (reemplazado por la
│   │                           #   tool consultar_grafo_conocimiento del grafo principal)
│   ├── ontologia/              # Ontologia OWL/RDF de vehiculos
│   │   └── vehiculos_completo.ttl
│   ├── data/                   # PDFs organizados por marca
│   ├── chroma_db/              # Vector store con chunking fijo
│   ├── chroma_db_semantic/     # Vector store con chunking semantico
│   └── test_kg.py              # Script de prueba de la integracion del KG
│
├── frontend/
│   └── streamlit_app.py        # Interfaz de chat con trazabilidad enriquecida
│
├── .env                        # Variables de entorno
├── requeriments.txt            # Dependencias Python
└── README.md
```

---

## Datos Disponibles

- **7 marcas**: Toyota, Mazda, Volkswagen, Peugeot, Opel, MG Emotor, Seat
- **50 modelos** indexados
- **584 chunks** (fijo) / **476 chunks** (semantico) en ChromaDB
- Knowledge Graph en GraphDB con ontologia OWL (`vehiculos_completo.ttl`)
- Metadata por chunk: `source`, `page`, `marca`, `modelo`, `doc_id`, `chunk_id`, `ocr`, `chunking`

---

## Arquitectura del Grafo

```
START
  │
  ▼
┌─────────────────────┐
│  classify_intent    │  Clasifica en Busqueda|Resumen|Comparacion|GENERAL
└──────────┬──────────┘
           │
    ┌──────┴────────────────────────────┐
    │                                   │
 GENERAL                          needs_retrieval
    │                                   │
    ▼                                   ▼
┌──────────────┐         ┌────────────────────────────┐
│answer_general│         │   query_transformer         │  Detecta automaticamente
└──────┬───────┘         │                             │  - HyDE (consulta corta/ambigua)
       │                 │  Transformaciones dinamicas │  - Decomposition (multiples preguntas)
       ▼                 └──────────────┬──────────────┘
      END                                │
                                         ▼
                              ┌────────────────────┐
                              │   react_agent       │  Loop ReAct (max 5 iteraciones)
                              │                     │  Thought → Action → Observation
                              │  10 tools disponibles│  Decide tools dinamicamente
                              └─────────┬──────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ generate_grounded │ ◄────┐
                              └────────┬─────────┘      │
                                       │                │
                                       ▼                │ retry con feedback
                              ┌──────────────────┐      │ (max 3 reintentos)
                              │evaluate_grounding │      │
                              │   (Reflecting)    │      │
                              └────────┬─────────┘      │
                                       │                │
                                       ▼                │
                              ┌──────────────────┐      │
                              │ evaluate_metrics  │      │
                              └────────┬─────────┘      │
                                       │                │
                           ┌───────────┼────────────────┘
                           │           │
                       APROBADO   RECHAZADO + 3 reintentos
                           │           │
                           ▼           ▼
                          END   ┌──────────────┐
                                │ web_fallback  │  Busqueda DuckDuckGo
                                │ + ingesta KB  │  Retroalimenta ChromaDB
                                └──────┬───────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │ evaluate_metrics  │
                              └────────┬─────────┘
                                       │
                                       ▼
                                      END
```

### Patrones implementados

| Patron | Implementacion |
|--------|----------------|
| **ReAct (Reasoning + Acting)** | El agente razona (Thought), elige tool (Action), observa resultado, decide si continuar. Max 5 iteraciones. |
| **Reflecting** | El critico evalua la respuesta y da feedback. Si score < 0.5, reintenta con correccion (max 3 veces). |
| **Query Transformer** | Detecta automaticamente si la consulta necesita HyDE (corta/ambigua) o Decomposition (multiples preguntas/condicionales). |
| **Web Fallback con feedback** | Tras 3 fallos de reflexion, busca en DuckDuckGo, responde, **e ingiere los resultados a ChromaDB** para futuras consultas. |
| **Memory conversacional** | `last_model`/`last_make` persisten entre turnos via `MemorySaver`. |

> **Nota: arquitectura unificada.** El frontend siempre usa el grafo principal
> via `/chat/stream`. El Knowledge Graph se accede como una **tool del agente
> ReAct** (`consultar_grafo_conocimiento`), no como un flujo separado. Esto
> garantiza que toda consulta pase por classify_intent (filtrando preguntas
> GENERAL), query_transformer, ReAct, Reflecting y evaluate_metrics. El archivo
> `kg_rag_agent.py` y el endpoint `/kg_chat` quedan deprecados.

---

## Transformaciones Automaticas de Consulta

El nodo `query_transformer` analiza la consulta del usuario **antes** del agente ReAct y aplica transformaciones segun el tipo de pregunta.

> **Optimizacion de latencia**: el `query_transformer` salta la llamada LLM
> cuando el intent es `Resumen` o `Comparación`, ya que estos tienen tools
> dedicadas (`resumir_ficha`, `comparar_modelos`) que ya manejan la complejidad.
> Solo ejecuta deteccion de HyDE/Decomposition para intents tipo `Búsqueda`.

### Detecciones automaticas

#### 1. HyDE (Hypothetical Document Embeddings)

Activa HyDE cuando la consulta es **corta o ambigua**:
- Menos de 6 palabras significativas
- Usa pronombres sin contexto ("cuanto mide eso")
- Una sola palabra clave ("potencia", "consumo")
- No menciona modelo o marca especifica

Cuando se activa, el agente prioriza `buscar_hyde` (genera doc hipotetico → busca chunks similares).

#### 2. Query Decomposition

Activa decomposition cuando la consulta tiene **multiples preguntas o condicionales**:
- Multiples signos de interrogacion
- Conjunciones tipo "y tambien", "ademas", "por otro lado"
- Listado de aspectos
- Condicionales tipo "si X, entonces Y"

Cuando se activa, descompone la pregunta en 2-4 sub-consultas y le indica al agente que las resuelva por separado.

---

## Tools del Agente ReAct (10 tools)

| Tool | Funcion | Tipo |
|------|---------|------|
| `buscar_vectorial` | Busqueda semantica con MMR y filtros de metadata | Vectorial |
| `buscar_hyde` | HyDE: documento hipotetico para mejor retrieval | Vectorial |
| `buscar_especificacion` | Dato tecnico puntual de un modelo | Vectorial |
| `buscar_por_marca` | Todos los modelos de una marca | Vectorial |
| `comparar_modelos` | Tabla comparativa entre 2 modelos | Vectorial + LLM |
| `resumir_ficha` | Resumen estructurado de un modelo | Vectorial + LLM |
| `descomponer_pregunta` | Divide preguntas complejas en sub-preguntas | LLM |
| `listar_modelos_disponibles` | Catalogo completo indexado | ChromaDB |
| `consultar_grafo_conocimiento` | **SPARQL sobre ontologia OWL** | **Knowledge Graph** |
| `buscar_web` | DuckDuckGo (solo en web_fallback) | Web |

### Knowledge Graph: `consultar_grafo_conocimiento`

Acciones disponibles:
- `especificaciones`: peso, longitud, baul, precio, anyo
- `motor`: potencia, cilindrada, combustible, autonomia, bateria
- `comparar`: dos modelos lado a lado
- `por_marca`: lista todos los modelos de una marca
- `electricos`: filtra electricos por autonomia minima
- `seguridad`: sistemas de seguridad del modelo

El agente usa el KG cuando necesita **datos estructurados precisos** (numericos exactos, relaciones formales) en lugar de chunks de texto.

### Busqueda MMR

Todas las tools vectoriales usan **Maximal Marginal Relevance** (`max_marginal_relevance_search`) en lugar de `similarity_search` para garantizar relevancia + diversidad.

| Tool | k | fetch_k | lambda_mult | Justificacion |
|------|---|---------|-------------|---------------|
| `buscar_especificacion` | 6 | 20 | **0.7** | Dato puntual: prioriza relevancia |
| `buscar_por_marca` | 10 | 30 | **0.5** | Catalogo: necesita diversidad de modelos |
| `comparar_modelos` | 8/modelo | 25 | **0.5** | Comparacion: cubrir distintas secciones |
| `resumir_ficha` | 10 | 30 | **0.5** | Resumen: maxima cobertura |
| `buscar_vectorial` | variable | k×3 | **0.7** | Tool principal: balance general |
| `buscar_hyde` | variable | k×3 | **0.7** | HyDE: doc hipotetico ya sesga |

---

## Reflecting + Web Fallback con Retroalimentacion

### Reflecting Loop

Despues de generar la respuesta, el `evaluate_grounding` evalua:
1. **supported_by_context**: la respuesta esta soportada por el contexto?
2. **has_citations**: incluye citas en formato `[doc_id=X; pagina=Y]`?
3. **complete_enough**: es suficientemente completa?
4. **score** (0.0-1.0)

Si `score < 0.5` y quedan reintentos (max 3), inyecta el feedback como correccion en el siguiente `generate_grounded`.

### Web Fallback con Retroalimentacion de KB

Tras 3 reintentos fallidos, el flujo escala a `web_fallback`:

1. **Busca en DuckDuckGo** con la pregunta original
2. **Genera respuesta** usando los resultados web
3. **Retroalimenta ChromaDB**: ingiere los resultados como nuevos chunks con metadata:
   - `origen=web_fallback`
   - `pregunta_origen=<pregunta del usuario>`
   - `timestamp=<YYYYMMDD_HHMMSS>`

**Ventaja**: la proxima vez que alguien pregunte algo similar, el RAG ya tiene esa informacion en la base vectorial — no necesita hacer fallback web de nuevo.

---

## Modulo de Evaluacion

### Metricas de Retrieval (con ground truth)
- **Recall@k**: fraccion de docs relevantes recuperados
- **Precision@k**: fraccion de docs recuperados que son relevantes
- **MRR (Mean Reciprocal Rank)**: posicion del primer doc relevante
- **nDCG@k**: relevancia ponderada por posicion

### LLM-as-Judge (sin ground truth)
- **Relevance**: la respuesta es relevante para la pregunta?
- **Faithfulness**: la respuesta es fiel al contexto (sin alucinacion)?

### Visualizacion en trazabilidad
Las metricas de retrieval se calculan **en cada query** y se muestran en el expander de trazabilidad. Las metricas LLM-as-Judge solo se ejecutan en modo evaluacion batch (`eval_mode=True`) para no agregar latencia al chat.

### Script de evaluacion batch

```powershell
cd backend
python run_evaluation.py
```

Ejecuta las 17 preguntas del dataset, calcula todas las metricas (incluyendo LLM-as-Judge) y guarda resultados en `eval_results.json`.

---

## Tracing con LangSmith

El sistema esta integrado con [LangSmith](https://smith.langchain.com/) para tracing completo del grafo. Cada nodo tiene `RunnableLambda(...).with_config({run_name, tags, metadata})`, asi que LangSmith captura automaticamente el arbol de ejecucion completo.

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

---

## Configuracion en Windows

### Requisitos previos

- **Python 3.12** — https://www.python.org/downloads/
- **Git** — https://git-scm.com/
- **GraphDB Free** — https://www.ontotext.com/products/graphdb/download/ (para el Knowledge Graph)
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
CHROMA_STORE=semantic

# LangSmith Tracing (opcional)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_tu_key_aqui
LANGCHAIN_PROJECT=rag-ontologica
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 5. Configurar GraphDB para el Knowledge Graph

1. Instalar GraphDB Free Desktop
2. Iniciar GraphDB (queda en http://localhost:7200)
3. Crear repositorio:
   - **Setup → Repositories → Create new repository → GraphDB Repository**
   - **Repository ID**: `vehiculos`
   - **Ruleset**: `OWL2-RL (Optimized)` (para inferencias)
4. Cargar la ontologia:
   - **Import → User data → Upload RDF files**
   - Seleccionar `backend/ontologia/vehiculos_completo.ttl`
5. Verificar con el script de prueba:

```powershell
cd backend
python test_kg.py
```

### 6. Agregar documentos PDF

```
backend/data/
├── Toyota/
│   ├── ficha-tecnica-hilux.pdf
│   └── ficha-tecnica-fortuner.pdf
├── Mazda/
│   └── ficha-tecnica-mazda-cx-5-2026.pdf
└── ...
```

### 7. Ejecutar el backend (Terminal 1)

```powershell
cd rag-ing-ontologica
.\.venv\Scripts\Activate
cd backend
uvicorn app:app --reload --port 8001
```

### 8. Ingestar documentos

Desde el frontend (boton "Ingestar") o con curl:

```powershell
# Chunking semantico (recomendado)
curl -X POST http://localhost:8001/ingest/semantic -H "Content-Type: application/json" -d "{\"data_dir\": \"./data\"}"

# Chunking fijo (1000 chars)
curl -X POST http://localhost:8001/ingest -H "Content-Type: application/json" -d "{\"data_dir\": \"./data\"}"
```

### 9. Ejecutar el frontend (Terminal 2)

```powershell
cd rag-ing-ontologica
.\.venv\Scripts\Activate
cd frontend
streamlit run streamlit_app.py
```

Acceder en: http://localhost:8501

### 10. (Opcional) Ejecutar evaluacion batch

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
- `trazabilidad` — JSON con la ruta completa del grafo, transformaciones, pasos ReAct, metricas, etc.
- `done` — fin del stream

---

## Trazabilidad en el Frontend

El expander "Trazabilidad de la respuesta" muestra:

- **Ruta del grafo**: nodos visitados (ej: `classify_intent → query_transformer → react_agent → generate_grounded → evaluate_grounding → evaluate_metrics`)
- **Clasificacion**: intent detectado y entidades extraidas
- **Transformaciones de consulta**: HyDE y/o Decomposition aplicadas con justificacion
- **Pasos ReAct**: cada thought, action y action input del agente
- **Chunks recuperados**: doc_id, pagina y chunk_id (max 12 despues de dedupe)
- **Chunks dedupe**: `{antes, despues, cap_aplicado}` mostrando cuantos chunks se descartaron
- **Verificacion**: score del critico, issues, reintentos
- **Metricas**: Recall@k, Precision@k, MRR, nDCG@k (siempre), Relevance + Faithfulness (en eval mode)
- **Web fallback**: indicador si se uso busqueda en internet
- **Retroalimentacion KB**: chunks web ingeridos a ChromaDB

---

## Solucion de problemas

### Error 401 de LangSmith

Si ves `LangSmithAuthError: 401 Unauthorized`:
1. Verifica que `LANGCHAIN_API_KEY` este correctamente seteada en `.env`
2. **Reinicia el backend** completamente para que cargue las nuevas variables

### GraphDB no responde

Si la tool `consultar_grafo_conocimiento` retorna error de conexion:
1. Verifica que GraphDB Desktop este corriendo (http://localhost:7200)
2. Verifica que el repositorio `vehiculos` exista y tenga la ontologia cargada
3. Ejecuta `python backend/test_kg.py` para diagnosticar

### Puerto en uso

```powershell
netstat -ano | findstr :8001
taskkill /PID <PID> /F
```

### Error de OpenAI 429 (cuota excedida)

Verificar creditos en: https://platform.openai.com/account/billing

### EasyOCR lento en primera ejecucion

Es normal: descarga modelos de ~100 MB la primera vez.

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

# Test del Knowledge Graph
cd rag-ing-ontologica && .\.venv\Scripts\Activate && cd backend && python test_kg.py
```

---

## Flujo end-to-end de una consulta

1. **Usuario** ingresa la consulta en el frontend
2. **classify_intent** clasifica en Busqueda/Resumen/Comparacion/GENERAL
   - **Regla critica**: si la pregunta menciona CUALQUIER marca o modelo del corpus
     (Toyota, Hilux, CX-5, etc.), NUNCA es GENERAL — siempre Busqueda/Resumen/Comparacion
   - GENERAL solo para preguntas conceptuales sin marca/modelo (ej: "que es un torque?")
3. **query_transformer** detecta automaticamente si aplica HyDE o Decomposition (skip para Resumen/Comparacion)
4. **react_agent** decide tools dinamicamente (vectorial MMR, KG, HyDE, etc.) en loop max 5 iteraciones
5. **generate_grounded** genera respuesta preliminar con citas obligatorias
6. **evaluate_grounding** evalua la respuesta (reflecting); si rechazada, reintenta con feedback (max 3)
7. **evaluate_metrics** calcula Recall@k, Precision@k, MRR, nDCG@k (+ LLM-as-Judge en eval mode)
8. **Si rechazada tras 3 intentos** → `web_fallback`:
   - Busca en DuckDuckGo
   - Genera respuesta desde resultados web
   - **Ingiere los resultados a ChromaDB** (retroalimentacion del KB)
   - Pasa por evaluate_metrics
9. Retorna respuesta + trazabilidad completa via SSE al frontend

---

## Configuracion del grafo

Limites configurables en `backend/rag_graph.py`:

| Constante | Valor | Descripcion |
|-----------|-------|-------------|
| `MAX_REACT_ITERATIONS` | `5` | Maximo de pasos Thought/Action/Observation del agente ReAct |
| `MAX_RETRIES` | `3` | Maximo de reintentos del reflecting loop antes de escalar a web_fallback |
| `MAX_FINAL_CHUNKS` | `12` | Cap de chunks finales pasados a `generate_grounded` (despues de dedupe) |

### Deduplicacion + cap de chunks

Despues de varias iteraciones del agente ReAct, los chunks acumulados pueden tener
duplicados (la misma seccion del PDF retornada por tools distintas). Al final del nodo
`react_agent` se aplica:

1. **Deduplicacion** por `chunk_id` (o `doc_id+page` como fallback)
2. **Cap** a los primeros 12 chunks unicos preservando el orden de aparicion
3. La trazabilidad incluye un campo `chunks_dedupe: {antes, despues, cap_aplicado}`
   para que veas cuantos chunks habia antes y despues del filtrado

### Filtro anti-contaminacion del web_fallback

`buscar_vectorial` y `buscar_hyde` aplican un filtro `{"origen": {"$ne": "web_fallback"}}`
para que los chunks ingeridos por web_fallback en corridas previas no contaminen las
busquedas vectoriales normales. Los chunks web quedan aislados y solo se usan si
explicitamente se invoca `web_fallback`.

### Optimizaciones de latencia aplicadas

- **Skip query_transformer LLM call** para intents `Resumen` y `Comparación`
- **LLM-as-Judge solo en `eval_mode=True`** (script batch), no en chat normal
- **Scratchpad ReAct truncado** a 250 chars por observation para reducir tokens
- **Singleton de embeddings** HuggingFace para evitar reload del modelo
- **Routing condicional** correcto en `route_after_metrics` (`< MAX_RETRIES` estricto)
- **Sanitizacion `_sanitize_for_llm()`** elimina caracteres de control y null bytes
  de chunks corruptos de PDFs (evita error 400 de OpenAI por JSON invalido)
- **Dedupe + cap de chunks** reduce contexto enviado a `generate_grounded`

---

## Licencia

Proyecto academico — RAG Ontologica 2026
