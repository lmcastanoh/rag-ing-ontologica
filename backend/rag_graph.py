# backend/rag_graph.py
# ==============================================================================
# Grafo LangGraph principal del sistema RAG para fichas tecnicas vehiculares.
#
# Arquitectura: ReAct Agent + Reflecting
#
# Flujo: classify_intent -> query_transformer -> react_agent -> generate_grounded ->
#        evaluate_grounding -> evaluate_metrics -> [retry | web_fallback | END]
#
# 4 rutas posibles:
#   A) GENERAL:      classify -> answer_general -> END (sin retrieval)
#   B) RAG + ReAct:  classify -> query_transformer -> react_agent -> generate -> evaluate -> metrics -> END
#   C) Retry:        ... -> evaluate (rechazada) -> metrics -> generate (con feedback)
#   D) Web Fallback: ... -> evaluate (3 fallos) -> metrics -> web_fallback -> END
#
# Query Transformer (transformaciones dinamicas):
#   - HyDE: detecta consultas cortas/ambiguas y le indica al ReAct usar buscar_hyde
#   - Decomposition: detecta consultas con multiples preguntas y las descompone
#
# Features:
#   - ReAct agent: razona sobre que tools usar (buscar_vectorial, hyde,
#     descomponer, comparar, resumir, etc.) en loop de max 7 iteraciones
#   - Reflecting: el critico puede rechazar y forzar reintentos (max 3)
#   - Web fallback: tras 3 reintentos fallidos, busca en internet
#   - Memory: last_model/last_make persisten entre turnos con reducer _keep_latest
#   - Trazabilidad: cada nodo registra su ruta, decisiones, pasos ReAct, chunks
# ==============================================================================
from __future__ import annotations

import json
import re
from typing import Annotated, Any, List, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.runnables import RunnableLambda

from prompts import (
    CLASSIFIER_SYSTEM_PROMPT,
    CLASSIFIER_USER_TEMPLATE,
    GROUNDED_GENERATION_SYSTEM_PROMPT,
    GROUNDED_GENERATION_USER_TEMPLATE,
    GROUNDING_CRITIC_SYSTEM_PROMPT,
    GROUNDING_CRITIC_USER_TEMPLATE,
    QUERY_TRANSFORMER_SYSTEM_PROMPT,
    QUERY_TRANSFORMER_USER_TEMPLATE,
    REACT_AGENT_SYSTEM_PROMPT,
    REACT_AGENT_USER_TEMPLATE,
    WEB_FALLBACK_SYSTEM_PROMPT,
    WEB_FALLBACK_USER_TEMPLATE,
)
from eval_dataset import EVAL_DATASET
from evaluation import compute_retrieval_metrics, compute_llm_judge_metrics
from schemas import (
    GroundingEvaluation,
    IntentClassification,
    QueryTransformation,
    eval_to_dict,
    intent_to_dict,
)
from tools import (
    buscar_especificacion,
    buscar_hyde,
    buscar_por_marca,
    buscar_vectorial,
    buscar_web,
    comparar_modelos,
    consultar_grafo_conocimiento,
    descomponer_pregunta,
    listar_modelos_disponibles,
    resumir_ficha,
    _retrieval_context,
    _fix_doubled_text,
)


# ── Configuracion ──────────────────────────────────────────────────────────
MAX_REACT_ITERATIONS = 5   # Maximo de pasos Thought/Action/Observation (era 7, reducido para latencia)
MAX_RETRIES = 3            # Maximo de reintentos del reflecting loop
MAX_FINAL_CHUNKS = 12      # Cap de chunks finales pasados a generate_grounded (despues de dedupe)


def _keep_latest(existing: Optional[str], new: Optional[str]) -> Optional[str]:
    """Reducer para last_model y last_make en RAGState.

    Conserva el ultimo valor no-None entre turnos conversacionales.
    """
    return new if new is not None else existing


# ── Estado del grafo ─────────────────────────────────────────────────────────
class RAGState(TypedDict):
    """Estado compartido entre todos los nodos del grafo LangGraph.

    Campos:
        question:        Pregunta actual del usuario.
        docs:            Documentos/contexto recopilado por el agente ReAct.
        answer:          Respuesta final generada.
        messages:        Historial de mensajes (con reducer add_messages).
        intent:          Resultado del clasificador (dict de IntentClassification).
        eval_result:     Resultado del critico (dict de GroundingEvaluation).
        trazabilidad:    Dict acumulativo con ruta, decisiones, pasos ReAct, chunks.
        last_model:      Ultimo modelo mencionado (persiste entre turnos).
        last_make:       Ultima marca mencionada (persiste entre turnos).
        retry_count:     Contador de reintentos del reflecting loop (max 3).
        critic_feedback: Lista de issues del critico para inyectar en el reintento.
        react_steps:     Pasos del agente ReAct (thought/action/observation).
        react_iteration: Numero de iteraciones del agente ReAct.
        web_search_used: Flag: True si se uso busqueda web como fallback.
        eval_mode:       Flag: True activa LLM-as-Judge (solo para evaluacion batch).
    """

    question: str
    docs: List[Document]
    answer: str
    messages: Annotated[List[BaseMessage], add_messages]
    intent: Optional[dict[str, Any]]
    eval_result: Optional[dict[str, Any]]
    trazabilidad: dict[str, Any]
    last_model: Annotated[Optional[str], _keep_latest]
    last_make: Annotated[Optional[str], _keep_latest]
    retry_count: int
    critic_feedback: Optional[list[str]]
    react_steps: Optional[list[dict[str, Any]]]
    react_iteration: int
    web_search_used: bool
    eval_mode: bool
    query_transformations: Optional[dict[str, Any]]


# ── Funciones auxiliares ─────────────────────────────────────────────────────

def _history_text(messages: List[BaseMessage], max_items: int = 8) -> str:
    """Construye texto de historial conversacional para el clasificador y el agente.

    Filtra ToolMessages y AIMessages con tool_calls (datos internos).
    Solo conserva mensajes Human y AI con contenido textual real.
    """
    if not messages:
        return ""
    conversational: list[BaseMessage] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            continue
        conversational.append(m)
    items = conversational[-max_items:]
    lines: list[str] = []
    for m in items:
        role = "Asistente" if isinstance(m, AIMessage) else "Usuario"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if content.strip():
            lines.append(f"{role}: {content.strip()}")
    return "\n".join(lines)


def _sanitize_for_llm(text: str) -> str:
    """Limpia texto antes de enviarlo a OpenAI.

    Elimina caracteres de control y null bytes que rompen la serializacion JSON
    del request body. Estos suelen venir de PDFs corruptos extraidos con OCR
    o pdfplumber.
    """
    if not text:
        return ""
    # Eliminar caracteres de control excepto \n, \r, \t
    cleaned = "".join(
        c for c in text if c == "\n" or c == "\r" or c == "\t" or ord(c) >= 32
    )
    # Eliminar null bytes residuales
    cleaned = cleaned.replace("\x00", "")
    return cleaned


def _retrieved_chunk_payload(docs: List[Document]) -> list[dict[str, Any]]:
    """Genera payload resumido de los chunks para trazabilidad y el critico."""
    chunks: list[dict[str, Any]] = []
    for d in docs:
        md = d.metadata or {}
        chunk = {
            "doc_id": md.get("doc_id") or md.get("source"),
            "source": md.get("source"),
            "page": md.get("page"),
        }
        if md.get("chunk_id"):
            chunk["chunk_id"] = md["chunk_id"]
        chunks.append(chunk)
    return chunks


# ── Parser de respuestas ReAct ──────────────────────────────────────────────

def _parse_react_response(text: str) -> tuple[str, str, dict]:
    """Parsea la respuesta del LLM en formato Thought/Action/Action Input.

    Extrae los tres componentes usando regex. Si el parsing falla,
    retorna valores de fallback que terminan el loop.

    Returns:
        Tupla (thought, action, action_input_dict)
    """
    thought = ""
    action = "FINISH"
    action_input: dict = {"summary": "No se pudo parsear la respuesta del agente."}

    # Extraer Thought
    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
    if thought_match:
        thought = thought_match.group(1).strip()

    # Extraer Action
    action_match = re.search(r"Action:\s*(.+?)(?=\nAction Input:|\Z)", text, re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()

    # Extraer Action Input (JSON)
    input_match = re.search(r"Action Input:\s*(.+)", text, re.DOTALL)
    if input_match:
        raw = input_match.group(1).strip()
        try:
            action_input = json.loads(raw)
        except json.JSONDecodeError:
            # Intentar extraer solo el primer JSON valido
            json_match = re.search(r"\{[^}]+\}", raw)
            if json_match:
                try:
                    action_input = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

    return thought, action, action_input


# Regex para extraer los dos modelos de una query de comparacion.
_RE_VS = re.compile(
    r"(?:compar[aáeo]\w*|diferencias?\s+entre)\s+(?:(?:el|la|los|las|del|al)\s+)?"
    r"(.+?)\s+(?:vs\.?|versus|contra|y|con)\s+(?:(?:el|la|los|las|del|al)\s+)?(.+)",
    re.IGNORECASE,
)


def _extract_comparison_models(question: str) -> list[str] | None:
    """Extrae los dos modelos de una query de comparacion usando regex."""
    m = _RE_VS.search(question)
    if not m:
        return None
    return [m.group(1).strip(), m.group(2).strip()]


# ── Construccion del grafo ───────────────────────────────────────────────────

def build_rag_graph():
    """Construye y compila el grafo LangGraph con arquitectura ReAct + Reflecting.

    Inicializa:
    - 3 instancias de LLM (router, answer, critic) todas gpt-5-nano
    - 1 instancia de LLM para el agente ReAct
    - 9 tools disponibles para el agente
    - 5 nodos del grafo con sus edges y conditional edges
    - MemorySaver como checkpointer para persistir estado entre turnos

    Returns:
        Grafo LangGraph compilado listo para .invoke() o .ainvoke()
    """
    # Mapa de tools disponibles para el agente ReAct
    tool_map = {
        "buscar_vectorial": buscar_vectorial,
        "buscar_hyde": buscar_hyde,
        "buscar_especificacion": buscar_especificacion,
        "buscar_por_marca": buscar_por_marca,
        "comparar_modelos": comparar_modelos,
        "resumir_ficha": resumir_ficha,
        "descomponer_pregunta": descomponer_pregunta,
        "listar_modelos_disponibles": listar_modelos_disponibles,
        "consultar_grafo_conocimiento": consultar_grafo_conocimiento,
    }

    # LLMs especializados (todos gpt-5-nano con diferentes temperatures)
    router_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)      # Clasificador
    transformer_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)  # Query transformer
    react_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)        # Agente ReAct
    answer_llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)     # Generador
    critic_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)       # Critico

    # ── Nodo 1: classify_intent ──────────────────────────────────────────
    def classify_intent(state: RAGState) -> dict[str, Any]:
        """Clasifica la intencion del usuario y determina la ruta del grafo.

        Proceso:
        1. Construye input con historial conversacional + pregunta actual
        2. LLM clasifica en Busqueda|Resumen|Comparacion|GENERAL
        3. Actualiza last_model/last_make para follow-ups futuros
        4. Fallback de memory: si no hay modelo, usa last_model del turno anterior
        5. Registra clasificacion y decision en trazabilidad
        """
        question = state["question"]
        history = _history_text(state.get("messages") or [])
        classifier_input = (
            f"Historial reciente:\n{history}\n\nConsulta actual:\n{question}"
            if history
            else question
        )
        structured = router_llm.with_structured_output(IntentClassification)
        result: IntentClassification = structured.invoke(
            [
                SystemMessage(content=CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=CLASSIFIER_USER_TEMPLATE.format(question=classifier_input)),
            ]
        )
        intent_data = intent_to_dict(result)

        entities = intent_data.get("entities") or {}
        updates: dict[str, Any] = {"intent": intent_data}
        current_model = entities.get("model")
        current_make = entities.get("make")

        comparison_models = _extract_comparison_models(question)
        if comparison_models:
            last_mentioned = comparison_models[-1]
            parts = last_mentioned.split()
            if len(parts) >= 2:
                updates["last_make"] = parts[0]
                updates["last_model"] = " ".join(parts[1:])
            else:
                updates["last_model"] = last_mentioned
        elif current_model:
            updates["last_model"] = current_model
            if current_make:
                updates["last_make"] = current_make

        # Fallback de memory: si no hay modelo, usar last_model del estado
        if not current_model and intent_data.get("needs_retrieval"):
            prev_model = state.get("last_model")
            prev_make = state.get("last_make")
            if prev_model:
                intent_data["entities"]["model"] = prev_model
                if prev_make and not current_make:
                    intent_data["entities"]["make"] = prev_make
                intent_data["_model_from_memory"] = True

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = ["classify_intent"]
        traza["clasificacion"] = intent_data
        traza["decision"] = {
            "ruta_seleccionada": "rag" if intent_data["needs_retrieval"] else "general",
            "motivo": intent_data["reason"],
        }
        if intent_data.get("_model_from_memory"):
            traza["model_memory_fallback"] = {
                "model": state.get("last_model"),
                "make": state.get("last_make"),
            }
        updates["trazabilidad"] = traza
        return updates

    # ── Nodo 1.5: query_transformer ──────────────────────────────────────
    def query_transformer(state: RAGState) -> dict[str, Any]:
        """Analiza la consulta y aplica transformaciones dinamicas (HyDE, decomposition).

        Solo se ejecuta si needs_retrieval=True (no para preguntas GENERAL).

        Detecta automaticamente:
        1. **HyDE necesario**: si la consulta es corta/ambigua, marca el flag para
           que el agente ReAct priorice buscar_hyde sobre buscar_vectorial.
        2. **Decomposition necesaria**: si la consulta tiene multiples preguntas
           o condicionales, llama a descomponer_pregunta y guarda las sub-consultas.

        Las transformaciones se inyectan como hints en el prompt del react_agent
        para que las use desde la primera iteracion.
        """
        question = state["question"]
        intent = state.get("intent") or {}
        intent_name = intent.get("intent", "")

        # Solo transformar si necesita retrieval (saltar GENERAL)
        if not intent.get("needs_retrieval", True):
            return {}

        # Optimizacion: saltar la llamada LLM cuando el intent es Resumen o Comparacion.
        # Estos intents tienen tools dedicadas (resumir_ficha, comparar_modelos) que
        # ya manejan la complejidad. HyDE/Decomposition son utiles principalmente para
        # consultas tipo Busqueda donde el agente puede beneficiarse de hints.
        if intent_name in ("Resumen", "Comparación"):
            transformations = {
                "needs_hyde": False,
                "hyde_reason": f"skipped (intent={intent_name} usa tool dedicada)",
                "needs_decomposition": False,
                "sub_queries": [],
                "decomposition_reason": f"skipped (intent={intent_name} usa tool dedicada)",
            }
            traza = dict(state.get("trazabilidad") or {})
            traza["ruta"] = traza.get("ruta", []) + ["query_transformer"]
            traza["query_transformations"] = transformations
            return {
                "query_transformations": transformations,
                "trazabilidad": traza,
            }

        # Detectar transformaciones necesarias con LLM estructurado (solo Busqueda)
        try:
            structured = transformer_llm.with_structured_output(QueryTransformation)
            result: QueryTransformation = structured.invoke(
                [
                    SystemMessage(content=QUERY_TRANSFORMER_SYSTEM_PROMPT),
                    HumanMessage(content=QUERY_TRANSFORMER_USER_TEMPLATE.format(
                        question=question,
                    )),
                ]
            )
            transformations = result.model_dump()
        except Exception as e:
            # Si falla el analisis, no aplicar transformaciones (degradacion segura)
            transformations = {
                "needs_hyde": False,
                "hyde_reason": f"error en analisis: {e}",
                "needs_decomposition": False,
                "sub_queries": [],
                "decomposition_reason": "skipped por error",
            }

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["query_transformer"]
        traza["query_transformations"] = transformations

        return {
            "query_transformations": transformations,
            "trazabilidad": traza,
        }

    # ── Nodo 2a: answer_general ──────────────────────────────────────────
    def answer_general(state: RAGState) -> dict[str, Any]:
        """Responde preguntas generales sin retrieval documental."""
        question = state["question"]
        history = _history_text(state.get("messages") or [])
        user_prompt = (
            f"Historial reciente:\n{history}\n\nConsulta actual:\n{question}"
            if history
            else question
        )

        response = answer_llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Eres un asistente automotriz. "
                        "Responde de forma clara y concisa. "
                        "No uses tools ni recuperación documental para esta respuesta."
                    )
                ),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["answer_general"]
        traza["chunks_recuperados"] = []
        return {
            "answer": answer,
            "messages": [response],
            "trazabilidad": traza,
        }

    # ── Nodo 2b: react_agent ────────────────────────────────────────────
    def react_agent(state: RAGState) -> dict[str, Any]:
        """Agente ReAct: razona sobre que herramientas usar en un loop controlado.

        Proceso:
        1. Construye contexto con historial y memoria conversacional
        2. En cada iteracion: LLM razona (Thought) -> elige tool (Action) -> ejecuta
        3. Acumula observaciones de cada tool como contexto
        4. Termina cuando el LLM dice FINISH o alcanza MAX_REACT_ITERATIONS
        5. Registra cada paso en trazabilidad (react_steps)

        Retorna: docs (contexto acumulado), react_steps, trazabilidad
        """
        question = state["question"]
        intent = state.get("intent") or {}
        history = _history_text(state.get("messages") or [])

        # Contexto de memoria conversacional
        history_context = ""
        if history:
            history_context = f"\nHistorial de conversacion:\n{history}"

        memory_context = ""
        last_model = state.get("last_model")
        last_make = state.get("last_make")
        if last_model:
            memory_context = f"\nContexto del turno anterior: modelo={last_model}"
            if last_make:
                memory_context += f", marca={last_make}"

        # Informacion del clasificador para guiar al agente
        intent_name = intent.get("intent", "")
        entities = intent.get("entities") or {}
        if entities.get("model") or entities.get("make"):
            entity_hint = f"\nEntidades detectadas: modelo={entities.get('model')}, marca={entities.get('make')}"
            memory_context += entity_hint

        # Hints de transformaciones dinamicas detectadas por query_transformer
        transformations = state.get("query_transformations") or {}
        if transformations.get("needs_hyde"):
            memory_context += (
                f"\n\n[TRANSFORMACION SUGERIDA] La consulta es corta/ambigua "
                f"({transformations.get('hyde_reason', '')}). "
                f"PRIORIZA usar buscar_hyde como primera tool en lugar de buscar_vectorial."
            )
        if transformations.get("needs_decomposition"):
            sub_qs = transformations.get("sub_queries", [])
            if sub_qs:
                sub_qs_text = "\n".join(f"  - {sq}" for sq in sub_qs)
                memory_context += (
                    f"\n\n[TRANSFORMACION APLICADA] La consulta fue descompuesta en sub-preguntas "
                    f"({transformations.get('decomposition_reason', '')}):\n"
                    f"{sub_qs_text}\n"
                    f"Resuelve cada sub-pregunta haciendo busquedas independientes y combina los resultados."
                )

        react_steps: list[dict[str, Any]] = []
        all_observations: list[str] = []
        scratchpad = ""

        for i in range(MAX_REACT_ITERATIONS):
            # Construir prompt con scratchpad acumulado
            user_msg = REACT_AGENT_USER_TEMPLATE.format(
                question=question,
                history_context=history_context,
                memory_context=memory_context,
            )
            if scratchpad:
                user_msg += f"\n\nPasos anteriores:\n{scratchpad}"

            # Llamar al LLM
            response = react_llm.invoke(
                [
                    SystemMessage(content=REACT_AGENT_SYSTEM_PROMPT),
                    HumanMessage(content=user_msg),
                ]
            )
            text = response.content if isinstance(response.content, str) else str(response.content)

            # Parsear Thought / Action / Action Input
            thought, action, action_input = _parse_react_response(text)

            # Si el agente decide terminar
            if action == "FINISH":
                react_steps.append({
                    "step": i + 1,
                    "thought": thought,
                    "action": "FINISH",
                    "action_input": action_input,
                    "observation": "",
                })
                break

            # Ejecutar la tool seleccionada
            tool_fn = tool_map.get(action)
            if tool_fn:
                try:
                    observation = tool_fn.invoke(action_input)
                except Exception as e:
                    observation = f"Error ejecutando {action}: {e}"
            else:
                observation = (
                    f"Herramienta desconocida: '{action}'. "
                    f"Disponibles: {', '.join(tool_map.keys())}"
                )

            # Sanitizar la observacion para evitar caracteres de control que
            # rompan la serializacion JSON del request a OpenAI
            observation_str = _sanitize_for_llm(str(observation))

            react_steps.append({
                "step": i + 1,
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "observation": observation_str[:500],  # Truncar para trazabilidad
            })

            all_observations.append(observation_str)

            # Construir scratchpad para la siguiente iteracion
            scratchpad += (
                f"\nThought: {thought}\n"
                f"Action: {action}\n"
                f"Action Input: {json.dumps(action_input, ensure_ascii=False)}\n"
                f"Observation: {observation_str[:250]}\n"
            )

        # Construir Documents desde las observaciones para generate_grounded.
        # Extraer metadata real de las cabeceras [doc_id=...; página=...] de los chunks.
        docs: List[Document] = []
        for obs in all_observations:
            if not obs.strip():
                continue
            # Separar bloques individuales de chunks (separados por ---)
            blocks = re.split(r"\n-{3,}\n", obs)
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                # Intentar extraer metadata de la cabecera del chunk
                header_match = re.match(
                    r"\[doc_id=([^;]+);\s*página=([^;\]]+)(?:;\s*chunk_id=([^\]]+))?\]",
                    block,
                )
                if header_match:
                    doc_id = header_match.group(1).strip()
                    page = header_match.group(2).strip()
                    chunk_id = header_match.group(3).strip() if header_match.group(3) else None
                    content = block[header_match.end():].strip()
                    meta = {"source": doc_id, "doc_id": doc_id, "page": page}
                    if chunk_id:
                        meta["chunk_id"] = chunk_id
                    docs.append(Document(page_content=content or block, metadata=meta))
                else:
                    # Sin cabecera (output de tools como comparar/resumir)
                    docs.append(Document(
                        page_content=block,
                        metadata={"source": "tool_output", "doc_id": "tool_output"},
                    ))

        # ── Deduplicacion + cap ─────────────────────────────────────────
        # Despues de varias iteraciones ReAct, los docs acumulados pueden tener
        # duplicados (la misma seccion del PDF retornada por tools distintas) o
        # ser demasiados (>20). Deduplicar por chunk_id (o doc_id+page como
        # fallback) y truncar a MAX_FINAL_CHUNKS preservando el orden de aparicion
        # (los chunks de iteraciones tempranas suelen ser los mas relevantes).
        seen_keys: set[str] = set()
        deduped: List[Document] = []
        total_before_dedupe = len(docs)
        for d in docs:
            md = d.metadata or {}
            # Clave de identidad: chunk_id si existe, sino doc_id+page, sino content[:80]
            key = (
                md.get("chunk_id")
                or f"{md.get('doc_id', '')}|{md.get('page', '')}"
                or d.page_content[:80]
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(d)
            if len(deduped) >= MAX_FINAL_CHUNKS:
                break
        docs = deduped

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["react_agent"]
        traza["react_steps"] = react_steps
        traza["react_iterations"] = len(react_steps)
        traza["chunks_recuperados"] = _retrieved_chunk_payload(docs)
        traza["k_utilizado"] = len(docs)
        traza["chunks_dedupe"] = {
            "antes": total_before_dedupe,
            "despues": len(docs),
            "cap_aplicado": MAX_FINAL_CHUNKS,
        }

        return {
            "docs": docs,
            "react_steps": react_steps,
            "react_iteration": len(react_steps),
            "trazabilidad": traza,
        }

    # ── Nodo 3: generate_grounded ────────────────────────────────────────
    def generate_grounded(state: RAGState) -> dict[str, Any]:
        """Genera la respuesta final con grounding estricto.

        Combina el contexto recopilado por el agente ReAct y genera la respuesta
        usando reglas estrictas anti-hallucination.

        Si es un reintento (critic_feedback presente), adjunta las instrucciones
        de correccion del critico al prompt.
        """
        docs = state.get("docs", [])
        question = state["question"]

        # Construir contexto desde los documentos recopilados por ReAct.
        # IMPORTANTE: re-inyectar la cabecera [doc_id=...; página=...; chunk_id=...]
        # desde la metadata. El react_agent parsea estas cabeceras y las quita del
        # page_content (rag_graph.py:621), asi que aqui hay que reconstruirlas, o
        # el LLM no tiene de donde copiar el doc_id real y termina inventando uno
        # a partir del titulo del PDF.
        def _format_block(d: Document) -> str:
            md = d.metadata or {}
            content = _sanitize_for_llm(d.page_content)
            doc_id = md.get("doc_id") or md.get("source", "desconocido")
            # Output de tools sin cabecera estructurada (ej: comparar_modelos,
            # resumir_ficha) -> sin cabecera falsa
            if doc_id in ("tool_output", "react_agent"):
                return content
            page = md.get("page", "N/A")
            chunk_id = md.get("chunk_id")
            if chunk_id:
                header = f"[doc_id={doc_id}; página={page}; chunk_id={chunk_id}]"
            else:
                header = f"[doc_id={doc_id}; página={page}]"
            return f"{header}\n{content}"

        context = "\n\n---\n\n".join(
            _format_block(d) for d in docs if d.page_content.strip()
        )

        # Si no hay contexto, retornar mensaje de "no encontrado"
        if not context.strip():
            answer = "No encontrado en el contexto recuperado."
            traza = dict(state.get("trazabilidad") or {})
            traza["ruta"] = traza.get("ruta", []) + ["generate_grounded"]
            return {
                "answer": answer,
                "messages": [AIMessage(content=answer)],
                "trazabilidad": traza,
            }

        # Construir instruccion de correccion si es un reintento
        critic_feedback = state.get("critic_feedback")
        critic_instruction = ""
        if critic_feedback:
            critic_instruction = (
                "\n\n=== CORRECCIÓN REQUERIDA ===\n"
                "Tu respuesta anterior fue rechazada por el crítico. Corrige estos problemas:\n"
                + "\n".join(f"- {fb}" for fb in critic_feedback)
                + "\nGenera una nueva respuesta corregida."
            )

        user_content = GROUNDED_GENERATION_USER_TEMPLATE.format(
            question=question,
            context=context,
        )
        if critic_instruction:
            user_content += critic_instruction

        response = answer_llm.invoke(
            [
                SystemMessage(content=GROUNDED_GENERATION_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["generate_grounded"]
        traza["prompt_repr"] = {
            "modo": "grounded_rag",
            "question": question,
            "retrieved_count": len(docs),
            "retry_count": state.get("retry_count", 0),
            "has_critic_feedback": bool(critic_feedback),
        }
        return {
            "answer": answer,
            "messages": [response],
            "trazabilidad": traza,
        }

    # ── Nodo 4: evaluate_grounding (Reflecting) ─────────────────────────
    def evaluate_grounding(state: RAGState) -> dict[str, Any]:
        """Critico de grounding: evalua la calidad de la respuesta generada.

        Patron Reflecting: evalua y puede disparar reintentos.

        Tres caminos posibles:
        1. Aprobada (score >= 0.5 o approved=true) -> flujo termina
        2. Rechazada + puede reintentar (retry_count < 3)
           -> guarda feedback, incrementa retry_count, redirige a generate_grounded
        3. Rechazada + sin reintentos (retry_count >= 3)
           -> redirige a web_fallback para buscar en internet
        """
        docs = state.get("docs", [])
        answer = state.get("answer", "")
        question = state["question"]
        retry_count = state.get("retry_count", 0)

        retrieved_chunks_json = json.dumps(_retrieved_chunk_payload(docs), ensure_ascii=False)
        structured = critic_llm.with_structured_output(GroundingEvaluation)
        result: GroundingEvaluation = structured.invoke(
            [
                SystemMessage(content=GROUNDING_CRITIC_SYSTEM_PROMPT),
                HumanMessage(
                    content=GROUNDING_CRITIC_USER_TEMPLATE.format(
                        question=question,
                        retrieved_chunks=retrieved_chunks_json,
                        answer=answer,
                    )
                ),
            ]
        )
        eval_data = eval_to_dict(result)

        updates: dict[str, Any] = {}

        # Logica de aceptacion/rechazo pragmatica:
        # - Si el critico aprobo explicitamente -> aceptar
        # - Si supported_by_context=True (el contenido es correcto) Y score >= 0.3 ->
        #   aceptar aunque has_citations sea False. Muchas veces el LLM genera
        #   respuestas correctas pero sin el formato estricto de cita [doc_id=...],
        #   y no tiene sentido rechazar respuestas con contenido valido.
        # - Caso contrario: rechazar (permite reintento o web_fallback).
        approved = eval_data.get("approved", False)
        score = eval_data.get("score", 0.0)
        supported = eval_data.get("supported_by_context", False)

        if approved:
            rejected = False
        elif supported and score >= 0.3:
            # Override: respuesta con contenido correcto pero sin citas formales
            rejected = False
            eval_data["override_reason"] = "supported_by_context=True con score aceptable"
        else:
            rejected = score < 0.35  # Threshold mas permisivo (antes 0.5)

        can_retry = rejected and retry_count < MAX_RETRIES

        if can_retry:
            # Reflecting loop: guardar feedback del critico e incrementar contador
            updates["retry_count"] = retry_count + 1
            updates["critic_feedback"] = eval_data.get("issues", [])
        elif rejected:
            # Max reintentos agotados — web_fallback se activara via routing
            updates["retry_count"] = retry_count  # Mantener el conteo

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["evaluate_grounding"]
        traza["verificacion"] = {
            "aprobada": eval_data.get("approved"),
            "puntuacion": eval_data.get("score"),
            "soportada_en_contexto": eval_data.get("supported_by_context"),
            "tiene_citas": eval_data.get("has_citations"),
            "suficiente": eval_data.get("complete_enough"),
            "issues": eval_data.get("issues", []),
            "pregunta_aclaracion": eval_data.get("clarification_question"),
            "reintentos": retry_count,
            "will_retry": can_retry,
            "will_web_fallback": rejected and not can_retry,
        }
        updates["eval_result"] = eval_data
        updates["trazabilidad"] = traza
        return updates

    # ── Nodo 5: web_fallback ─────────────────────────────────────────────
    def web_fallback(state: RAGState) -> dict[str, Any]:
        """Fallback: busca en internet cuando la base interna falla tras 3 reintentos.

        Proceso:
        1. Llama a buscar_web con la pregunta del usuario
        2. Genera respuesta final usando los resultados web
        3. **Retroalimenta la base de conocimiento**: ingiere los resultados
           web como nuevos chunks en ChromaDB para que futuras consultas similares
           puedan responderse sin necesidad de fallback web.
        4. Marca web_search_used=True en trazabilidad
        """
        from datetime import datetime
        from rag_store import get_active_vector_store

        question = state["question"]

        # Buscar en internet
        web_results = buscar_web.invoke({"query": question})
        web_results_str = str(web_results)

        # Generar respuesta desde resultados web
        response = answer_llm.invoke(
            [
                SystemMessage(content=WEB_FALLBACK_SYSTEM_PROMPT),
                HumanMessage(content=WEB_FALLBACK_USER_TEMPLATE.format(
                    question=question,
                    web_results=web_results_str,
                )),
            ]
        )
        answer = response.content if isinstance(response.content, str) else str(response.content)

        # ── Retroalimentacion: ingerir resultados web a ChromaDB ────────
        feedback_status = "no_ingerido"
        feedback_chunks = 0
        try:
            if web_results_str and "Error" not in web_results_str[:20]:
                vs = get_active_vector_store()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                doc_id = f"web_fallback_{timestamp}"
                # Dividir resultados web por separador "---" (formato de buscar_web)
                web_blocks = [b.strip() for b in web_results_str.split("\n\n---\n\n") if b.strip()]
                docs_to_add: list[Document] = []
                for i, block in enumerate(web_blocks):
                    docs_to_add.append(Document(
                        page_content=f"Pregunta original: {question}\n\n{block}",
                        metadata={
                            "source": f"web_fallback_{timestamp}",
                            "doc_id": doc_id,
                            "chunk_id": f"{doc_id}_w{i}",
                            "page": 1,
                            "marca": "Web",
                            "modelo": "web_search_result",
                            "ocr": False,
                            "origen": "web_fallback",
                            "pregunta_origen": question,
                            "timestamp": timestamp,
                        },
                    ))
                if docs_to_add:
                    vs.add_documents(documents=docs_to_add)
                    feedback_chunks = len(docs_to_add)
                    feedback_status = "ingerido"
        except Exception as e:
            feedback_status = f"error: {e}"

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["web_fallback"]
        traza["web_search_used"] = True
        traza["web_results_snippet"] = web_results_str[:500]
        traza["kb_feedback"] = {
            "status": feedback_status,
            "chunks_ingeridos": feedback_chunks,
        }

        return {
            "answer": answer,
            "messages": [response],
            "web_search_used": True,
            "trazabilidad": traza,
        }

    # ── Nodo 6: evaluate_metrics ───────────────────────────────────────────
    def evaluate_metrics(state: RAGState) -> dict[str, Any]:
        """Calcula metricas de evaluacion: retrieval + LLM-as-Judge.

        Metricas de retrieval (Recall@k, Precision@k, MRR, nDCG):
        - Se calculan si la pregunta tiene ground truth en el dataset
        - Compara los doc_ids recuperados vs los relevantes esperados

        Metricas LLM-as-Judge (Relevance, Faithfulness):
        - Se calculan siempre que haya respuesta y contexto
        - Relevance: la respuesta es relevante para la pregunta?
        - Faithfulness: la respuesta es fiel al contexto (no alucina)?

        Todas las metricas se agregan a trazabilidad["metricas"].
        """
        question = state["question"]
        docs = state.get("docs", [])
        answer = state.get("answer", "")

        metricas: dict[str, Any] = {}

        # ── Metricas de retrieval ───────────────────────────────────────
        # Buscar ground truth para esta pregunta en el dataset
        retrieved_doc_ids = []
        for d in docs:
            md = d.metadata or {}
            doc_id = md.get("doc_id", "")
            if doc_id and doc_id != "react_agent":
                retrieved_doc_ids.append(doc_id)

        # Buscar pregunta en dataset. Matching en 2 niveles:
        # 1) Match exacto/substring (legacy)
        # 2) Match por overlap de palabras clave del modelo (cx-30, cx-5, hilux, etc.)
        #    para que preguntas parecidas ("potencia del CX-30" vs "Cual es la potencia
        #    del Mazda CX-30?") activen las metricas de retrieval.
        ground_truth = None
        q_lower = question.lower()

        # Nivel 1: substring match
        for entry in EVAL_DATASET:
            if not entry["relevant_doc_ids"]:
                continue
            eq = entry["question"].lower()
            if eq in q_lower or q_lower in eq:
                ground_truth = entry
                break

        # Nivel 2: overlap de tokens significativos si no hubo match exacto
        if ground_truth is None:
            import re as _re
            import unicodedata as _ud

            stopwords = {
                "que", "cual", "como", "donde", "cuando", "para", "por", "con",
                "una", "uno", "los", "las", "del", "dame", "son", "tiene", "esta",
                "este", "esto", "esa", "ese", "eso",
            }

            def _tokenize(text: str) -> set[str]:
                # Normalizar tildes: "transmisión" -> "transmision" para que matchee
                # con el dataset (que esta sin acentos).
                nfkd = _ud.normalize("NFKD", text.lower())
                ascii_text = "".join(c for c in nfkd if not _ud.combining(c))
                # Captura tokens alfanumericos permitiendo guiones internos para
                # identificadores de modelo (cx-5, cx-30, mx-5, mazda3, etc.).
                raw = _re.findall(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", ascii_text)
                tokens: set[str] = set()
                for t in raw:
                    if t in stopwords:
                        continue
                    # Aceptar tokens >= 3 chars, o cualquier token con digito o
                    # guion (probable identificador de modelo: "cx-5", "m3", etc.).
                    if len(t) >= 3 or any(c.isdigit() for c in t) or "-" in t:
                        tokens.add(t)
                return tokens

            q_tokens = _tokenize(q_lower)
            best_entry = None
            best_overlap = 0
            for entry in EVAL_DATASET:
                if not entry["relevant_doc_ids"]:
                    continue
                eq_tokens = _tokenize(entry["question"])
                overlap = len(q_tokens & eq_tokens)
                # Requerir al menos 3 tokens en comun para considerar match
                if overlap >= 3 and overlap > best_overlap:
                    best_overlap = overlap
                    best_entry = entry
            ground_truth = best_entry

        if ground_truth and ground_truth["relevant_doc_ids"] and retrieved_doc_ids:
            k = len(retrieved_doc_ids)
            metricas["retrieval"] = compute_retrieval_metrics(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=ground_truth["relevant_doc_ids"],
                k=k,
            )
        else:
            metricas["retrieval"] = None

        # ── Metricas LLM-as-Judge (solo en modo evaluacion) ─────────────
        # Las llamadas LLM-as-Judge agregan ~5s de latencia.
        # Solo se ejecutan cuando eval_mode=True (script batch o endpoint /evaluate).
        eval_mode = state.get("eval_mode", False)
        if eval_mode and answer.strip() and docs:
            context = "\n\n---\n\n".join(
                _sanitize_for_llm(d.page_content[:500]) for d in docs if d.page_content.strip()
            )
            metricas["llm_judge"] = compute_llm_judge_metrics(
                question=question,
                context=context,
                answer=answer,
            )
        else:
            metricas["llm_judge"] = None

        # Agregar score del critico existente
        eval_result = state.get("eval_result") or {}
        metricas["grounding_score"] = eval_result.get("score", None)

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["evaluate_metrics"]
        traza["metricas"] = metricas
        return {"trazabilidad": traza}

    # ── Funciones de routing (conditional edges) ─────────────────────────

    def route_after_metrics(state: RAGState) -> str:
        """Decide si reintentar, usar web fallback, o terminar despues de metricas.

        Aplica la misma logica de aceptacion/rechazo pragmatica que evaluate_grounding:
        - Si el critico aprobo -> END
        - Si supported_by_context=True Y score >= 0.3 -> END (override)
        - Si score < 0.35 y hay reintentos -> generate_grounded
        - Si score < 0.35 sin reintentos -> web_fallback
        """
        # Si ya pasamos por web_fallback, terminar definitivamente
        if state.get("web_search_used"):
            return END

        eval_data = state.get("eval_result") or {}
        retry_count = state.get("retry_count", 0)
        approved = eval_data.get("approved", False)
        score = eval_data.get("score", 0.0)
        supported = eval_data.get("supported_by_context", False)

        # Override: si el contenido es correcto aunque falten citas formales
        if approved or (supported and score >= 0.3):
            return END

        # Rechazado: reintentar si hay reintentos disponibles
        if score < 0.35 and retry_count < MAX_RETRIES and state.get("critic_feedback"):
            return "generate_grounded"
        return "web_fallback"

    def route_after_classify(state: RAGState) -> str:
        """Decide ruta: query_transformer (RAG) o answer_general directo."""
        intent = state.get("intent") or {}
        if intent.get("needs_retrieval", True):
            return "query_transformer"
        return "answer_general"

    # ── Ensamblaje del grafo ─────────────────────────────────────────────

    graph = (
        StateGraph(RAGState)
        # Nodos
        .add_node(
            "classify_intent",
            RunnableLambda(classify_intent).with_config({
                "run_name": "Intent Classifier",
                "tags": ["rag", "intent", "classification"],
                "metadata": {"node": "classify_intent"}
            })
        )
        
        .add_node(
            "answer_general",
            RunnableLambda(answer_general).with_config({
                "run_name": "General Answer",
                "tags": ["rag", "no-retrieval", "llm"],
                "metadata": {"node": "answer_general"}
            })
        )

        .add_node(
            "query_transformer",
            RunnableLambda(query_transformer).with_config({
                "run_name": "Query Transformer (HyDE + Decomposition)",
                "tags": ["rag", "query", "transformation"],
                "metadata": {"node": "query_transformer"}
            })
        )

        .add_node(
            "react_agent",
            RunnableLambda(react_agent).with_config({
                "run_name": "ReAct Agent",
                "tags": ["rag", "react", "agent"],
                "metadata": {"node": "react_agent"}
            })
        )

        .add_node(
            "generate_grounded",
            RunnableLambda(generate_grounded).with_config({
                "run_name": "Grounded Generator",
                "tags": ["rag", "generation", "llm"],
                "metadata": {"node": "generate_grounded"}
            })
        )

        .add_node(
            "evaluate_grounding",
            RunnableLambda(evaluate_grounding).with_config({
                "run_name": "Grounding Critic (Reflecting)",
                "tags": ["rag", "evaluation", "reflecting"],
                "metadata": {"node": "evaluate_grounding"}
            })
        )

        .add_node(
            "evaluate_metrics",
            RunnableLambda(evaluate_metrics).with_config({
                "run_name": "Metrics Evaluator",
                "tags": ["rag", "metrics", "evaluation"],
                "metadata": {"node": "evaluate_metrics"}
            })
        )

        .add_node(
            "web_fallback",
            RunnableLambda(web_fallback).with_config({
                "run_name": "Web Fallback",
                "tags": ["rag", "web", "fallback"],
                "metadata": {"node": "web_fallback"}
            })
        )

        # Edges
        .add_edge(START, "classify_intent")
        .add_conditional_edges(
            "classify_intent",
            route_after_classify,
            {"query_transformer": "query_transformer", "answer_general": "answer_general"},
        )
        .add_edge("answer_general", END)
        .add_edge("query_transformer", "react_agent")
        .add_edge("react_agent", "generate_grounded")
        .add_edge("generate_grounded", "evaluate_grounding")
        .add_edge("evaluate_grounding", "evaluate_metrics")
        .add_conditional_edges(
            "evaluate_metrics",
            route_after_metrics,
            {"generate_grounded": "generate_grounded", "web_fallback": "web_fallback", END: END},
        )
        .add_edge("web_fallback", "evaluate_metrics")
        .compile(checkpointer=MemorySaver())
    )
    print(graph.get_graph().draw_ascii())
    return graph
