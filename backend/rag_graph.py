# backend/rag_graph.py
# ==============================================================================
# Grafo LangGraph principal del sistema RAG para fichas tecnicas vehiculares.
#
# Arquitectura: ReAct Agent + Reflecting
#
# Flujo: classify_intent -> [react_agent] -> generate_grounded ->
#        evaluate_grounding -> [retry | web_fallback | END]
#
# 3 rutas posibles:
#   A) GENERAL:     classify -> answer_general -> END (sin retrieval)
#   B) RAG + ReAct: classify -> react_agent -> generate -> evaluate -> END
#   C) Web Fallback: ... -> evaluate (3 fallos) -> web_fallback -> END
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
    REACT_AGENT_SYSTEM_PROMPT,
    REACT_AGENT_USER_TEMPLATE,
    WEB_FALLBACK_SYSTEM_PROMPT,
    WEB_FALLBACK_USER_TEMPLATE,
)
from schemas import GroundingEvaluation, IntentClassification, eval_to_dict, intent_to_dict
from tools import (
    buscar_especificacion,
    buscar_hyde,
    buscar_por_marca,
    buscar_vectorial,
    buscar_web,
    comparar_modelos,
    descomponer_pregunta,
    listar_modelos_disponibles,
    resumir_ficha,
    _retrieval_context,
    _fix_doubled_text,
)


# ── Configuracion ──────────────────────────────────────────────────────────
MAX_REACT_ITERATIONS = 7   # Maximo de pasos Thought/Action/Observation
MAX_RETRIES = 3            # Maximo de reintentos del reflecting loop


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
    }

    # LLMs especializados (todos gpt-5-nano con diferentes temperatures)
    router_llm = ChatOpenAI(model="gpt-5-nano", temperature=0)      # Clasificador
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

            observation_str = str(observation)

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
                f"Observation: {observation_str[:800]}\n"
            )

        # Construir Documents desde las observaciones para generate_grounded
        docs: List[Document] = []
        for obs in all_observations:
            if obs.strip():
                docs.append(Document(
                    page_content=obs,
                    metadata={"source": "react_agent", "doc_id": "react_agent"},
                ))

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["react_agent"]
        traza["react_steps"] = react_steps
        traza["react_iterations"] = len(react_steps)
        traza["chunks_recuperados"] = _retrieved_chunk_payload(docs)
        traza["k_utilizado"] = len(docs)

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

        # Construir contexto desde los documentos recopilados por ReAct
        context = "\n\n---\n\n".join(
            d.page_content for d in docs if d.page_content.strip()
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
        rejected = not eval_data.get("approved", True) and eval_data.get("score", 1.0) < 0.5
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
        3. Marca web_search_used=True en trazabilidad
        """
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

        traza = dict(state.get("trazabilidad") or {})
        traza["ruta"] = traza.get("ruta", []) + ["web_fallback"]
        traza["web_search_used"] = True
        traza["web_results_snippet"] = web_results_str[:500]

        return {
            "answer": answer,
            "messages": [response],
            "web_search_used": True,
            "trazabilidad": traza,
        }

    # ── Funciones de routing (conditional edges) ─────────────────────────

    def route_after_classify(state: RAGState) -> str:
        """Decide ruta: retrieval via ReAct o respuesta general directa."""
        intent = state.get("intent") or {}
        if intent.get("needs_retrieval", True):
            return "react_agent"
        return "answer_general"

    def route_after_reflect(state: RAGState) -> str:
        """Decide si reintentar, usar web fallback, o terminar.

        - Aprobada o score >= 0.5: END
        - Rechazada y retries < 3: generate_grounded (reintento con feedback)
        - Rechazada y retries >= 3: web_fallback (buscar en internet)
        """
        eval_data = state.get("eval_result") or {}
        retry_count = state.get("retry_count", 0)
        rejected = not eval_data.get("approved", True) and eval_data.get("score", 1.0) < 0.5

        if not rejected:
            return END
        if retry_count <= MAX_RETRIES and state.get("critic_feedback"):
            return "generate_grounded"
        return "web_fallback"

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

        .add_node("answer_general", answer_general)

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
            {"react_agent": "react_agent", "answer_general": "answer_general"},
        )
        .add_edge("answer_general", END)
        .add_edge("react_agent", "generate_grounded")
        .add_edge("generate_grounded", "evaluate_grounding")
        .add_conditional_edges(
            "evaluate_grounding",
            route_after_reflect,
            {"generate_grounded": "generate_grounded", "web_fallback": "web_fallback", END: END},
        )
        .add_edge("web_fallback", END)
        .compile(checkpointer=MemorySaver())
    )
    print(graph.get_graph().draw_ascii())
    return graph
