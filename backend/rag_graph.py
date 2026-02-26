# backend/rag_graph.py
from __future__ import annotations

from typing import Annotated, List, Literal, Optional, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel

from rag_store import get_vector_store
from tools import (
    buscar_especificacion,
    buscar_por_marca,
    comparar_modelos,
    listar_modelos_disponibles,
    resumir_ficha,
)

TOOLS = [
    listar_modelos_disponibles,
    buscar_especificacion,
    buscar_por_marca,
    comparar_modelos,
    resumir_ficha,
]

# k de recuperación según intención de la consulta
K_POR_INTENCION = {
    "comparación": 10,
    "resumen":      8,
    "búsqueda":     4,
    "general":      0,
}


# ---------------------------------------------------------------------------
# Modelo de clasificación (Fase 3)
# ---------------------------------------------------------------------------

class ClasificacionConsulta(BaseModel):
    """Structured output del nodo clasificador."""
    intencion: Literal["búsqueda", "resumen", "comparación", "general"]
    marcas_mencionadas: List[str]
    modelos_mencionados: List[str]
    requiere_rag: bool


class ResultadoVerificacion(BaseModel):
    """Structured output del nodo verificador."""
    aprobada:        bool
    puntuacion:      float   # 0.0 – 1.0
    motivo_rechazo:  str     # vacío si aprobada
    reintentos:      int


# ---------------------------------------------------------------------------
# Estado del grafo
# ---------------------------------------------------------------------------

class RAGState(TypedDict):
    question:       str
    clasificacion:  Optional[ClasificacionConsulta]
    docs:           List[Document]
    answer:         str
    messages:       Annotated[List[BaseMessage], add_messages]
    verificacion:   Optional[ResultadoVerificacion]
    reintentos:     int
    trazabilidad:   dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_grounding_context(docs: List[Document]) -> tuple[str, str]:
    """Construye el contexto de recuperación con IDs de fuente para citación."""
    chunks: list[str] = []
    catalog: list[str] = []

    for idx, d in enumerate(docs, start=1):
        sid    = f"S{idx}"
        source = d.metadata.get("source", "desconocido")
        page   = d.metadata.get("page", "")
        marca  = d.metadata.get("marca", "")
        modelo = d.metadata.get("modelo", "")

        chunks.append(f"[{sid}] source={source} page={page} marca={marca} modelo={modelo}\n{d.page_content}")
        catalog.append(f"- [{sid}] {source} (p.{page})")

    return "\n\n".join(chunks), "\n".join(catalog)


# ---------------------------------------------------------------------------
# Construcción del grafo
# ---------------------------------------------------------------------------

def build_rag_graph():
    vs  = get_vector_store()
    llm = ChatOpenAI(model="gpt-5-nano", temperature=0.2)

    # LLM para clasificación con structured output
    llm_clasificador = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
        ClasificacionConsulta
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    # -----------------------------------------------------------------------
    # Nodo 1 — Clasificador (Fase 3)
    # -----------------------------------------------------------------------
    def clasificar(state: RAGState) -> dict:
        """Clasifica la intención y extrae entidades de la consulta."""
        prompt = (
            "Eres un clasificador de consultas sobre fichas técnicas de vehículos.\n"
            "Analiza la siguiente consulta y responde con:\n"
            "- intencion: 'búsqueda' (dato puntual), 'resumen' (overview de un modelo),\n"
            "  'comparación' (entre dos o más modelos) o 'general' (pregunta conceptual sin modelo específico)\n"
            "- marcas_mencionadas: lista de marcas detectadas (ej: ['Toyota', 'Mazda'])\n"
            "- modelos_mencionados: lista de modelos detectados (ej: ['Hilux', 'CX-5'])\n"
            "- requiere_rag: True si necesita consultar documentos; False si es pregunta general\n\n"
            f"Consulta: {state['question']}"
        )
        clasificacion: ClasificacionConsulta = llm_clasificador.invoke(prompt)
        traza = state.get("trazabilidad") or {}
        traza["ruta"] = ["clasificar"]
        traza["clasificacion"] = {
            "intencion":           clasificacion.intencion,
            "marcas_mencionadas":  clasificacion.marcas_mencionadas,
            "modelos_mencionados": clasificacion.modelos_mencionados,
            "requiere_rag":        clasificacion.requiere_rag,
        }
        return {"clasificacion": clasificacion, "trazabilidad": traza}

    # -----------------------------------------------------------------------
    # Nodo 2 — Recuperación semántica con k dinámico (Fase 4)
    # -----------------------------------------------------------------------
    def retrieve(state: RAGState) -> dict:
        """Recupera chunks con k ajustado según la intención clasificada."""
        clasificacion = state.get("clasificacion")
        intencion = clasificacion.intencion if clasificacion else "búsqueda"
        k = K_POR_INTENCION.get(intencion, 4)

        retriever = vs.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(state["question"])
        traza = state.get("trazabilidad") or {}
        traza["ruta"] = traza.get("ruta", []) + ["retrieve"]
        traza["k_utilizado"] = k
        traza["chunks_recuperados"] = [
            {
                "source": d.metadata.get("source", ""),
                "page":   d.metadata.get("page", ""),
                "marca":  d.metadata.get("marca", ""),
                "modelo": d.metadata.get("modelo", ""),
            }
            for d in docs
        ]
        return {"docs": docs, "trazabilidad": traza}

    # -----------------------------------------------------------------------
    # Nodo 3 — Generación con RAG (Fase 5)
    # -----------------------------------------------------------------------
    def generate(state: RAGState) -> dict:
        """Genera respuesta usando el contexto recuperado y las tools disponibles."""
        context, source_catalog = _build_grounding_context(state["docs"])

        clasificacion = state.get("clasificacion")
        intencion_hint = f"Intención detectada: {clasificacion.intencion}." if clasificacion else ""

        system_prompt = (
            "Eres un asistente experto en fichas técnicas de vehículos. "
            "Tienes acceso a herramientas para consultar el catálogo. "
            f"{intencion_hint}\n"
            "Responde siempre en español. Prioriza legibilidad: usa encabezados y bullets.\n"
            "No repitas disculpas ni texto redundante.\n"
            "Si faltan datos, indícalo brevemente.\n"
            "Evita tablas grandes llenas de N/D.\n\n"
            "Reglas de grounding:\n"
            "- Toda afirmación factual debe respaldarse en el contexto o en la salida de tools.\n"
            "- Cita con formato [S1], [S2], etc., inmediatamente después de cada afirmación.\n"
            "- No inventes valores. Si no hay evidencia, dilo explícitamente.\n\n"
            f"Fuentes disponibles:\n{source_catalog}\n\n"
            f"Contexto recuperado:\n{context}"
        )

        history = state.get("messages") or []
        if not history:
            history = [HumanMessage(content=state["question"])]

        response = llm_with_tools.invoke([SystemMessage(content=system_prompt), *history])

        answer_content = response.content if isinstance(response.content, str) else ""
        if answer_content and state["docs"]:
            answer_content += f"\n\n---\nFuentes recuperadas:\n{source_catalog}"
            response.content = answer_content

        traza = state.get("trazabilidad") or {}
        traza["ruta"] = traza.get("ruta", []) + ["generate"]
        traza["prompt_generador"] = system_prompt
        return {"messages": [response], "answer": answer_content, "trazabilidad": traza}

    # -----------------------------------------------------------------------
    # Nodo 4 — Verificador / crítica (Fase 6)
    # -----------------------------------------------------------------------
    llm_verificador = ChatOpenAI(model="gpt-5-nano", temperature=0).with_structured_output(
        ResultadoVerificacion
    )

    def verificar(state: RAGState) -> dict:
        """Evalúa si la respuesta está respaldada por el contexto recuperado."""
        _, source_catalog = _build_grounding_context(state["docs"])
        reintentos = state.get("reintentos", 0)

        prompt = (
            "Eres un verificador de respuestas en un sistema RAG sobre fichas técnicas de vehículos.\n"
            "Evalúa si la siguiente respuesta está correctamente respaldada por las fuentes indicadas.\n\n"
            f"Consulta original: {state['question']}\n\n"
            f"Fuentes disponibles:\n{source_catalog}\n\n"
            f"Respuesta generada:\n{state['answer']}\n\n"
            "Criterios de evaluación:\n"
            "1. ¿La respuesta usa solo información de las fuentes disponibles?\n"
            "2. ¿Responde directamente la consulta del usuario?\n"
            "3. ¿Evita afirmaciones inventadas o sin soporte?\n\n"
            f"Reintentos previos: {reintentos}\n"
            "Responde con: aprobada, puntuacion (0.0-1.0), motivo_rechazo (vacío si aprobada), "
            f"reintentos={reintentos}."
        )

        verificacion: ResultadoVerificacion = llm_verificador.invoke(prompt)
        traza = state.get("trazabilidad") or {}
        traza["ruta"] = traza.get("ruta", []) + ["verificar"]
        traza["verificacion"] = {
            "aprobada":       verificacion.aprobada,
            "puntuacion":     verificacion.puntuacion,
            "motivo_rechazo": verificacion.motivo_rechazo,
            "reintentos":     reintentos,
        }
        return {
            "verificacion": verificacion,
            "reintentos": reintentos + (0 if verificacion.aprobada else 1),
            "trazabilidad": traza,
        }

    # -----------------------------------------------------------------------
    # Nodo 5 — Respuesta directa para consultas generales (sin RAG)
    # -----------------------------------------------------------------------
    def generate_direct(state: RAGState) -> dict:
        """Responde preguntas generales conceptuales sin necesidad de RAG."""
        prompt = (
            "Eres un asistente experto en vehículos y tecnología automotriz. "
            "Responde en español de forma clara y concisa.\n\n"
            f"Consulta: {state['question']}"
        )
        response = llm.invoke(prompt)
        answer = response.content if isinstance(response.content, str) else str(response.content)
        traza = state.get("trazabilidad") or {}
        traza["ruta"] = traza.get("ruta", []) + ["generate_direct"]
        return {"answer": answer, "trazabilidad": traza}

    # -----------------------------------------------------------------------
    # Funciones de routing
    # -----------------------------------------------------------------------
    def route_after_classify(state: RAGState) -> str:
        """Decide si ir al RAG o responder directo según la clasificación."""
        clasificacion = state.get("clasificacion")
        if clasificacion and not clasificacion.requiere_rag:
            return "generate_direct"
        return "retrieve"

    def should_use_tool(state: RAGState) -> str:
        """Decide si el LLM quiere llamar una tool o pasa al verificador."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "verificar"

    def route_after_verify(state: RAGState) -> str:
        """Loop de reintentos: si falla y quedan intentos, vuelve al generador."""
        v = state.get("verificacion")
        reintentos = state.get("reintentos", 0)
        if v and not v.aprobada and reintentos < 2:
            return "generate"
        return END

    # -----------------------------------------------------------------------
    # Ensamblaje del grafo
    # -----------------------------------------------------------------------
    tool_node = ToolNode(TOOLS)
    memory    = MemorySaver()

    graph = (
        StateGraph(RAGState)
        .add_node("clasificar",       clasificar)
        .add_node("retrieve",         retrieve)
        .add_node("generate",         generate)
        .add_node("generate_direct",  generate_direct)
        .add_node("tools",            tool_node)
        .add_node("verificar",        verificar)
        .add_edge(START, "clasificar")
        .add_conditional_edges(
            "clasificar",
            route_after_classify,
            {"retrieve": "retrieve", "generate_direct": "generate_direct"},
        )
        .add_edge("retrieve",        "generate")
        .add_conditional_edges(
            "generate",
            should_use_tool,
            {"tools": "tools", "verificar": "verificar"},
        )
        .add_edge("tools",           "generate")
        .add_conditional_edges(
            "verificar",
            route_after_verify,
            {"generate": "generate", END: END},
        )
        .add_edge("generate_direct", END)
        .compile(checkpointer=memory)
    )
    print(graph.get_graph().draw_ascii())
    return graph
