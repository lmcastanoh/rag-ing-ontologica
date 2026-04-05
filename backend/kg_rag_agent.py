# backend/kg_rag_agent.py
# ==============================================================================
# Agente KG-RAG: combina Knowledge Graph (GraphDB + SPARQL) con Vector Store
# (ChromaDB) y LLM para responder preguntas sobre fichas técnicas vehiculares.
#
# Arquitectura:
#   Usuario → LangGraph Agent → decide herramienta →
#       ├── sparql_kg_*  → GraphDB (datos estructurados exactos)
#       └── vector_search → ChromaDB (contexto semántico de PDFs)
#       → LLM sintetiza respuesta final con citas
#
# Integración con el RAG existente:
#   Importa get_vectorstore() de rag_store.py para no duplicar el vector store.
#   Se puede invocar como endpoint /kg_chat o integrarlo en rag_graph.py como
#   nodo adicional enriquecido con datos del KG.
# ==============================================================================

from __future__ import annotations

import os
import logging
from typing import Annotated, TypedDict, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from kg_retriever import (
    kg_buscar_especificaciones,
    kg_buscar_motor,
    kg_comparar_modelos,
    kg_listar_modelos_por_marca,
    kg_electricos_por_autonomia,
    kg_sistemas_seguridad,
    kg_format_para_llm,
)
from rag_store import get_vector_store as get_vectorstore

load_dotenv()
logger = logging.getLogger(__name__)

LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


# ==============================================================================
# TOOLS DEL AGENTE
# Cada tool envuelve una función de kg_retriever o el vector store.
# El LLM decide cuál usar según la pregunta del usuario.
# ==============================================================================

@tool
def sparql_especificaciones(modelo: str) -> str:
    """
    Consulta el Knowledge Graph para obtener especificaciones técnicas exactas
    de un modelo de vehículo: peso, longitud, capacidad de baúl, precio, año.
    Usar cuando se pregunten datos concretos de un modelo específico.

    Args:
        modelo: Nombre del modelo (ej: 'Golf', 'Corolla Hybrid', 'ZS EV').
    """
    resultados = kg_buscar_especificaciones(modelo)
    return kg_format_para_llm(resultados, f"Especificaciones de {modelo}")


@tool
def sparql_motor(modelo: str) -> str:
    """
    Consulta el Knowledge Graph para obtener datos del motor de un vehículo:
    tipo (eléctrico/combustión), potencia en CV, cilindrada, combustible,
    autonomía eléctrica o consumo.
    Usar cuando se pregunten características del motor o propulsión.

    Args:
        modelo: Nombre del modelo.
    """
    resultados = kg_buscar_motor(modelo)
    return kg_format_para_llm(resultados, f"Motor de {modelo}")


@tool
def sparql_comparar(modelo1: str, modelo2: str) -> str:
    """
    Consulta el Knowledge Graph para comparar dos modelos de vehículos lado a lado.
    Recupera peso, longitud, baúl, precio, consumo y autonomía de ambos.
    Usar cuando el usuario quiera comparar dos modelos concretos.

    Args:
        modelo1: Primer modelo a comparar.
        modelo2: Segundo modelo a comparar.
    """
    resultados = kg_comparar_modelos(modelo1, modelo2)
    return kg_format_para_llm(resultados, f"Comparación {modelo1} vs {modelo2}")


@tool
def sparql_marca(marca: str) -> str:
    """
    Lista todos los modelos disponibles de una marca, con su categoría
    (Sedán, SUV, Hatchback…) y tipo de propulsión.
    Usar cuando se pregunte qué modelos tiene una marca.

    Args:
        marca: Nombre de la marca (ej: 'Toyota', 'Volkswagen', 'MG Emotor').
    """
    resultados = kg_listar_modelos_por_marca(marca)
    return kg_format_para_llm(resultados, f"Modelos de {marca}")


@tool
def sparql_electricos(autonomia_minima: float = 300.0) -> str:
    """
    Lista vehículos eléctricos con autonomía WLTP mayor al umbral indicado,
    incluyendo capacidad de batería y precio.
    Usar para preguntas sobre eléctricos con buena autonomía.

    Args:
        autonomia_minima: Autonomía mínima en km. Default 300 km.
    """
    resultados = kg_electricos_por_autonomia(autonomia_minima)
    return kg_format_para_llm(resultados, f"Eléctricos con autonomía ≥ {autonomia_minima} km")


@tool
def sparql_seguridad(modelo: str) -> str:
    """
    Consulta los sistemas de seguridad activa incluidos en un modelo de vehículo.
    Usar cuando se pregunten sistemas ADAS, asistencias al conductor o seguridad.

    Args:
        modelo: Nombre del modelo.
    """
    resultados = kg_sistemas_seguridad(modelo)
    return kg_format_para_llm(resultados, f"Sistemas de seguridad de {modelo}")


@tool
def vector_buscar(pregunta: str, marca: str = "", modelo: str = "") -> str:
    """
    Busca en el vector store (ChromaDB) con los PDFs de fichas técnicas para
    encontrar información detallada, descripciones narrativas, equipamiento
    y cualquier dato no estructurado sobre vehículos.
    Usar para preguntas abiertas, equipamiento de serie o información contextual.

    Args:
        pregunta: Pregunta o descripción de lo que se busca.
        marca: Filtrar por marca (opcional).
        modelo: Filtrar por modelo (opcional).
    """
    vs = get_vectorstore()
    filtros: dict[str, str] = {}
    if marca:
        filtros["marca"] = marca
    if modelo:
        filtros["modelo"] = modelo

    docs = vs.similarity_search(
        pregunta,
        k=6,
        filter=filtros if filtros else None,
    )
    if not docs:
        return "[Vector Store] No se encontraron documentos relevantes."

    fragmentos = []
    for doc in docs:
        meta = doc.metadata
        ref = f"[{meta.get('doc_id', '?')}; página {meta.get('page', '?')}]"
        fragmentos.append(f"{ref}\n{doc.page_content[:500]}")

    return "[Vector Store]\n" + "\n\n".join(fragmentos)


# ==============================================================================
# ESTADO DEL AGENTE
# ==============================================================================

class KGAgentState(TypedDict):
    messages:  Annotated[list[BaseMessage], add_messages]
    respuesta: str


# ==============================================================================
# NODOS DEL GRAFO
# ==============================================================================

TOOLS = [
    sparql_especificaciones,
    sparql_motor,
    sparql_comparar,
    sparql_marca,
    sparql_electricos,
    sparql_seguridad,
    vector_buscar,
]

SYSTEM_PROMPT = """Eres un asistente experto en fichas técnicas de vehículos automotores.
Tienes acceso a dos fuentes de conocimiento complementarias:

1. **Knowledge Graph (GraphDB + SPARQL)**: datos estructurados exactos como peso, potencia,
   autonomía, precio, transmisión, sistemas de seguridad. Úsalo para datos cuantitativos precisos.

2. **Vector Store (ChromaDB)**: fragmentos de PDFs con descripciones narrativas, equipamiento
   detallado, notas técnicas. Úsalo para información contextual o no estructurada.

Estrategia:
- Para datos exactos (peso, consumo, potencia, precio): usa tools sparql_*
- Para descripciones, equipamiento, tecnología: usa vector_buscar
- Para comparaciones: combina sparql_comparar + vector_buscar
- Si una tool no retorna datos, intenta con la otra fuente

Siempre incluye las fuentes en tu respuesta usando el formato:
- Para KG: [KG - tipo de consulta]
- Para vector: [doc_id=nombre_archivo; página=N]

Responde en español, de forma clara y estructurada."""


def nodo_agente(state: KGAgentState) -> dict:
    """Nodo principal: el LLM decide qué tools invocar."""
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    llm_con_tools = llm.bind_tools(TOOLS)

    mensajes = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    respuesta = llm_con_tools.invoke(mensajes)
    return {"messages": [respuesta]}


def ruta_continuar(state: KGAgentState) -> str:
    """Decide si continuar ejecutando tools o terminar."""
    ultimo_mensaje = state["messages"][-1]
    if hasattr(ultimo_mensaje, "tool_calls") and ultimo_mensaje.tool_calls:
        return "tools"
    return "fin"


def nodo_fin(state: KGAgentState) -> dict:
    """Extrae la respuesta final del último mensaje."""
    ultimo = state["messages"][-1]
    return {"respuesta": getattr(ultimo, "content", str(ultimo))}


# ==============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ==============================================================================

def construir_kg_agent() -> Any:
    """Construye y compila el grafo LangGraph del agente KG-RAG."""
    tool_node = ToolNode(TOOLS)

    grafo = StateGraph(KGAgentState)
    grafo.add_node("agente", nodo_agente)
    grafo.add_node("tools",  tool_node)
    grafo.add_node("fin",    nodo_fin)

    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges(
        "agente",
        ruta_continuar,
        {"tools": "tools", "fin": "fin"},
    )
    grafo.add_edge("tools", "agente")
    grafo.add_edge("fin", END)

    return grafo.compile()


# ==============================================================================
# INTERFAZ PÚBLICA
# ==============================================================================

_kg_agent = None


def get_kg_agent():
    """Singleton del agente KG-RAG."""
    global _kg_agent
    if _kg_agent is None:
        _kg_agent = construir_kg_agent()
    return _kg_agent


def responder_con_kg(pregunta: str) -> str:
    """
    Responde una pregunta usando el agente KG-RAG.

    Args:
        pregunta: Pregunta del usuario en lenguaje natural.

    Returns:
        Respuesta generada combinando KG y vector store.
    """
    agente = get_kg_agent()
    estado_inicial = {"messages": [HumanMessage(content=pregunta)]}
    resultado = agente.invoke(estado_inicial)
    return resultado.get("respuesta", "No se pudo generar una respuesta.")


# ==============================================================================
# INTEGRACIÓN CON EL ENDPOINT FASTAPI EXISTENTE
# ==============================================================================
# Para agregar un endpoint /kg_chat en backend/app.py, añade:
#
#   from kg_rag_agent import responder_con_kg
#
#   @app.post("/kg_chat")
#   async def kg_chat(req: ChatRequest):
#       respuesta = responder_con_kg(req.question)
#       return {"answer": respuesta}
#
# O para enriquecer el RAG existente, agrega un nodo "enriquecer_con_kg" en
# rag_graph.py que llame a kg_buscar_especificaciones() cuando el intent sea
# Búsqueda o Comparación, y añade los datos KG al contexto del generador.
# ==============================================================================

if __name__ == "__main__":
    print("Probando KG-RAG Agent...")
    print("(Requiere GraphDB en localhost:7200 con repo 'vehiculos')\n")

    preguntas = [
        "¿Cuánto pesa el Toyota Corolla y cuál es su consumo?",
        "Compara el VW Golf con el Peugeot 208 en precio y peso",
        "¿Qué vehículos eléctricos tienen más de 350 km de autonomía?",
        "¿Qué sistemas de seguridad tiene el MGZS EV?",
    ]

    for p in preguntas:
        print(f"Pregunta: {p}")
        print(f"Respuesta: {responder_con_kg(p)}")
        print("-" * 60)
