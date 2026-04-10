# backend/tools.py
# ==============================================================================
# Tools de LangGraph para el sistema RAG de fichas tecnicas vehiculares.
#
# Tools disponibles:
#   - listar_modelos_disponibles: catalogo de modelos indexados
#   - buscar_especificacion:      dato tecnico puntual de un modelo
#   - buscar_por_marca:           todos los modelos de una marca
#   - comparar_modelos:           tabla comparativa entre 2 modelos
#   - resumir_ficha:              resumen estructurado de un modelo
#   - buscar_vectorial:           busqueda semantica en ChromaDB con filtros
#   - buscar_hyde:                HyDE retrieval (documento hipotetico)
#   - descomponer_pregunta:       descomposicion de preguntas complejas
#   - buscar_web:                 busqueda en internet (fallback)
# ==============================================================================
from __future__ import annotations

import re
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from rag_store import get_active_vector_store


def _get_llm():
    """Retorna instancia de LLM para generacion dentro de tools (comparar, resumir).

    Usa gpt-5-nano con temperature=0 para respuestas consistentes y deterministas.
    """
    return ChatOpenAI(model="gpt-5-nano", temperature=0)


# ── Helpers compartidos (usados por tools y por rag_graph) ──────────────────

def _model_variants(model: str, make: str | None = None) -> list[str]:
    """Genera variantes normalizadas de un nombre de modelo para matching flexible.

    Cubre variaciones comunes: con/sin guion, con/sin marca prefijada,
    Title case vs original.
    """
    no_hyphen = model.replace("-", " ")
    bases = {model, model.title(), no_hyphen, no_hyphen.title()}
    variants: set[str] = set(bases)
    if make:
        for b in bases:
            variants.add(f"{make} {b}")
    return list(variants)


def _build_retrieval_filter(entities: dict[str, Any] | None) -> dict | None:
    """Construye filtro de metadata para ChromaDB a partir de entidades."""
    if not entities:
        return None
    model = entities.get("model")
    make = entities.get("make")
    if model and len(model) >= 2:
        return {"modelo": {"$in": _model_variants(model, make)}}
    if make and len(make) >= 2:
        return {"marca": make}
    return None


def _fix_doubled_text(text: str) -> str:
    """Corrige texto con caracteres duplicados por extraccion corrupta de PDF.

    Aplica la correccion LINEA POR LINEA en lugar de al texto completo, porque
    en muchos PDFs solo los titulos/encabezados tienen letras duplicadas
    (ej: "EESSPPEECCIIFFIICCAACCIIOONNEESS") mientras que el resto del chunk
    (incluyendo datos numericos como "20,39 kg-m") esta en formato normal.

    Aplicar text[::2] al chunk entero destruia los datos reales. Ahora solo se
    aplica a lineas individuales que cumplan el criterio de duplicacion.
    """
    if len(text) < 10:
        return text

    def _line_is_doubled(line: str) -> bool:
        if len(line) < 10:
            return False
        sample = line[:60]
        doubles = len(re.findall(r"([A-Za-z0-9])\1", sample))
        alphanums = len(re.findall(r"[A-Za-z0-9]", sample))
        return alphanums > 4 and (doubles * 2) / alphanums > 0.7

    lines = text.split("\n")
    fixed_lines = []
    for line in lines:
        if _line_is_doubled(line):
            fixed_lines.append(line[::2])
        else:
            fixed_lines.append(line)
    return "\n".join(fixed_lines)


def _retrieval_context(docs: List[Document]) -> str:
    """Formatea documentos recuperados como contexto con cabeceras de trazabilidad."""
    blocks: list[str] = []
    for d in docs:
        md = d.metadata or {}
        content = _fix_doubled_text(d.page_content)
        doc_id = md.get("doc_id") or md.get("source", "desconocido")
        page = md.get("page", "N/A")
        chunk_id = md.get("chunk_id")
        if chunk_id:
            header = f"[doc_id={doc_id}; página={page}; chunk_id={chunk_id}]"
        else:
            header = f"[doc_id={doc_id}; página={page}]"
        blocks.append(f"{header}\n{content}")
    return "\n\n---\n\n".join(blocks)


@tool
def listar_modelos_disponibles(marca: str = "") -> str:
    """Retorna el catalogo de modelos indexados en la base de conocimiento.

    Consulta directamente la coleccion ChromaDB (sin similarity search)
    para listar todos los modelos unicos agrupados por marca.
    Si se indica una marca, filtra solo los modelos de esa marca.

    Usada cuando el usuario pregunta que modelos o vehiculos estan disponibles.

    Args:
        marca: Nombre de la marca a filtrar (ej: 'Toyota', 'Mazda'). Opcional.

    Returns:
        Lista formateada de modelos por marca, o mensaje si no hay resultados.
    """
    vs = get_active_vector_store()

    where = {"marca": marca} if marca else None
    result = vs._collection.get(where=where, include=["metadatas"])

    modelos_por_marca: dict[str, set[str]] = {}
    for meta in result["metadatas"]:
        m = meta.get("marca", "Desconocida")
        mod = meta.get("modelo", "Desconocido")
        modelos_por_marca.setdefault(m, set()).add(mod)

    if not modelos_por_marca:
        return "No se encontraron modelos en el catálogo."

    lineas = []
    for m in sorted(modelos_por_marca):
        for mod in sorted(modelos_por_marca[m]):
            lineas.append(f"- {m}: {mod}")

    return "Modelos disponibles:\n" + "\n".join(lineas)


@tool
def buscar_especificacion(especificacion: str, modelo: str) -> str:
    """Busca un dato tecnico puntual para un modelo especifico.

    Realiza busqueda MMR combinando la especificacion y el modelo
    para encontrar los chunks mas relevantes y diversos (k=6).

    Usada cuando el usuario pregunta por una caracteristica tecnica concreta
    como potencia, torque, autonomia, consumo o dimensiones.

    Args:
        especificacion: El dato tecnico buscado (ej: 'potencia', 'torque', 'autonomia').
        modelo:         El nombre del modelo del vehiculo (ej: 'Hilux', 'CX-5').

    Returns:
        Fragmentos de contexto con metadata [source, pagina], o mensaje si no hay datos.
    """
    vs = get_active_vector_store()

    # Excluir chunks de web_fallback para no contaminar la busqueda
    base_filter = {"origen": {"$ne": "web_fallback"}}

    # Query enriquecida: usar terminos que aparecen en las tablas de specs de los PDFs
    # ("maxima", "motor", "kg-m", "Nm", "hp", "rpm") para que el MMR rankee mejor
    # los chunks con datos numericos en lugar de los chunks de marketing.
    rich_query = f"{especificacion} maximo motor especificaciones tecnicas {modelo}"

    # Intento 1: con filtro de modelo (variantes) - mejor precision
    where_filter = {
        "$and": [
            base_filter,
            {"modelo": {"$in": _model_variants(modelo)}},
        ]
    }
    results = vs.max_marginal_relevance_search(
        rich_query,
        k=8,
        fetch_k=25,
        lambda_mult=0.6,  # mas diversidad para cubrir distintas versiones del motor
        filter=where_filter,
    )

    # Intento 2: sin filtro de modelo (solo excluir web_fallback) - mas recall
    if not results:
        results = vs.max_marginal_relevance_search(
            rich_query,
            k=8,
            fetch_k=25,
            lambda_mult=0.6,
            filter=base_filter,
        )

    if not results:
        return f"No se encontró información sobre '{especificacion}' para el modelo '{modelo}'."

    # Usar _retrieval_context para formato estandar [doc_id=...; página=...; chunk_id=...]
    # asi el parser del react_agent puede extraer la metadata correctamente y
    # el generate_grounded puede citar correctamente los chunks.
    return _retrieval_context(results)


@tool
def buscar_por_marca(marca: str) -> str:
    """Recupera informacion general de todos los modelos de una marca especifica.

    Realiza similarity search con filtro de metadata por marca (k=10).
    Util para preguntas sobre el catalogo completo de una marca.

    Usada cuando el usuario pregunta por una marca en general o quiere
    explorar modelos de una misma marca.

    Args:
        marca: Nombre de la marca (ej: 'Toyota', 'Volkswagen', 'Mazda').

    Returns:
        Fragmentos de contexto con metadata [modelo, pagina], o mensaje si no hay datos.
    """
    vs = get_active_vector_store()

    # Filtrar por marca y excluir web_fallback
    where_filter = {
        "$and": [
            {"origen": {"$ne": "web_fallback"}},
            {"marca": marca},
        ]
    }
    results = vs.max_marginal_relevance_search(
        marca,
        k=10,
        fetch_k=30,
        lambda_mult=0.5,
        filter=where_filter,
    )

    if not results:
        return f"No se encontró información para la marca '{marca}'."

    # Formato estandar para que el parser del react_agent extraiga metadata
    return f"Información de {marca}:\n\n" + _retrieval_context(results)


@tool
def comparar_modelos(modelo1: str, modelo2: str) -> str:
    """Genera una tabla comparativa en markdown entre dos modelos.

    Proceso:
    1. Busca chunks de cada modelo por separado (k=8 cada uno)
    2. Envia ambos contextos al LLM con instrucciones de formato
    3. El LLM genera tabla markdown solo con datos reales disponibles
    4. Si faltan demasiados datos, genera bullets explicativos en vez de tabla

    Usada cuando el usuario quiere comparar dos vehiculos entre si.

    Args:
        modelo1: Nombre del primer modelo (ej: 'Hilux', 'Corolla Cross').
        modelo2: Nombre del segundo modelo (ej: 'Fortuner', 'Yaris Cross').

    Returns:
        Tabla comparativa markdown o explicacion de datos faltantes.
    """
    vs = get_active_vector_store()

    def _buscar(modelo: str):
        """Busca chunks relevantes y diversos para un modelo especifico via MMR."""
        docs = vs.max_marginal_relevance_search(
            modelo, k=8, fetch_k=25, lambda_mult=0.5,
        )
        return "\n\n".join(d.page_content for d in docs)

    ctx1 = _buscar(modelo1)
    ctx2 = _buscar(modelo2)

    if not ctx1 and not ctx2:
        return f"No se encontró información para '{modelo1}' ni '{modelo2}'."
    prompt = f"""Eres un experto en fichas tecnicas de vehiculos.
Con base UNICAMENTE en la informacion proporcionada, genera una comparativa clara
en markdown entre **{modelo1}** y **{modelo2}**.

Reglas de formato:
- Si hay datos comparables, usa una tabla markdown limpia.
- Incluye solo filas con informacion real para al menos uno de los modelos.
- No llenes la tabla con N/D masivo.
- Si faltan demasiados datos para comparar, NO hagas tabla: responde en 2-4 bullets
  explicando que no hay informacion suficiente y que datos faltan.
- Cierra con una recomendacion corta (1-2 lineas) solo si hay sustento en los datos.

### Informacion de {modelo1}:
{ctx1}

### Informacion de {modelo2}:
{ctx2}
"""

    response = _get_llm().invoke(prompt)
    return str(response.content)


@tool
def resumir_ficha(modelo: str) -> str:
    """Genera un resumen estructurado en markdown de la ficha tecnica de un modelo.

    Proceso:
    1. Busca chunks del modelo (k=10 para cobertura amplia)
    2. Envia contexto al LLM con instrucciones de formato por secciones
    3. El LLM organiza en: Motor, Rendimiento, Dimensiones, Equipamiento, Versiones
    4. Si hay pocos datos, agrega seccion 'Datos faltantes' con bullets

    Usada cuando el usuario pide un resumen, overview o descripcion general.

    Args:
        modelo: Nombre del modelo (ej: 'Prado', 'BZ4X', 'Mazda Cx 5 2026').

    Returns:
        Resumen estructurado en markdown o mensaje si no hay datos.
    """
    vs = get_active_vector_store()

    docs = vs.max_marginal_relevance_search(
        modelo, k=10, fetch_k=30, lambda_mult=0.5,
    )

    if not docs:
        return f"No se encontró información para el modelo '{modelo}'."

    ctx = "\n\n".join(d.page_content for d in docs)
    prompt = f"""Eres un experto en fichas tecnicas de vehiculos.
Con base UNICAMENTE en la siguiente informacion, genera un resumen estructurado
en markdown del **{modelo}**.

Reglas de formato:
- Usa titulo y secciones cortas, faciles de leer.
- Organiza en estas secciones y omite las que no tengan datos:
  - Motor y transmision
  - Rendimiento y consumo
  - Dimensiones y capacidades
  - Equipamiento destacado
  - Versiones disponibles
- No repitas frases de disculpa.
- Si hay pocos datos, usa una seccion final 'Datos faltantes' con bullets.
- Evita tablas enormes con N/D.

### Informacion disponible:
{ctx}
"""

    response = _get_llm().invoke(prompt)
    return str(response.content)


# ── Nuevas tools para el agente ReAct ───────────────────────────────────────


@tool
def buscar_vectorial(query: str, k: int = 6, modelo: str = "", marca: str = "") -> str:
    """Busqueda semantica en la base de conocimiento vectorial (ChromaDB).

    Realiza similarity search con filtros opcionales de metadata por modelo y/o marca.
    Genera variantes del nombre del modelo para matching flexible.
    Si el filtro no retorna resultados, reintenta sin filtro como fallback.

    Usada como herramienta principal de recuperacion de informacion.

    Args:
        query:  Texto de busqueda (pregunta o terminos tecnicos).
        k:      Numero de chunks a recuperar (default: 6).
        modelo: Nombre del modelo para filtrar (ej: 'Hilux', 'CX-5'). Opcional.
        marca:  Nombre de la marca para filtrar (ej: 'Toyota', 'Mazda'). Opcional.

    Returns:
        Fragmentos de contexto con cabeceras [doc_id; pagina; chunk_id].
    """
    vs = get_active_vector_store()

    # Filtro base: excluir chunks de web_fallback (acumulados de busquedas web pasadas)
    # para que no contaminen los resultados de los PDFs reales.
    base_filter = {"origen": {"$ne": "web_fallback"}}

    results = []

    # Estrategia de fallback en cascada para maximizar la probabilidad de match:
    # 1) Filtro estricto por modelo (con variantes) + base_filter
    # 2) Si falla, filtro por marca + base_filter (aprovecha que el clasificador
    #    extrajo marca aunque el modelo no haya matcheado variantes)
    # 3) Si falla, solo base_filter (busqueda semantica pura sin restriccion)

    # Intento 1: modelo + marca
    if modelo and len(modelo) >= 2:
        where_filter = {
            "$and": [
                base_filter,
                {"modelo": {"$in": _model_variants(modelo, marca or None)}},
            ]
        }
        results = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=k * 3, lambda_mult=0.7, filter=where_filter,
        )

    # Intento 2: solo marca (cuando el filtro de modelo no matcheo nada)
    if not results and marca and len(marca) >= 2:
        where_filter = {"$and": [base_filter, {"marca": marca}]}
        results = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=k * 3, lambda_mult=0.7, filter=where_filter,
        )

    # Intento 3: solo base_filter (sin restricciones de marca/modelo)
    if not results:
        results = vs.max_marginal_relevance_search(
            query, k=k, fetch_k=k * 3, lambda_mult=0.7, filter=base_filter,
        )

    if not results:
        return "No se encontraron documentos relevantes en la base de conocimiento."

    return _retrieval_context(results)


@tool
def buscar_hyde(pregunta: str, k: int = 6) -> str:
    """HyDE (Hypothetical Document Embeddings): genera un documento hipotetico
    y lo usa como query para busqueda semantica, mejorando la relevancia.

    Proceso:
    1. El LLM genera una respuesta hipotetica a la pregunta
    2. Ese texto hipotetico se usa como query para similarity_search
    3. Los documentos reales similares al hipotetico se retornan

    Usada cuando la busqueda vectorial directa no retorna resultados relevantes
    o cuando la pregunta es abstracta y necesita reformularse.

    Args:
        pregunta: Pregunta del usuario en lenguaje natural.
        k:        Numero de chunks a recuperar (default: 6).

    Returns:
        Fragmentos de contexto encontrados via HyDE.
    """
    llm = _get_llm()

    hypo_response = llm.invoke(
        "Genera un parrafo tecnico breve (maximo 150 palabras) que responda "
        "esta pregunta sobre fichas tecnicas vehiculares. Incluye especificaciones "
        f"numericas plausibles:\n\n{pregunta}"
    )
    hypo_text = str(hypo_response.content)

    vs = get_active_vector_store()
    # Excluir chunks de web_fallback acumulados para no contaminar resultados
    results = vs.max_marginal_relevance_search(
        hypo_text, k=k, fetch_k=k * 3, lambda_mult=0.7,
        filter={"origen": {"$ne": "web_fallback"}},
    )

    if not results:
        return "HyDE no encontro documentos relevantes."

    return f"[HyDE] Documento hipotetico: {hypo_text[:200]}...\n\n" + _retrieval_context(results)


@tool
def descomponer_pregunta(pregunta: str) -> str:
    """Descompone una pregunta compleja en 2-4 sub-preguntas mas simples y especificas.

    Usada cuando la pregunta del usuario involucra multiples aspectos, modelos
    o comparaciones complejas que se benefician de busquedas separadas.

    Args:
        pregunta: Pregunta compleja del usuario.

    Returns:
        Lista numerada de sub-preguntas.
    """
    llm = _get_llm()

    response = llm.invoke(
        "Descompone la siguiente pregunta sobre vehiculos en 2-4 sub-preguntas "
        "mas simples y especificas. Cada sub-pregunta debe poder responderse "
        "con una busqueda independiente en una base de fichas tecnicas.\n"
        "Devuelve SOLO las sub-preguntas numeradas, sin explicacion.\n\n"
        f"Pregunta: {pregunta}"
    )
    return str(response.content)


@tool
def buscar_web(query: str) -> str:
    """Busca informacion en internet usando DuckDuckGo.

    Usada UNICAMENTE como ultimo recurso cuando la base de conocimiento interna
    no tiene respuesta despues de multiples reintentos.

    Args:
        query: Consulta de busqueda en lenguaje natural.

    Returns:
        Resultados de busqueda web formateados.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))

        if not results:
            return "No se encontraron resultados en la web."

        formatted = []
        for r in results:
            formatted.append(
                f"**{r.get('title', 'Sin titulo')}**\n"
                f"URL: {r.get('href', '')}\n"
                f"{r.get('body', '')}"
            )
        return "\n\n---\n\n".join(formatted)
    except ImportError:
        return "Error: paquete duckduckgo-search no instalado. Ejecutar: pip install duckduckgo-search"
    except Exception as e:
        return f"Error en busqueda web: {e}"


@tool
def consultar_grafo_conocimiento(
    accion: str,
    modelo: str = "",
    modelo2: str = "",
    marca: str = "",
    autonomia_minima: float = 300.0,
) -> str:
    """Consulta el Knowledge Graph (ontologia OWL) via SPARQL para extraer
    relaciones estructuradas y datos enriquecidos sobre vehiculos.

    Esta tool complementa la busqueda vectorial: mientras los embeddings encuentran
    contenido similar en lenguaje natural, el KG proporciona datos estructurados
    precisos y relaciones semanticas entre entidades (modelo-marca-motor-categoria-etc).

    Acciones disponibles:
    - "especificaciones" : datos generales del modelo (peso, longitud, baul, precio, anyo)
    - "motor"            : datos del motor (potencia, cilindrada, combustible, autonomia, bateria)
    - "comparar"         : compara dos modelos lado a lado (requiere modelo y modelo2)
    - "por_marca"        : lista todos los modelos de una marca
    - "electricos"       : vehiculos electricos con autonomia >= autonomia_minima
    - "seguridad"        : sistemas de seguridad de un modelo

    Usar cuando se necesiten:
    - Datos numericos exactos (peso, precio, dimensiones)
    - Relaciones estructuradas (modelo -> motor -> combustible)
    - Filtros tipo "electricos con autonomia > 400 km"
    - Comparaciones precisas con valores

    Args:
        accion:           Una de las acciones listadas arriba.
        modelo:           Nombre del modelo (ej: "Hilux", "ZS EV"). Requerido para
                          especificaciones, motor, comparar, seguridad.
        modelo2:          Segundo modelo (solo para accion="comparar").
        marca:            Nombre de la marca (solo para accion="por_marca").
        autonomia_minima: Umbral de autonomia en km (solo para accion="electricos").

    Returns:
        Datos estructurados del KG formateados como texto, o mensaje de error.
    """
    try:
        from kg_retriever import (
            kg_buscar_especificaciones,
            kg_buscar_motor,
            kg_comparar_modelos,
            kg_listar_modelos_por_marca,
            kg_electricos_por_autonomia,
            kg_sistemas_seguridad,
            kg_format_para_llm,
        )
    except ImportError as e:
        return f"Error importando kg_retriever: {e}. Verifica que SPARQLWrapper este instalado."

    accion_lower = accion.lower().strip()

    try:
        if accion_lower == "especificaciones":
            if not modelo:
                return "Error: 'especificaciones' requiere el parametro 'modelo'."
            results = kg_buscar_especificaciones(modelo)
            return kg_format_para_llm(results, f"especificaciones de {modelo}")

        elif accion_lower == "motor":
            if not modelo:
                return "Error: 'motor' requiere el parametro 'modelo'."
            results = kg_buscar_motor(modelo)
            return kg_format_para_llm(results, f"motor de {modelo}")

        elif accion_lower == "comparar":
            if not modelo or not modelo2:
                return "Error: 'comparar' requiere los parametros 'modelo' y 'modelo2'."
            results = kg_comparar_modelos(modelo, modelo2)
            return kg_format_para_llm(results, f"comparativa {modelo} vs {modelo2}")

        elif accion_lower == "por_marca":
            if not marca:
                return "Error: 'por_marca' requiere el parametro 'marca'."
            results = kg_listar_modelos_por_marca(marca)
            return kg_format_para_llm(results, f"modelos de {marca}")

        elif accion_lower == "electricos":
            results = kg_electricos_por_autonomia(autonomia_minima)
            return kg_format_para_llm(results, f"electricos con autonomia >= {autonomia_minima} km")

        elif accion_lower == "seguridad":
            if not modelo:
                return "Error: 'seguridad' requiere el parametro 'modelo'."
            results = kg_sistemas_seguridad(modelo)
            return kg_format_para_llm(results, f"sistemas de seguridad de {modelo}")

        else:
            return (
                f"Accion desconocida: '{accion}'. "
                "Acciones validas: especificaciones, motor, comparar, por_marca, electricos, seguridad."
            )
    except Exception as e:
        return f"Error consultando el grafo de conocimiento: {e}"
