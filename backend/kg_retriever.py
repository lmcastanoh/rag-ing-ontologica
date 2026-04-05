# backend/kg_retriever.py
# ==============================================================================
# Módulo de recuperación desde el Knowledge Graph (GraphDB).
#
# Provee funciones que el agente KG-RAG puede llamar para consultar la
# ontología OWL via SPARQL, complementando la búsqueda vectorial en ChromaDB.
# ==============================================================================

from __future__ import annotations

import logging
from typing import Any

from SPARQLWrapper import SPARQLWrapper, JSON, GET

logger = logging.getLogger(__name__)

GRAPHDB_ENDPOINT = "http://localhost:7200/repositories/vehiculos"

PREFIXES = """
PREFIX :     <http://www.semanticweb.org/rag-ontologica/vehiculos#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


def _run_sparql(query: str) -> list[dict[str, Any]]:
    """Ejecuta una consulta SELECT en GraphDB y retorna lista de filas."""
    try:
        sparql = SPARQLWrapper(GRAPHDB_ENDPOINT)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.setMethod(GET)
        raw = sparql.query().convert()
        return [
            {k: v["value"] for k, v in row.items()}
            for row in raw["results"]["bindings"]
        ]
    except Exception as e:
        logger.error("Error ejecutando SPARQL: %s", e)
        return []


def kg_buscar_especificaciones(modelo: str) -> list[dict]:
    """
    Recupera especificaciones técnicas estructuradas de un modelo desde el KG.

    Args:
        modelo: Nombre del modelo (ej. "Golf", "Corolla", "ZS EV").

    Returns:
        Lista de dicts con peso, longitud, baúl, precio, tipo de vehículo, etc.
    """
    query = PREFIXES + f"""
    SELECT ?nombreModelo ?marca ?tipo ?peso ?longitud ?baul ?precio ?anyo
    WHERE {{
        ?v :tieneNombreModelo ?nombreModelo .
        FILTER(LCASE(STR(?nombreModelo)) = LCASE("{modelo}"))
        ?v :tieneMarca  ?marca .
        ?v a            ?tipo .
        FILTER(?tipo IN (:VehiculoCombustion, :VehiculoElectrico, :VehiculoHibrido))
        OPTIONAL {{ ?v :tienePeso              ?peso }}
        OPTIONAL {{ ?v :tieneLongitudMm        ?longitud }}
        OPTIONAL {{ ?v :tieneCapacidadBaulL    ?baul }}
        OPTIONAL {{ ?v :tienePrecioBase        ?precio }}
        OPTIONAL {{ ?v :tieneAnyoLanzamiento   ?anyo }}
    }}
    """
    return _run_sparql(query)


def kg_buscar_motor(modelo: str) -> list[dict]:
    """
    Recupera datos del motor para un modelo (potencia, cilindrada, combustible, autonomía).

    Args:
        modelo: Nombre del modelo.

    Returns:
        Lista de dicts con tipo de motor, potencia, cilindrada y combustible.
    """
    query = PREFIXES + f"""
    SELECT ?nombreModelo ?tipoMotor ?potencia ?cilindrada ?combustible ?autonomia ?bateria
    WHERE {{
        ?v :tieneNombreModelo ?nombreModelo .
        FILTER(LCASE(STR(?nombreModelo)) = LCASE("{modelo}"))
        ?v :tieneMotor ?motor .
        ?motor :tienePotenciaCV ?potencia .
        OPTIONAL {{ ?motor :tieneCilindradaCc ?cilindrada }}
        OPTIONAL {{ ?motor :usaCombustible ?comb .
                   ?comb rdfs:label ?combustible . FILTER(LANG(?combustible) = "es") }}
        OPTIONAL {{ ?v :tieneAutonomiaKm         ?autonomia }}
        OPTIONAL {{ ?v :tieneCapacidadBateriaKwh  ?bateria }}
        BIND(IF(EXISTS{{?motor a :MotorElectrico}}, "Eléctrico", "Combustión") AS ?tipoMotor)
    }}
    """
    return _run_sparql(query)


def kg_comparar_modelos(modelo1: str, modelo2: str) -> list[dict]:
    """
    Recupera especificaciones de dos modelos para comparación directa.

    Args:
        modelo1: Primer modelo.
        modelo2: Segundo modelo.

    Returns:
        Lista con filas de ambos modelos (peso, longitud, baúl, consumo/autonomía, precio).
    """
    query = PREFIXES + f"""
    SELECT ?nombreModelo ?marca ?peso ?longitud ?baul ?precio ?consumo ?autonomia
    WHERE {{
        ?v :tieneNombreModelo ?nombreModelo .
        FILTER(
            LCASE(STR(?nombreModelo)) = LCASE("{modelo1}") ||
            LCASE(STR(?nombreModelo)) = LCASE("{modelo2}")
        )
        ?v :tieneMarca ?marca .
        OPTIONAL {{ ?v :tienePeso           ?peso }}
        OPTIONAL {{ ?v :tieneLongitudMm     ?longitud }}
        OPTIONAL {{ ?v :tieneCapacidadBaulL ?baul }}
        OPTIONAL {{ ?v :tienePrecioBase     ?precio }}
        OPTIONAL {{ ?v :tieneConsumoL100km  ?consumo }}
        OPTIONAL {{ ?v :tieneAutonomiaKm    ?autonomia }}
    }}
    ORDER BY ?nombreModelo
    """
    return _run_sparql(query)


def kg_listar_modelos_por_marca(marca: str) -> list[dict]:
    """
    Lista todos los modelos de una marca con su categoría y tipo de propulsión.

    Args:
        marca: Nombre de la marca (ej. "Toyota", "Volkswagen").

    Returns:
        Lista de dicts con modelo, categoría y tipo de vehículo.
    """
    query = PREFIXES + f"""
    SELECT ?nombreModelo ?categoria ?tipo ?precio
    WHERE {{
        ?v :tieneMarca ?marca .
        FILTER(LCASE(STR(?marca)) = LCASE("{marca}"))
        ?v :tieneNombreModelo ?nombreModelo .
        ?v a ?tipo .
        FILTER(?tipo IN (:VehiculoCombustion, :VehiculoElectrico, :VehiculoHibrido))
        OPTIONAL {{ ?v :perteneceACategoria/:rdfs:label ?categoria .
                   FILTER(LANG(?categoria) = "es") }}
        OPTIONAL {{ ?v :tienePrecioBase ?precio }}
    }}
    ORDER BY ?nombreModelo
    """
    return _run_sparql(query)


def kg_electricos_por_autonomia(autonomia_minima: float = 300.0) -> list[dict]:
    """
    Recupera vehículos eléctricos con autonomía mayor al umbral indicado.

    Args:
        autonomia_minima: Autonomía mínima en km (WLTP). Default 300 km.

    Returns:
        Lista de dicts ordenada por autonomía descendente.
    """
    query = PREFIXES + f"""
    SELECT ?nombreModelo ?marca ?autonomia ?bateria ?precio
    WHERE {{
        ?v a :VehiculoElectrico .
        ?v :tieneNombreModelo        ?nombreModelo .
        ?v :tieneMarca               ?marca .
        ?v :tieneAutonomiaKm         ?autonomia .
        ?v :tieneCapacidadBateriaKwh ?bateria .
        OPTIONAL {{ ?v :tienePrecioBase ?precio }}
        FILTER(?autonomia >= {autonomia_minima})
    }}
    ORDER BY DESC(?autonomia)
    """
    return _run_sparql(query)


def kg_sistemas_seguridad(modelo: str) -> list[dict]:
    """
    Recupera los sistemas de seguridad de un modelo.

    Args:
        modelo: Nombre del modelo.

    Returns:
        Lista de dicts con nombre del sistema de seguridad.
    """
    query = PREFIXES + f"""
    SELECT ?nombreModelo ?sistemaSeguridad
    WHERE {{
        ?v :tieneNombreModelo ?nombreModelo .
        FILTER(LCASE(STR(?nombreModelo)) = LCASE("{modelo}"))
        ?v :tieneSistemaSeguridad ?sist .
        ?sist rdfs:label ?sistemaSeguridad .
        FILTER(LANG(?sistemaSeguridad) = "es")
    }}
    """
    return _run_sparql(query)


def kg_format_para_llm(resultados: list[dict], contexto: str = "") -> str:
    """
    Formatea los resultados SPARQL como texto estructurado para el LLM.

    Args:
        resultados: Lista de dicts con resultados SPARQL.
        contexto: Descripción del tipo de consulta realizada.

    Returns:
        Texto formateado listo para incluir en el prompt del LLM.
    """
    if not resultados:
        return f"[KG] No se encontraron datos en el Knowledge Graph para: {contexto}"

    lineas = [f"[KG - {contexto}]"]
    for i, fila in enumerate(resultados, 1):
        campos = []
        for k, v in fila.items():
            # Limpiar URIs para mostrar solo el fragmento local
            if v.startswith("http"):
                v = v.split("#")[-1].split("/")[-1]
            campos.append(f"{k}: {v}")
        lineas.append(f"  [{i}] " + " | ".join(campos))

    return "\n".join(lineas)
