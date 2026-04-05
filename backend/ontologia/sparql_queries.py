# backend/ontologia/sparql_queries.py
# ==============================================================================
# Consultas SPARQL implementadas con RDFLib conectándose a GraphDB.
#
# Cubre todos los tipos requeridos:
#   SELECT, FILTER, ORDER BY, LIMIT, UPDATE (INSERT DATA + DELETE DATA + DELETE/INSERT)
#
# Prerequisitos:
#   pip install SPARQLWrapper rdflib
#
# GraphDB debe estar corriendo en localhost:7200 con un repositorio llamado "vehiculos"
# que tenga cargada la ontología vehiculos.ttl.
# ==============================================================================

from SPARQLWrapper import SPARQLWrapper, JSON, POST, GET
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed
import json

# ── Configuración de conexión ──────────────────────────────────────────────────
GRAPHDB_ENDPOINT   = "http://localhost:7200/repositories/vehiculos"
GRAPHDB_UPDATE_EP  = "http://localhost:7200/repositories/vehiculos/statements"
BASE_PREFIX        = "http://www.semanticweb.org/rag-ontologica/vehiculos#"

PREFIXES = """
PREFIX :     <http://www.semanticweb.org/rag-ontologica/vehiculos#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


def _sparql_select(query: str) -> list[dict]:
    """Ejecuta una consulta SELECT contra GraphDB y retorna lista de resultados."""
    sparql = SPARQLWrapper(GRAPHDB_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(GET)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def _sparql_update(query: str) -> None:
    """Ejecuta una consulta UPDATE (INSERT/DELETE) contra GraphDB."""
    sparql = SPARQLWrapper(GRAPHDB_UPDATE_EP)
    sparql.setQuery(query)
    sparql.setMethod(POST)
    sparql.query()


def _print_results(resultados: list[dict], titulo: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {titulo}")
    print(f"{'='*60}")
    for row in resultados:
        fila = {k: v["value"] for k, v in row.items()}
        print(json.dumps(fila, ensure_ascii=False, indent=2))
    print(f"  -> {len(resultados)} resultado(s)\n")


# ==============================================================================
# 1. CONSULTA SELECT — Todos los vehículos con su marca y modelo
# ==============================================================================
def consulta_select_todos_los_vehiculos():
    """SELECT básico: recupera todos los vehículos con marca y nombre de modelo."""
    query = PREFIXES + """
    SELECT DISTINCT ?vehiculo ?nombreModelo ?marca
    WHERE {
        ?vehiculo a/rdfs:subClassOf* :Vehiculo .
        ?vehiculo :tieneNombreModelo ?nombreModelo .
        ?vehiculo :tieneMarca       ?marca .
    }
    """
    resultados = _sparql_select(query)
    _print_results(resultados, "SELECT – Todos los vehículos con marca y modelo")
    return resultados


# ==============================================================================
# 2. CONSULTA con FILTER — Vehículos eléctricos con autonomía mayor a 300 km
# ==============================================================================
def consulta_filter_autonomia():
    """SELECT + FILTER: vehículos eléctricos con autonomía WLTP > 300 km."""
    query = PREFIXES + """
    SELECT ?vehiculo ?modelo ?autonomia ?bateria
    WHERE {
        ?vehiculo a :VehiculoElectrico .
        ?vehiculo :tieneNombreModelo    ?modelo .
        ?vehiculo :tieneAutonomiaKm     ?autonomia .
        ?vehiculo :tieneCapacidadBateriaKwh ?bateria .
        FILTER (?autonomia > 300.0)
    }
    """
    resultados = _sparql_select(query)
    _print_results(resultados, "FILTER – Eléctricos con autonomía > 300 km")
    return resultados


# ==============================================================================
# 3. CONSULTA con ORDER BY — Vehículos ordenados por peso ascendente
# ==============================================================================
def consulta_order_by_peso():
    """SELECT + ORDER BY: todos los vehículos ordenados de menor a mayor peso."""
    query = PREFIXES + """
    SELECT ?modelo ?marca ?tipo ?peso
    WHERE {
        ?vehiculo :tieneNombreModelo ?modelo .
        ?vehiculo :tieneMarca        ?marca .
        ?vehiculo :tienePeso         ?peso .
        ?vehiculo a                  ?tipo .
        FILTER (?tipo IN (:VehiculoCombustion, :VehiculoElectrico, :VehiculoHibrido))
    }
    ORDER BY ASC(?peso)
    """
    resultados = _sparql_select(query)
    _print_results(resultados, "ORDER BY – Vehículos ordenados por peso (ASC)")
    return resultados


# ==============================================================================
# 4. CONSULTA con LIMIT — Top 3 vehículos más baratos
# ==============================================================================
def consulta_limit_mas_baratos():
    """SELECT + ORDER BY + LIMIT: top 3 motores con mayor potencia en CV."""
    query = PREFIXES + """
    SELECT ?modelo ?marca ?potencia ?tipoMotor
    WHERE {
        ?vehiculo :tieneNombreModelo ?modelo .
        ?vehiculo :tieneMarca        ?marca .
        ?vehiculo :tieneMotor        ?motor .
        ?motor    :tienePotenciaCV   ?potencia .
        BIND(IF(EXISTS{?motor a :MotorElectrico}, "Electrico", "Combustion") AS ?tipoMotor)
    }
    ORDER BY DESC(?potencia)
    LIMIT 3
    """
    resultados = _sparql_select(query)
    _print_results(resultados, "LIMIT – Top 3 motores mas potentes")
    return resultados


# ==============================================================================
# 5. CONSULTA UPDATE — Operando 1: INSERT DATA
#    Agrega un nuevo vehículo eléctrico (MG4)
# ==============================================================================
def consulta_update_insert_data():
    """UPDATE INSERT DATA: insertar un nuevo individuo MG4 como VehiculoElectrico."""
    query = PREFIXES + """
    INSERT DATA {
        :MG4_Electric a :VehiculoElectrico ;
            :tieneNombreModelo        "MG4 Electric" ;
            :tieneMarca               "MG Emotor" ;
            :tieneAnyoLanzamiento     "2024"^^xsd:gYear ;
            :fabricadoPor             :MGMotor ;
            :perteneceACategoria      :Hatchback ;
            :tieneTransmision         :TransmisionEV ;
            :tienePeso                1655.0 ;
            :tieneAutonomiaKm         435.0 ;
            :tieneCapacidadBateriaKwh 64.0 ;
            :tieneLongitudMm          4287 ;
            :tieneCapacidadBaulL      363 ;
            :tienePrecioBase          28990.0 .
    }
    """
    _sparql_update(query)
    print("\n[OK] INSERT DATA ejecutado: MG4_Electric añadido al grafo.")


# ==============================================================================
# 6. CONSULTA UPDATE — Operando 2: DELETE DATA
#    Elimina el precio base del MG4 insertado anteriormente
# ==============================================================================
def consulta_update_delete_data():
    """UPDATE DELETE DATA: eliminar el precio base del MG4."""
    query = PREFIXES + """
    DELETE DATA {
        :MG4_Electric :tienePrecioBase 28990.0 .
    }
    """
    _sparql_update(query)
    print("\n[OK] DELETE DATA ejecutado: precio base de MG4 eliminado del grafo.")


# ==============================================================================
# 7. CONSULTA UPDATE — Operando 3: DELETE + INSERT (actualización atómica)
#    Actualiza el peso del Toyota Corolla
# ==============================================================================
def consulta_update_delete_insert():
    """UPDATE DELETE/INSERT WHERE: actualiza el peso del ToyotaCorolla."""
    query = PREFIXES + """
    DELETE {
        :ToyotaCorolla :tienePeso ?pesoAntiguo .
    }
    INSERT {
        :ToyotaCorolla :tienePeso 1340.0 .
    }
    WHERE {
        :ToyotaCorolla :tienePeso ?pesoAntiguo .
    }
    """
    _sparql_update(query)
    print("\n[OK] DELETE+INSERT ejecutado: peso del ToyotaCorolla actualizado a 1340.0 kg.")


# ==============================================================================
# CONSULTAS ADICIONALES ÚTILES PARA EL RAG
# ==============================================================================

def consulta_comparar_electricos():
    """Compara autonomía y precio de todos los vehículos eléctricos."""
    query = PREFIXES + """
    SELECT ?modelo ?marca ?autonomia ?bateria ?precio
    WHERE {
        ?v a :VehiculoElectrico .
        ?v :tieneNombreModelo        ?modelo .
        ?v :tieneMarca               ?marca .
        ?v :tieneAutonomiaKm         ?autonomia .
        ?v :tieneCapacidadBateriaKwh ?bateria .
        OPTIONAL { ?v :tienePrecioBase ?precio }
    }
    ORDER BY DESC(?autonomia)
    """
    resultados = _sparql_select(query)
    _print_results(resultados, "COMPARACIÓN – Eléctricos por autonomía (DESC)")
    return resultados


def consulta_hibridos_toyota():
    """Obtiene todos los vehículos híbridos de Toyota con su consumo."""
    query = PREFIXES + """
    SELECT ?modelo ?consumo ?precio
    WHERE {
        ?v a :VehiculoHibrido .
        ?v :fabricadoPor     :Toyota .
        ?v :tieneNombreModelo ?modelo .
        OPTIONAL { ?v :tieneConsumoL100km ?consumo }
        OPTIONAL { ?v :tienePrecioBase    ?precio }
    }
    ORDER BY ASC(?consumo)
    """
    resultados = _sparql_select(query)
    _print_results(resultados, "HÍBRIDOS – Toyota ordenados por consumo")
    return resultados


def consulta_motor_por_vehiculo(nombre_modelo: str):
    """Recupera datos del motor para un modelo específico."""
    query = PREFIXES + f"""
    SELECT ?modelo ?tipoMotor ?potencia ?cilindrada ?combustible
    WHERE {{
        ?v :tieneNombreModelo "{nombre_modelo}" .
        ?v :tieneMotor ?motor .
        ?motor :tienePotenciaCV ?potencia .
        OPTIONAL {{ ?motor :tieneCilindradaCc ?cilindrada }}
        OPTIONAL {{ ?motor :usaCombustible/:rdfs:label ?combustible }}
        BIND(IF(EXISTS{{?motor a :MotorElectrico}}, "Eléctrico", "Combustión") AS ?tipoMotor)
        BIND("{nombre_modelo}" AS ?modelo)
    }}
    """
    resultados = _sparql_select(query)
    _print_results(resultados, f"MOTOR – Datos de {nombre_modelo}")
    return resultados


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("Conectando a GraphDB en", GRAPHDB_ENDPOINT)
    print("Asegurate de que GraphDB este corriendo y el repositorio 'vehiculos' exista.\n")

    # --- UPDATEs primero (enriquecen los datos antes de los SELECTs) ---

    # 5. UPDATE – INSERT DATA
    consulta_update_insert_data()

    # 6. UPDATE – DELETE DATA
    consulta_update_delete_data()

    # 7. UPDATE – DELETE + INSERT
    consulta_update_delete_insert()

    # --- SELECTs sobre datos completos ---

    # 1. SELECT basico
    consulta_select_todos_los_vehiculos()

    # 2. FILTER (usa MG4_Electric insertado arriba)
    consulta_filter_autonomia()

    # 3. ORDER BY
    consulta_order_by_peso()

    # 4. LIMIT
    consulta_limit_mas_baratos()

    # Adicionales
    consulta_comparar_electricos()
    consulta_hibridos_toyota()
    consulta_motor_por_vehiculo("Golf")
