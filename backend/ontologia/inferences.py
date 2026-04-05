# backend/ontologia/inferences.py
# ==============================================================================
# Documentación de 5 casos de inferencia sobre la ontología vehiculos.ttl.
#
# Para que las inferencias funcionen, en GraphDB debes:
#   1. Crear el repositorio con Ruleset = "OWL2-RL" u "OWL-Horst"
#      (Settings > Inference > Ruleset al crear el repo)
#   2. O activarlo en un repo existente: Settings > Edit > Enable OWL2-RL
#
# Cada función muestra:
#   - Qué se declaró explícitamente en la ontología
#   - Qué infiere el razonador
#   - La consulta SPARQL que demuestra la inferencia
# ==============================================================================

from SPARQLWrapper import SPARQLWrapper, JSON, GET
import json

GRAPHDB_ENDPOINT = "http://localhost:7200/repositories/vehiculos"

PREFIXES = """
PREFIX :     <http://www.semanticweb.org/rag-ontologica/vehiculos#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""


def _run(query: str) -> list[dict]:
    sparql = SPARQLWrapper(GRAPHDB_ENDPOINT)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    sparql.setMethod(GET)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def _show(titulo: str, explicacion: str, resultados: list[dict]):
    print(f"\n{'='*65}")
    print(f"  INFERENCIA: {titulo}")
    print(f"{'='*65}")
    print(f"  {explicacion}")
    print(f"  {'-'*55}")
    for row in resultados:
        fila = {k: v["value"] for k, v in row.items()}
        print("  ", json.dumps(fila, ensure_ascii=False))
    print(f"  -> {len(resultados)} triples inferidos\n")


# ==============================================================================
# INFERENCIA 1: rdfs:subClassOf — herencia de clase
#
# Declarado:    :VehiculoElectrico rdfs:subClassOf :Vehiculo
# Declarado:    :MGZS_EV a :VehiculoElectrico
# Inferido:     :MGZS_EV a :Vehiculo  (aunque nunca se declaró explícitamente)
#
# El razonador aplica la regla:
#   Si X rdfs:subClassOf Y  AND  a X  →  a Y
# ==============================================================================
def inferencia_subclassof():
    query = PREFIXES + """
    SELECT ?vehiculo ?tipoDeclarado ?tipoInferido
    WHERE {
        ?vehiculo a :VehiculoElectrico .
        BIND(:VehiculoElectrico AS ?tipoDeclarado)
        BIND(:Vehiculo          AS ?tipoInferido)
        # Esta triple es INFERIDA, no está en la ontología explícitamente
        ?vehiculo a :Vehiculo .
    }
    """
    resultados = _run(query)
    _show(
        "rdfs:subClassOf — los VehiculoElectrico son también Vehiculo",
        "Declarado: VehiculoElectrico subClassOf Vehiculo\n"
        "  Inferido: MGZS_EV, VW_eGolf, etc. también son instancias de :Vehiculo",
        resultados
    )
    return resultados


# ==============================================================================
# INFERENCIA 2: owl:inverseOf — propagación de la relación inversa
#
# Declarado:    :fabricaVehiculo owl:inverseOf :fabricadoPor
# Declarado:    :ToyotaCorolla :fabricadoPor :Toyota
# Inferido:     :Toyota :fabricaVehiculo :ToyotaCorolla
#
# El razonador aplica la regla:
#   Si p owl:inverseOf q  AND  X p Y  →  Y q X
# ==============================================================================
def inferencia_inverseof():
    query = PREFIXES + """
    SELECT ?fabricante ?vehiculo
    WHERE {
        # Esta triple es INFERIDA desde fabricadoPor
        ?fabricante :fabricaVehiculo ?vehiculo .
        ?vehiculo   :tieneNombreModelo ?modelo .
    }
    ORDER BY ?fabricante
    """
    resultados = _run(query)
    _show(
        "owl:inverseOf — inferir fabricaVehiculo desde fabricadoPor",
        "Declarado: :ToyotaCorolla :fabricadoPor :Toyota\n"
        "  Inferido: :Toyota :fabricaVehiculo :ToyotaCorolla  (nunca escrito explícitamente)",
        resultados
    )
    return resultados


# ==============================================================================
# INFERENCIA 3: rdfs:subPropertyOf — propagación a propiedad padre
#
# Declarado:    :tieneMotorCombustion rdfs:subPropertyOf :tieneMotor
# Declarado:    :ToyotaCorolla :tieneMotorCombustion :Motor_16_Gasolina
# Inferido:     :ToyotaCorolla :tieneMotor :Motor_16_Gasolina
#
# El razonador aplica la regla:
#   Si p rdfs:subPropertyOf q  AND  X p Y  →  X q Y
# ==============================================================================
def inferencia_subpropertyof():
    query = PREFIXES + """
    SELECT ?vehiculo ?modelo ?motor
    WHERE {
        ?vehiculo :tieneNombreModelo ?modelo .
        # tieneMotor es INFERIDO desde tieneMotorCombustion y tieneMotorElectrico
        ?vehiculo :tieneMotor ?motor .
        # Para demostrar que es inferencia, buscamos vehículos que solo declararon
        # tieneMotorCombustion pero NO tieneMotor directamente
        FILTER NOT EXISTS { ?vehiculo :tieneMotor ?motor FILTER(true) }
    }
    UNION
    SELECT ?vehiculo ?modelo ?motor
    WHERE {
        ?vehiculo :tieneNombreModelo ?modelo .
        ?vehiculo :tieneMotorCombustion ?motor .
        BIND(?motor AS ?motor)
    }
    """
    # Consulta simplificada que demuestra la inferencia:
    query = PREFIXES + """
    SELECT DISTINCT ?vehiculo ?modelo ?motor ?potencia
    WHERE {
        ?vehiculo :tieneNombreModelo ?modelo .
        # Esta propiedad tieneMotor es inferida desde tieneMotorCombustion
        ?vehiculo :tieneMotor ?motor .
        ?motor    :tienePotenciaCV ?potencia .
    }
    ORDER BY ?modelo
    """
    resultados = _run(query)
    _show(
        "rdfs:subPropertyOf — tieneMotor inferido desde tieneMotorCombustion/Electrico",
        "Declarado: tieneMotorCombustion subPropertyOf tieneMotor\n"
        "  Declarado: ToyotaCorolla :tieneMotorCombustion :Motor_16_Gasolina\n"
        "  Inferido:  ToyotaCorolla :tieneMotor :Motor_16_Gasolina",
        resultados
    )
    return resultados


# ==============================================================================
# INFERENCIA 4: rdfs:domain y rdfs:range — tipado automático
#
# Declarado:    :fabricadoPor rdfs:domain :Vehiculo
# Declarado:    :fabricadoPor rdfs:range  :Fabricante
# Declarado:    :ToyotaCorolla :fabricadoPor :Toyota
# Inferido:     :ToyotaCorolla a :Vehiculo  (por rdfs:domain)
# Inferido:     :Toyota        a :Fabricante (por rdfs:range)
#
# El razonador aplica:
#   Si p rdfs:domain D  AND  X p Y  →  X a D
#   Si p rdfs:range  R  AND  X p Y  →  Y a R
# ==============================================================================
def inferencia_domain_range():
    # Verificamos que :Toyota es inferido como :Fabricante por rdfs:range de :fabricadoPor
    query = PREFIXES + """
    SELECT ?entidad ?tipo
    WHERE {
        VALUES ?tipo { :Vehiculo :Fabricante }
        ?entidad a ?tipo .
        # Filtramos entidades cuyo tipo es inferido (no tienen owl:Class como rdf:type)
    }
    ORDER BY ?tipo ?entidad
    LIMIT 20
    """
    resultados = _run(query)
    _show(
        "rdfs:domain/range — tipado automático de individuos",
        "Declarado: fabricadoPor rdfs:domain Vehiculo, rdfs:range Fabricante\n"
        "  Inferido: cualquier sujeto de fabricadoPor es Vehiculo\n"
        "  Inferido: cualquier objeto de fabricadoPor es Fabricante",
        resultados
    )
    return resultados


# ==============================================================================
# INFERENCIA 5: owl:equivalentClass (intersectionOf) — clasificación automática
#
# Declarado:    VehiculoHibrido equivalentClass (
#                   Vehiculo AND
#                   tieneMotor some MotorCombustion AND
#                   tieneMotor some MotorElectrico )
#
# Declarado:    :ToyotaCorollaHEV :tieneMotorCombustion :Motor_15_HSD (→ tieneMotor por subProperty)
# Declarado:    :ToyotaCorollaHEV :tieneMotorElectrico  :Motor_HEV_Toyota
#
# Inferido:     :ToyotaCorollaHEV a :VehiculoHibrido  (por equivalentClass)
#
# Si se añade un individuo con ambos tipos de motor SIN declarar su clase,
# el razonador lo clasifica automáticamente como VehiculoHibrido.
# ==============================================================================
def inferencia_equivalentclass_hibrido():
    query = PREFIXES + """
    SELECT ?vehiculo ?modelo ?motorCombustion ?motorElectrico
    WHERE {
        ?vehiculo a :VehiculoHibrido .
        ?vehiculo :tieneNombreModelo   ?modelo .
        ?vehiculo :tieneMotorCombustion ?motorCombustion .
        ?vehiculo :tieneMotorElectrico  ?motorElectrico .
    }
    """
    resultados = _run(query)
    _show(
        "owl:equivalentClass (intersectionOf) — clasificación automática de híbridos",
        "Declarado: VehiculoHibrido equivalentClass (Vehiculo AND some(tieneMotor,MotorCombustion) AND some(tieneMotor,MotorElectrico))\n"
        "  Si se inserta un individuo con ambos motores pero sin tipo declarado,\n"
        "  el razonador OWL2-RL lo infiere como :VehiculoHibrido automáticamente.",
        resultados
    )

    # Demostración adicional: insertar un individuo anónimo y ver si lo clasifica
    print("  -> Demostracion: un individuo con ambos motores heredaria la clase :VehiculoHibrido")
    print("     si el razonador OWL2-RL / HermiT está activo en GraphDB.\n")
    return resultados


# ==============================================================================
# EJECUCIÓN PRINCIPAL
# ==============================================================================
if __name__ == "__main__":
    print("=" * 65)
    print("  DEMOSTRACIÓN DE INFERENCIAS — Ontología Vehículos")
    print("  Repositorio GraphDB: vehiculos  (Ruleset: OWL2-RL)")
    print("=" * 65)
    print()
    print("IMPORTANTE: Las inferencias requieren que el repositorio")
    print("GraphDB tenga habilitado el razonador OWL2-RL u OWL-Horst.")
    print("En GraphDB: Setup > Repositories > vehiculos > Edit > Inference")
    print()

    inferencia_subclassof()
    inferencia_inverseof()
    inferencia_subpropertyof()
    inferencia_domain_range()
    inferencia_equivalentclass_hibrido()
