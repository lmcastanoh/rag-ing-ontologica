# backend/ontologia/generar_individuos.py
# ==============================================================================
# Genera individuos OWL reales extrayendo datos de ChromaDB con un LLM.
#
# Flujo:
#   1. Lee todos los chunks de ChromaDB agrupados por (marca, modelo)
#   2. Por cada modelo envía los chunks más relevantes al LLM
#   3. El LLM extrae especificaciones técnicas estructuradas (Pydantic)
#   4. Genera el bloque de individuos en Turtle y lo escribe en vehiculos.ttl
#
# Uso (desde backend/):
#   python ontologia/generar_individuos.py
#
# Prerrequisitos:
#   - ChromaDB ya ingested (python app.py + POST /ingest)
#   - OPENAI_API_KEY en .env
# ==============================================================================

from __future__ import annotations

import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

import chromadb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Buscar .env en backend/ y en la raíz del proyecto
_here = Path(__file__).resolve().parent.parent  # backend/
load_dotenv(_here / ".env")
load_dotenv(_here.parent / ".env")  # raíz del proyecto

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Configuración ──────────────────────────────────────────────────────────────
CHROMA_DIR  = os.getenv("CHROMA_DIR", str(_here / "chroma_db"))
COLLECTION  = os.getenv("CHROMA_COLLECTION", "rag_collection")
LLM_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SCHEMA_FILE  = Path(__file__).parent / "vehiculos.ttl"       # solo schema (no tocar)
OUTPUT_FILE  = Path(__file__).parent / "vehiculos_completo.ttl"  # schema + individuos reales

BASE_IRI = "http://www.semanticweb.org/rag-ontologica/vehiculos#"
PREFIXES = """@prefix :     <http://www.semanticweb.org/rag-ontologica/vehiculos#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
@prefix dc:   <http://purl.org/dc/elements/1.1/> .
"""

# Mapeo marca → URI individual del Fabricante
FABRICANTE_MAP: dict[str, str] = {
    "toyota":    ":Toyota",
    "mazda":     ":Mazda",
    "volkswagen":":Volkswagen",
    "vw":        ":Volkswagen",
    "peugeot":   ":Peugeot",
    "opel":      ":Opel",
    "seat":      ":Seat",
    "mg emotor": ":MGMotor",
    "mg":        ":MGMotor",
}

# Mapeo palabras clave → URI CategoriaVehiculo
CATEGORIA_MAP: dict[str, str] = {
    "suv":        ":SUV",
    "crossover":  ":Crossover",
    "sedan":      ":Sedan",
    "sedán":      ":Sedan",
    "hatchback":  ":Hatchback",
    "berlina":    ":Berlina",
    "compacto":   ":Compacto",
    "pickup":     ":Pickup",
    "furgoneta":  ":Furgoneta",
    "van":        ":Furgoneta",
    "deportivo":  ":Deportivo",
    "coupe":      ":Coupe",
    "coupé":      ":Coupe",
}

# Individuos de seguridad ya declarados en el schema
SEGURIDAD_MAP: dict[str, str] = {
    "abs":                 ":SistABS",
    "antibloqueo":         ":SistABS",
    "esp":                 ":SistESP",
    "estabilidad":         ":SistESP",
    "frenada de emergencia": ":SistFrenEmergencia",
    "aeb":                 ":SistFrenEmergencia",
    "frenado de emergencia": ":SistFrenEmergencia",
    "control de crucero":  ":SistControlCrucero",
    "crucero adaptativo":  ":SistControlCrucero",
    "alerta de colision":  ":SistAlertaColision",
    "alerta colision":     ":SistAlertaColision",
    "colisión frontal":    ":SistAlertaColision",
}


# ==============================================================================
# SCHEMA DE EXTRACCIÓN (Pydantic)
# ==============================================================================
class EspecificacionVehiculo(BaseModel):
    tipo_propulsion: str = Field(
        description="Tipo de propulsión: 'combustion', 'electrico' o 'hibrido'"
    )
    categoria: str = Field(
        description="Categoría del vehículo: SUV, Sedan, Hatchback, Pickup, Crossover, Furgoneta, Deportivo, Berlina, Coupe u Otro"
    )
    peso_kg: Optional[float] = Field(None, description="Peso en vacío en kg (ej: 1516.0). Si hay rango, tomar el menor.")
    longitud_mm: Optional[int] = Field(None, description="Longitud total en mm")
    capacidad_baul_l: Optional[int] = Field(None, description="Capacidad del baúl/maletero en litros")
    potencia_cv: Optional[int] = Field(None, description="Potencia máxima del motor en CV/HP")
    cilindrada_cc: Optional[int] = Field(None, description="Cilindrada del motor en cc. None si es eléctrico puro.")
    consumo_l100km: Optional[float] = Field(None, description="Consumo de combustible en L/100km. None si es eléctrico puro.")
    autonomia_km: Optional[float] = Field(None, description="Autonomía eléctrica en km. Solo para eléctricos/híbridos enchufables.")
    capacidad_bateria_kwh: Optional[float] = Field(None, description="Capacidad de batería en kWh. Solo para eléctricos.")
    tipo_combustible: Optional[str] = Field(None, description="'Gasolina', 'Diesel', 'Electrico' o None si no aplica")
    tipo_transmision: str = Field(description="Descripción breve de la transmisión: 'Manual 6 velocidades', 'Automático CVT', 'Automático 8 velocidades', etc.")
    sistemas_seguridad: list[str] = Field(
        default_factory=list,
        description="Lista de sistemas de seguridad activa mencionados (ABS, ESP, frenada de emergencia, control de crucero, etc.)"
    )
    anyo: Optional[int] = Field(None, description="Año modelo del vehículo")


# ==============================================================================
# FUNCIONES DE APOYO
# ==============================================================================

def _uri_safe(texto: str) -> str:
    """Convierte texto a identificador URI válido (sin puntos ni caracteres especiales)."""
    texto = texto.strip()
    reemplazos = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "ñ": "n", "ü": "u", "Á": "A", "É": "E", "Í": "I",
        "Ó": "O", "Ú": "U", "Ñ": "N",
    }
    for orig, rep in reemplazos.items():
        texto = texto.replace(orig, rep)
    # Separar por cualquier carácter no alfanumérico (incluye punto, coma, etc.)
    partes = re.split(r"[^a-zA-Z0-9]+", texto)
    return "".join(p.capitalize() for p in partes if p)


def _escape_ttl(s: str) -> str:
    """Escapa una cadena para usarla como literal Turtle."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _mapear_fabricante(marca: str) -> str:
    return FABRICANTE_MAP.get(marca.lower(), f":{_uri_safe(marca)}")


def _mapear_categoria(categoria: str) -> str:
    cat_lower = categoria.lower()
    for k, v in CATEGORIA_MAP.items():
        if k in cat_lower:
            return v
    return ":SUV"  # default razonable


def _mapear_seguridad(sistemas: list[str]) -> list[str]:
    """Convierte sistemas de seguridad a URIs conocidos o crea nuevos."""
    uris: list[str] = []
    for sist in sistemas:
        sist_lower = sist.lower()
        encontrado = None
        for k, v in SEGURIDAD_MAP.items():
            if k in sist_lower:
                encontrado = v
                break
        if encontrado and encontrado not in uris:
            uris.append(encontrado)
    # Siempre incluir ABS como mínimo
    if ":SistABS" not in uris:
        uris.append(":SistABS")
    return uris


def _clase_vehiculo(tipo: str) -> str:
    tipo = tipo.lower()
    if "electr" in tipo:
        return ":VehiculoElectrico"
    if "hibri" in tipo or "hybrid" in tipo:
        return ":VehiculoHibrido"
    return ":VehiculoCombustion"


def _clase_motor(tipo: str) -> str:
    tipo = tipo.lower()
    if "electr" in tipo:
        return ":MotorElectrico"
    return ":MotorCombustion"


def _propiedad_motor(tipo: str) -> str:
    tipo = tipo.lower()
    if "electr" in tipo:
        return ":tieneMotorElectrico"
    return ":tieneMotorCombustion"


# ==============================================================================
# EXTRACCIÓN CON LLM
# ==============================================================================

def extraer_specs(marca: str, modelo: str, chunks: list[str], llm) -> EspecificacionVehiculo:
    """Llama al LLM con el texto de los chunks para extraer especificaciones."""

    # Priorizar los últimos chunks (donde están las tablas técnicas)
    # y tomar algunos del inicio para contexto de tipo de vehículo
    n = len(chunks)
    if n <= 8:
        seleccionados = chunks
    else:
        seleccionados = chunks[:2] + chunks[max(0, n-8):]

    texto = "\n---\n".join(seleccionados)
    # Limitar a ~3000 caracteres para no exceder contexto
    if len(texto) > 3000:
        texto = texto[-3000:]

    prompt = f"""Eres un experto en fichas técnicas de vehículos automotores.
A continuación tienes fragmentos de la ficha técnica del vehículo:
  Marca: {marca}
  Modelo: {modelo}

TEXTO DE LA FICHA:
{texto}

Extrae ÚNICAMENTE los valores que aparecen explícitamente en el texto.
Si un valor no aparece, devuelve null/None.
Para el peso, si hay múltiples versiones, toma el valor más bajo.
Para la potencia, extrae el valor en CV o HP (si está en kW, multiplica por 1.36).
"""

    llm_estructurado = llm.with_structured_output(EspecificacionVehiculo)
    try:
        return llm_estructurado.invoke(prompt)
    except Exception as e:
        logger.warning(f"Error LLM para {marca} {modelo}: {e}")
        # Devolver valores mínimos por defecto
        return EspecificacionVehiculo(
            tipo_propulsion="combustion",
            categoria="SUV",
            tipo_transmision="Manual",
            sistemas_seguridad=["ABS"],
            peso_kg=None,
            longitud_mm=None,
            capacidad_baul_l=None,
            potencia_cv=None,
            cilindrada_cc=None,
            consumo_l100km=None,
            autonomia_km=None,
            capacidad_bateria_kwh=None,
            tipo_combustible=None,
            anyo=None,
        )


# ==============================================================================
# GENERACIÓN DE TURTLE
# ==============================================================================

def generar_bloque_motor(
    id_vehiculo: str, specs: EspecificacionVehiculo
) -> tuple[str, Optional[tuple[str, str]], Optional[tuple[str, str]]]:
    """
    Genera el bloque Turtle del motor y retorna (id_motor, bloque_ttl).
    Para híbridos genera dos motores.
    """
    bloques: list[str] = []
    prop_motor_combustion = None
    prop_motor_electrico = None
    id_motor_comb = f":Motor_{id_vehiculo}_Comb"
    id_motor_elec = f":Motor_{id_vehiculo}_Elec"

    tipo = specs.tipo_propulsion.lower()

    # Motor de combustión
    if "combustion" in tipo or "hibri" in tipo or "hybrid" in tipo:
        lineas = [f"{id_motor_comb} a :MotorCombustion ;"]
        if specs.potencia_cv:
            lineas.append(f"    :tienePotenciaCV   {specs.potencia_cv} ;")
        if specs.cilindrada_cc:
            lineas.append(f"    :tieneCilindradaCc {specs.cilindrada_cc} ;")
        combustible = ":Gasolina"
        if specs.tipo_combustible:
            c = specs.tipo_combustible.lower()
            if "diesel" in c or "diésel" in c:
                combustible = ":Diesel"
            elif "gas" in c and "natural" in c:
                combustible = ":GNC"
        lineas.append(f"    :usaCombustible    {combustible} .")
        bloques.append("\n".join(lineas))
        prop_motor_combustion = (":tieneMotorCombustion", id_motor_comb)

    # Motor eléctrico
    if "electr" in tipo or "hibri" in tipo or "hybrid" in tipo:
        lineas = [f"{id_motor_elec} a :MotorElectrico ;"]
        # Para híbridos la potencia eléctrica no siempre se extrae separada
        pot = specs.potencia_cv if "electr" in tipo else None
        if pot:
            lineas.append(f"    :tienePotenciaCV {pot} .")
        else:
            # Quitar el último ";" si no hay potencia
            lineas.append("    rdfs:label \"Motor eléctrico\"@es .")
        bloques.append("\n".join(lineas))
        prop_motor_electrico = (":tieneMotorElectrico", id_motor_elec)

    return "\n\n".join(bloques), prop_motor_combustion, prop_motor_electrico


def generar_individuo_ttl(
    marca: str,
    modelo: str,
    specs: EspecificacionVehiculo,
) -> str:
    """Genera el bloque Turtle completo para un vehículo."""

    id_v = f":{_uri_safe(marca)}_{_uri_safe(modelo)}"
    clase = _clase_vehiculo(specs.tipo_propulsion)
    fabricante = _mapear_fabricante(marca)
    categoria = _mapear_categoria(specs.categoria)
    sistemas = _mapear_seguridad(specs.sistemas_seguridad)

    # Motor
    motor_ttl, prop_comb, prop_elec = generar_bloque_motor(
        f"{_uri_safe(marca)}_{_uri_safe(modelo)}", specs
    )

    # Transmisión: crear un nuevo individuo si la transmisión no encaja en los estándar
    trans_uri = ":Manual6V"  # default
    t = specs.tipo_transmision.lower()
    if "cvt" in t:
        trans_uri = ":AutomaticoCVT"
    elif "dsg" in t or "doble embrague" in t or "dct" in t:
        trans_uri = ":AutomaticoDSG"
    elif "automát" in t or "automat" in t:
        # Crear individuo nuevo para transmisiones automáticas no catalogadas
        trans_id = f":Trans_{_uri_safe(specs.tipo_transmision)}"
        motor_ttl = f"{trans_id} a :Transmision ;\n    rdfs:label \"{_escape_ttl(specs.tipo_transmision)}\"@es .\n\n" + motor_ttl
        trans_uri = trans_id
    elif "electr" in specs.tipo_propulsion.lower():
        trans_uri = ":TransmisionEV"
    elif "5" in t:
        trans_uri = ":Manual5V"

    # Construir propiedades del vehículo
    props = [
        f"{id_v} a {clase} ;",
        f'    :tieneNombreModelo    "{_escape_ttl(modelo)}" ;',
        f'    :tieneMarca           "{_escape_ttl(marca)}" ;',
        f"    :fabricadoPor         {fabricante} ;",
        f"    :perteneceACategoria  {categoria} ;",
        f"    :tieneTransmision     {trans_uri} ;",
    ]

    if specs.anyo:
        props.append(f'    :tieneAnyoLanzamiento "{specs.anyo}"^^xsd:gYear ;')

    if specs.peso_kg:
        props.append(f"    :tienePeso            {specs.peso_kg} ;")

    if specs.longitud_mm:
        props.append(f"    :tieneLongitudMm      {specs.longitud_mm} ;")

    if specs.capacidad_baul_l:
        props.append(f"    :tieneCapacidadBaulL  {specs.capacidad_baul_l} ;")

    # Propiedades de combustión
    if specs.consumo_l100km and "electr" not in specs.tipo_propulsion.lower():
        props.append(f"    :tieneConsumoL100km   {specs.consumo_l100km} ;")

    # Propiedades eléctricas
    if specs.autonomia_km:
        props.append(f"    :tieneAutonomiaKm     {specs.autonomia_km} ;")
    if specs.capacidad_bateria_kwh:
        props.append(f"    :tieneCapacidadBateriaKwh {specs.capacidad_bateria_kwh} ;")

    # Motores
    if prop_comb:
        props.append(f"    {prop_comb[0]}    {prop_comb[1]} ;")
    if prop_elec:
        props.append(f"    {prop_elec[0]}   {prop_elec[1]} ;")

    # Sistemas de seguridad
    if sistemas:
        sist_str = ", ".join(sistemas)
        props.append(f"    :tieneSistemaSeguridad {sist_str} ;")

    # Terminar con punto (reemplazar último ";" por ".")
    props[-1] = props[-1].rstrip(" ;") + " ."

    vehiculo_ttl = "\n".join(props)
    return motor_ttl + "\n\n" + vehiculo_ttl if motor_ttl else vehiculo_ttl


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def leer_chromadb() -> dict[tuple[str, str], list[str]]:
    """Lee ChromaDB y agrupa chunks por (marca, modelo)."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_collection(COLLECTION)
    logger.info(f"ChromaDB: {col.count()} chunks totales")

    all_data = col.get(include=["documents", "metadatas"])  # type: ignore[arg-type]
    docs_raw: list = all_data.get("documents") or []   # type: ignore[assignment]
    metas_raw: list = all_data.get("metadatas") or []  # type: ignore[assignment]
    grupos: dict[tuple[str, str], list[tuple[int, str]]] = {}

    for doc, meta in zip(docs_raw, metas_raw):  # type: ignore[arg-type]
        marca  = str(meta.get("marca", "Desconocida"))
        modelo = str(meta.get("modelo", "Desconocido"))
        page   = int(meta.get("page", 0))
        key: tuple[str, str] = (marca, modelo)
        if key not in grupos:
            grupos[key] = []
        grupos[key].append((page, str(doc)))

    # Ordenar chunks por página para cada modelo
    result: dict[tuple[str, str], list[str]] = {}
    for key, chunks_con_pagina in grupos.items():
        chunks_con_pagina.sort(key=lambda x: x[0])
        result[key] = [c for _, c in chunks_con_pagina]

    logger.info(f"Modelos únicos: {len(result)}")
    return result


def leer_schema_ttl() -> str:
    """Lee el schema base (clases, propiedades, restricciones) desde vehiculos.ttl."""
    if not SCHEMA_FILE.exists() or SCHEMA_FILE.stat().st_size == 0:
        raise FileNotFoundError(
            f"Schema vacío o inexistente: {SCHEMA_FILE}\n"
            "Asegúrate de que vehiculos.ttl tenga el schema OWL (clases y propiedades)."
        )
    return SCHEMA_FILE.read_text(encoding="utf-8")


def separar_schema_de_individuos(ttl: str) -> tuple[str, str]:
    """
    Separa el archivo TTL en dos partes:
      - schema: hasta el marcador de individuos de vehículos/motores generados
      - individuos: la parte generada automáticamente (si existe)
    """
    marcador = "# ====[ INDIVIDUOS GENERADOS AUTOMÁTICAMENTE ]"
    if marcador in ttl:
        idx = ttl.index(marcador)
        return ttl[:idx].rstrip(), ""
    # Si no existe el marcador, conservar todo el schema original
    return ttl.rstrip(), ""


def generar_individuos_fijos() -> str:
    """Genera los individuos de soporte (Fabricantes, Categorías, etc.) que siempre son fijos."""
    return """
# ==============================================================================
# INDIVIDUOS FIJOS — FABRICANTES
# ==============================================================================
:Toyota      a :Fabricante ; rdfs:label "Toyota"@es .
:Mazda       a :Fabricante ; rdfs:label "Mazda"@es .
:Volkswagen  a :Fabricante ; rdfs:label "Volkswagen"@es .
:Peugeot     a :Fabricante ; rdfs:label "Peugeot"@es .
:Opel        a :Fabricante ; rdfs:label "Opel"@es .
:Seat        a :Fabricante ; rdfs:label "SEAT"@es .
:MGMotor     a :Fabricante ; rdfs:label "MG Emotor"@es .

# CATEGORÍAS
:Sedan      a :CategoriaVehiculo ; rdfs:label "Sedán"@es .
:SUV        a :CategoriaVehiculo ; rdfs:label "SUV"@es .
:Hatchback  a :CategoriaVehiculo ; rdfs:label "Hatchback"@es .
:Berlina    a :CategoriaVehiculo ; rdfs:label "Berlina"@es .
:Crossover  a :CategoriaVehiculo ; rdfs:label "Crossover"@es .
:Compacto   a :CategoriaVehiculo ; rdfs:label "Compacto"@es .
:Pickup     a :CategoriaVehiculo ; rdfs:label "Pickup"@es .
:Furgoneta  a :CategoriaVehiculo ; rdfs:label "Furgoneta / Van"@es .
:Deportivo  a :CategoriaVehiculo ; rdfs:label "Deportivo"@es .
:Coupe      a :CategoriaVehiculo ; rdfs:label "Coupé"@es .

# COMBUSTIBLES
:Gasolina   a :TipoCombustible ; rdfs:label "Gasolina"@es .
:Diesel     a :TipoCombustible ; rdfs:label "Diésel"@es .
:GLP        a :TipoCombustible ; rdfs:label "Gas Licuado (GLP)"@es .
:GNC        a :TipoCombustible ; rdfs:label "Gas Natural Comprimido (GNC)"@es .

# TRANSMISIONES
:Manual6V       a :Transmision ; rdfs:label "Manual 6 velocidades"@es .
:AutomaticoCVT  a :Transmision ; rdfs:label "Automático CVT"@es .
:AutomaticoDSG  a :Transmision ; rdfs:label "Automático DSG (doble embrague)"@es .
:Manual5V       a :Transmision ; rdfs:label "Manual 5 velocidades"@es .
:TransmisionEV  a :Transmision ; rdfs:label "Transmisión eléctrica directa"@es .

# SISTEMAS DE SEGURIDAD
:SistABS            a :SistemaSeguridad ; rdfs:label "ABS – Antibloqueo de frenos"@es .
:SistESP            a :SistemaSeguridad ; rdfs:label "ESP – Control de estabilidad"@es .
:SistFrenEmergencia a :SistemaSeguridad ; rdfs:label "Frenada de emergencia (AEB)"@es .
:SistControlCrucero a :SistemaSeguridad ; rdfs:label "Control de crucero adaptativo"@es .
:SistAlertaColision a :SistemaSeguridad ; rdfs:label "Alerta de colisión frontal"@es .
"""


def main():
    logger.info("=== GENERADOR DE INDIVIDUOS OWL DESDE CHROMADB ===")

    # 1. Leer datos reales de ChromaDB
    grupos = leer_chromadb()

    # 2. Inicializar LLM
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    logger.info(f"LLM: {LLM_MODEL}")

    # 3. Leer schema base y separar
    schema_ttl = leer_schema_ttl()
    schema_base, _ = separar_schema_de_individuos(schema_ttl)

    # Quitar la sección de individuos del schema original (si existe)
    # para reemplazarla completamente con datos reales
    marcadores_a_eliminar = [
        "# ==============================================================================\n# INDIVIDUOS — FABRICANTES",
        "# ==============================================================================\n# INDIVIDUOS FIJOS",
    ]
    for marcador in marcadores_a_eliminar:
        if marcador in schema_base:
            idx = schema_base.index(marcador)
            schema_base = schema_base[:idx].rstrip()
            break

    # 4. Generar individuos para cada modelo
    bloques_generados: list[str] = []
    total = len(grupos)

    for i, ((marca, modelo), chunks) in enumerate(grupos.items(), 1):
        logger.info(f"[{i}/{total}] Extrayendo: {marca} — {modelo} ({len(chunks)} chunks)")

        specs = extraer_specs(marca, modelo, chunks, llm)

        bloque = generar_individuo_ttl(marca, modelo, specs)
        bloques_generados.append(
            f"# --- {marca} · {modelo} ---\n{bloque}"
        )

        # Pequeña pausa para no saturar la API
        if i % 10 == 0:
            time.sleep(1)

    # 5. Ensamblar el archivo final
    marcador = "# ====[ INDIVIDUOS GENERADOS AUTOMÁTICAMENTE ]=============================="
    individuos_fijos = generar_individuos_fijos()
    individuos_generados = "\n\n".join(bloques_generados)

    ttl_final = "\n\n".join([
        schema_base,
        marcador,
        "# Generado automáticamente desde ChromaDB. No editar manualmente.",
        individuos_fijos,
        "# ==============================================================================",
        "# VEHÍCULOS Y MOTORES (datos reales de los PDFs)",
        "# ==============================================================================",
        individuos_generados,
    ])

    # 6. Escribir archivo
    OUTPUT_FILE.write_text(ttl_final, encoding="utf-8")
    logger.info(f"\n✅ Ontología actualizada: {OUTPUT_FILE}")
    logger.info(f"   Modelos procesados: {total}")
    logger.info(f"   Tamaño del archivo: {OUTPUT_FILE.stat().st_size // 1024} KB")
    logger.info("\nPróximo paso: reimportar vehiculos.ttl en GraphDB")
    logger.info("  GraphDB → Import → Upload RDF file → vehiculos.ttl")
    logger.info("  Marcar: 'Clear repository before import' para reemplazar datos anteriores")


if __name__ == "__main__":
    # Asegurarse de que el working directory sea backend/
    script_dir = Path(__file__).parent.parent  # backend/
    os.chdir(script_dir)
    main()
