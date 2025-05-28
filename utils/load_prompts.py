# prompts
from datos_sinteticos.guia_graduacion_privado_2025 import guia_graduacion_y_privado_2025
from datos_sinteticos.normativo_eps import normativo_eps
from datos_sinteticos.normativo_general_evaluacion_promocion import normativo_general_evaluacion_promocion
from datos_sinteticos.normativo_practicas_finales import normativo_practicas_finales 
from datos_sinteticos.preguntas_frecuentes import preguntas_frecuentes
# contexts
from documentos_procesados.guia_graduacion_privado_2025 import guia_graduacion_y_privado_2025_context
from documentos_procesados.normativo_eps import normativo_eps_context
from documentos_procesados.normativo_general_evaluacion_promocion import normativo_general_evaluacion_promocion_context
from documentos_procesados.normativo_practicas_finales import normativo_practicas_finales_context
from documentos_procesados.preguntas_frecuentes import preguntas_frecuentes_context

# pensums prompts
from datos_sinteticos.pensum_civil import pensum_civil
from datos_sinteticos.pensum_electrica import pensum_electrica
from datos_sinteticos.pensum_electronica import pensum_electronica
from datos_sinteticos.pensum_industrial import pensum_industrial
from datos_sinteticos.pensum_mecanica_industrial import pensum_mecanica_industrial
from datos_sinteticos.pensum_mecanica import pensum_mecanica
from datos_sinteticos.pensum_quimica import pensum_quimica
from datos_sinteticos.pensum_sistemas import pensum_sistemas
# pensum context
from documentos_procesados.pensum_civil import pensum_civil_context
from documentos_procesados.pensum_electrica import pensum_electrica_context
from documentos_procesados.pensum_electronica import pensum_electronica_context
from documentos_procesados.pensum_industrial import pensum_industrial_context
from documentos_procesados.pensum_mecanica_industrial import pensum_mecanica_industrial_context
from documentos_procesados.pensum_mecanica import pensum_mecanica_context
from documentos_procesados.pensum_quimica import pensum_quimica_context
from documentos_procesados.pensum_sistemas import pensum_sistemas_context


def load_prompts_with_context(kind:str ): 
    if kind == "guia_graduacion_privado_2025":
        return {
            "questions":  guia_graduacion_y_privado_2025(),
            "context": guia_graduacion_y_privado_2025_context()
        }
    if kind == "normativo_eps":
        return {
            "questions": normativo_eps(),
            "context": normativo_eps_context()
        }
    if kind == "normativo_general_evaluacion_promocion":
        return {
            "questions": normativo_general_evaluacion_promocion(),
            "context": normativo_general_evaluacion_promocion_context()
        }
    if kind == "normativo_practicas_finales":
        return {
            "questions": normativo_practicas_finales(),
            "context": normativo_practicas_finales_context()
        }
    if kind == "preguntas_frecuentes":
        return {
            "questions": preguntas_frecuentes(),
            "context": preguntas_frecuentes_context()
        }
    if kind == "pensum_civil":
        return {
            "questions": pensum_civil(),
            "context": pensum_civil_context()
        }
    if kind == "pensum_electrica":
        return {
            "questions": pensum_electrica(),
            "context": pensum_electrica_context()
        }
    if kind == "pensum_electronica":
        return {
            "questions": pensum_electronica(),
            "context": pensum_electronica_context()
        }
    if kind == "pensum_industrial":
        return {
            "questions": pensum_industrial(),
            "context": pensum_industrial_context()
        }
    if kind == "pensum_mecanica_industrial":
        return {
            "questions": pensum_mecanica_industrial(),
            "context": pensum_mecanica_industrial_context()
        }
    if kind == "pensum_mecanica":
        return {
            "questions": pensum_mecanica(),
            "context": pensum_mecanica_context()
        }
    if kind == "pensum_quimica":
        return {
            "questions": pensum_quimica(),
            "context": pensum_quimica_context()
        }
    if kind == "pensum_sistemas":
        return {
            "questions": pensum_sistemas(),
            "context": pensum_sistemas_context()
        }
    else:
        raise ValueError(f"Unknown kind: {kind}.")