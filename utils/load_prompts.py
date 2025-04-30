# from datos_sinteticos import guia_graduacion_privado_2025, leyes_reglas_usac, normativo_eps, normativo_general_evaluacion_promocion, normativo_practicas_finales,pensum_civil,pensum_electrica,pensum_electronica,pensum_mecanica,pensum_quimica, pensum_industrial, pensum_mecanica_industrial,pensum_sistemas,preguntas_frecuentes
from datos_sinteticos.guia_graduacion_privado_2025 import guia_graduacion_y_privado_2025 
from documentos_procesados.guia_graduacion_privado_2025 import guia_graduacion_y_privado_2025_context

data_list = [
    "guia_graduacion_privado_2025",
    "leyes_reglas_usac",
    "normativo_eps",
    "normativo_general_evaluacion_promocion",
    "normativo_practicas_finales",
    "pensum_civil",
    "pensum_electrica",
    "pensum_electronica",
    "pensum_mecanica",
    "pensum_quimica",
    "pensum_industrial",
    "pensum_mecanica_industrial",
    "pensum_sistemas",
    "preguntas_frecuentes"
]

def load_prompts_with_context(kind:str ): 
    if kind == "guia_graduacion_privado_2025":
        return {
            "questions":  guia_graduacion_y_privado_2025(),
            "context": guia_graduacion_y_privado_2025_context()
        }