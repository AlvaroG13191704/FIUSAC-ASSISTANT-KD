# import docs
# from utils.docs import get_document_text_for_prompt # <--- Necesitarás una función que mapee pregunta a texto de doc
# from utils.prompts import returnPrompt
from utils.load_prompts import load_prompts_with_context
# libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os
import warnings # Para advertir sobre longitudes de contexto

# --- Configuración ---
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
model_name = "meta-llama/Llama-3.1-8B-Instruct"
quantization_config = None
torch_dtype = torch.bfloat16
tokenizer_max_length = 128000 # <-- Aumentar si es necesario, pero cuidado con VRAM/tiempo
max_new_tokens = 150 # Quizás aumentar un poco si las respuestas necesitan ser más largas
output_dir = "kd_data_llama3_8B_full_context"
os.makedirs(output_dir, exist_ok=True)
output_jsonl_path = os.path.join(output_dir, "metadata.jsonl")
logits_dir = os.path.join(output_dir, "logits")
os.makedirs(logits_dir, exist_ok=True)
CONTEXT_LENGTH_WARNING_THRESHOLD = 100000 # Umbral para advertir sobre prompts muy largos (en tokens)

# Tokenizer y Modelo LLM Profesor
print(f"Cargando tokenizer y modelo: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch_dtype,
    quantization_config=quantization_config,
    device_map="auto"
    # attn_implementation="flash_attention_2" # Opcional: usar Flash Attention 2 si está instalado y GPU compatible (A100, H100, etc.) para acelerar contextos largos
)
model.eval()
print("Modelo LLM cargado.")

# --- Pipeline de Generación de Datos KD (Sin RAG) ---

print(f"Iniciando generación de datos KD con contexto completo. Guardando en: {output_dir}")
with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
    data_id_counter = 0
    
    questions_w_context = load_prompts_with_context("guia_graduacion_privado_2025")
    
    for prompt_info in questions_w_context["questions"]:
        data_id_counter += 1
        current_data_id = f"data_{data_id_counter:05d}"

        # 1. Obtener Pregunta y Contexto del Documento Completo
        pregunta = prompt_info.get("pregunta")
        if not pregunta:
            print(f"Advertencia ID {current_data_id}: Falta 'pregunta' en prompt_info. Saltando.")
            continue

        try:
            full_document_text = questions_w_context["context"]
            if not isinstance(full_document_text, str) or not full_document_text:
                 raise ValueError("Texto de documento inválido o vacío.")
        except Exception as e:
            print(f"Error ID {current_data_id}: No se pudo obtener/cargar el texto del documento para la pregunta '{pregunta[:50]}...': {e}. Saltando.")
            continue

        # 2. Construir Prompt para el Profesor
        #    Incorpora Fundamentación, Tono Experto (sin citas directas), Precisión, Manejo de Ausencia.
        system_message = f"""Eres un asistente AI experto en normativas y procedimientos de la Facultad de Ingeniería. Tu tarea es responder la pregunta del usuario basándote ÚNICA Y EXCLUSIVAMENTE en la información contenida en el siguiente documento.

        **Instrucciones Cruciales:**
        1.  **Fundamentación Absoluta:** Deriva tu respuesta estrictamente de la información explícita en el documento. No supongas, no inventes, no uses conocimiento externo.
        2.  **Tono de Experto Informado:** Responde de forma natural y directa, como si conocieras perfectamente el contenido del documento. EVITA frases como "Según el artículo X...", "El documento establece que...", "En la sección Y dice...". Simplemente presenta la información relevante directamente.
        3.  **Precisión y Concisión:** Ofrece respuestas claras, precisas y fáciles de entender para un estudiante. Cíñete a lo esencial para responder la pregunta.
        4.  **Manejo de Información Ausente:** Si la respuesta a la pregunta NO se encuentra de forma explícita en el documento proporcionado, indícalo claramente diciendo algo como "La información solicitada no se encuentra en el documento proporcionado." o "El documento proporcionado no contiene información sobre eso." NO intentes responder de otra manera.

        **Contexto del Documento Completo:**
        --- INICIO DOCUMENTO ---
        {full_document_text}
        --- FIN DOCUMENTO ---"""

        user_message = pregunta # La pregunta del estudiante

        # El formato Llama 3 Instruct que ya usas es CORRECTO:
        full_prompt_for_teacher = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_message}<|eot_id|>" # Fin del bloque system
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_message}<|eot_id|>" # Fin del bloque user
            f"<|start_header_id|>assistant<|end_header_id|>\n\n" # Inicio del bloque assistant (el modelo completa desde aquí)
        )
        # 3. Tokenizar
        inputs = tokenizer(
            full_prompt_for_teacher,
            return_tensors="pt",
            padding=False, # No padding necesario para una sola secuencia
            truncation=True, # Truncará si excede tokenizer_max_length
            max_length=tokenizer_max_length, # Asegura que no exceda el límite teórico
        ).to(model.device)

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_length_tokens = input_ids.shape[-1]

        # *** Advertencia de Longitud de Contexto ***
        if prompt_length_tokens >= CONTEXT_LENGTH_WARNING_THRESHOLD:
             warnings.warn(
                 f"ID {current_data_id}: El prompt tokenizado tiene {prompt_length_tokens} tokens, "
                 f"acercándose o excediendo el umbral de {CONTEXT_LENGTH_WARNING_THRESHOLD}. "
                 f"La generación puede ser lenta, consumir mucha VRAM o haber sido truncada.",
                 UserWarning
             )
        # Comprobar si hubo truncamiento real (si la longitud es exactamente max_length)
        if tokenizer.truncation_side == 'right' and prompt_length_tokens == tokenizer_max_length:
             warnings.warn(
                 f"ID {current_data_id}: ¡El prompt fue truncado a {tokenizer_max_length} tokens! "
                 f"El final del documento fue cortado.", UserWarning)
        elif tokenizer.truncation_side == 'left' and prompt_length_tokens == tokenizer_max_length:
             warnings.warn(
                 f"ID {current_data_id}: ¡El prompt fue truncado a {tokenizer_max_length} tokens! "
                 f"El inicio del documento/prompt fue cortado.", UserWarning)


        # 4. Generar Respuesta y Logits (Sin cambios en la llamada)
        try:
            with torch.no_grad():
                outputs = model.generate(
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     max_new_tokens=max_new_tokens,
                     do_sample=False, # Para consistencia en KD, mejor no usar sampling aquí
                     output_scores=True,
                     return_dict_in_generate=True,
                     pad_token_id=tokenizer.pad_token_id,
                     eos_token_id=tokenizer.eos_token_id # Usar ambos EOS definidos por Llama3.1
                 )
        except Exception as e:
             print(f"Error ID {current_data_id}: Falló la generación para la pregunta '{pregunta[:50]}...': {e}")
             # Podría ser un error Out-of-Memory (OOM) por contexto largo
             continue # Saltar al siguiente prompt


        # 5. Procesar Salida para KD (Sin cambios lógicos)
        generated_ids_all = outputs.sequences[0]
        prompt_length = input_ids.shape[-1] # Longitud real del prompt tokenizado
        target_token_ids = generated_ids_all[prompt_length:]
        teacher_logits_list = outputs.scores
        
        num_generated_tokens = len(target_token_ids)
        num_logit_steps = len(teacher_logits_list)

        if num_generated_tokens == 0:
             print(f"Advertencia ID {current_data_id}: No se generaron tokens para la pregunta: {pregunta}")
             continue # Saltar si no se generó nada

        # Alinear logits y tokens si hay discrepancia (esto puede ocurrir por varias razones)
        min_len = min(num_logit_steps, num_generated_tokens)
        if min_len == 0:
            print(f"Advertencia ID {current_data_id}: Longitud mínima 0 tras alinear logits/tokens. Saltando.")
            continue
        if num_logit_steps != num_generated_tokens:
            target_token_ids = target_token_ids[:min_len]
            teacher_logits_list = teacher_logits_list[:min_len]

        # Apilar logits
        try:
             teacher_logits_tensor = torch.stack(teacher_logits_list).squeeze(1).to(device='cpu', dtype=torch.float32)
        except RuntimeError as e:
             print(f"Error ID {current_data_id}: Error al apilar logits (posiblemente lista vacía o tamaños inconsistentes): {e}. Saltando.")
             continue


        # 6. Guardar Datos para KD
        logits_filename = f"{current_data_id}_logits.pt"
        logits_filepath = os.path.join(logits_dir, logits_filename)
        torch.save(teacher_logits_tensor, logits_filepath)

        metadata = {
            "id": current_data_id,
            "pregunta": pregunta,
            "respuesta_esperada_original": prompt_info.get("respuesta", "N/A"), # La respuesta original si la tienes
            "prompt_length_tokens": prompt_length_tokens, # Útil para análisis
            "respuesta_generada_profesor_ids": target_token_ids.tolist(),
            "respuesta_generada_profesor_texto": tokenizer.decode(target_token_ids, skip_special_tokens=True),
            "logits_profesor_path": logits_filepath
        }

        f_out.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        if data_id_counter % 5 == 0: # Imprimir más frecuentemente si es lento
            print(f"Procesados {data_id_counter} ejemplos...")


print(f"Proceso completado. Metadatos guardados en: {output_jsonl_path}")
print(f"Tensores de logits guardados en: {logits_dir}")