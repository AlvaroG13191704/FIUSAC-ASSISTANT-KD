from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# --- Configuración ---
MODEL_DIR = "alvarog1318/KD_USAC_ASSISTANT" # Directorio donde guardaste tu modelo afinado
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Usa GPU si está disponible

# --- Cargar Tokenizer y Modelo ---
print(f"Cargando tokenizer y modelo desde: {MODEL_DIR}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16, # Usa el mismo dtype que en el entrenamiento si es posible
        attn_implementation='eager', # Puedes añadir esto si es necesario, pero para inferencia a veces sdpa es mejor
        # device_map="auto" # O especifica el dispositivo: device_map=DEVICE
    )
    # model.to(DEVICE) # Asegúrate de que el modelo esté en el dispositivo correcto si no usas device_map="auto"
    print("Modelo y tokenizer cargados.")
except Exception as e:
    print(f"Error al cargar el modelo o el tokenizer: {e}")
    exit()


text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE == "cuda" else -1 # device=0 para la primera GPU, -1 para CPU
)

# --- Función para Formatear el Prompt para Gemma ---
SYSTEM_PROMPT = (
    "Has sido entrenado exhaustivamente como un experto en los procedimientos, normativas y malla curricular de las escuelas de ingeniería civil, electrica, electronica, industrial, y sistemas de la Facultad de Ingeniería. Tu propósito principal es asistir con consultas relacionadas con esta información específica.\n\n"
    "Considera lo siguiente al responder:\n"
    "-   **Tu base de conocimiento es la información oficial de la Facultad de Ingeniería utilizada durante tu entrenamiento.** Prioriza esta fuente sobre cualquier otro conocimiento general.\n"
    "-   Si una pregunta se desvía de los temas de la Facultad de Ingeniería o la respuesta no se encuentra en tu base de datos de entrenamiento, es mejor indicar que no posees esa información específica.\n"
    "-   Proporciona respuestas directas, informativas y fáciles de entender para los estudiantes y personal de la facultad.\n"
    "-   Tu objetivo es ser un recurso confiable sobre la Facultad de Ingeniería."
)

# --- Función para Formatear el Prompt para Gemma ---
def format_prompt_gemma(pregunta: str) -> str:
    return (
        f"<start_of_turn>system\n{SYSTEM_PROMPT}<end_of_turn>\n"
        f"<start_of_turn>user\n{pregunta}<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


# --- Bucle de Prueba ---
print("\n--- Prueba del Modelo Afinad ---")
print("Escribe tu pregunta (o 'salir' para terminar):")

while True:
    user_input = input("Pregunta: ")
    if user_input.lower() == 'salir':
        break
    if not user_input.strip():
        continue

    formatted_prompt = format_prompt_gemma(user_input)

    print("Generando respuesta...")
    try:
        outputs = text_generator(
            formatted_prompt,
            max_new_tokens=150,  # Número máximo de tokens nuevos a generar
            do_sample=True,      # Activa el muestreo para respuestas más variadas
            temperature=0.7,     # Controla la aleatoriedad (más bajo = más determinista)
            top_k=50,            # Considera solo los k tokens más probables
            top_p=0.95,          # Muestreo por nucleus
            pad_token_id=tokenizer.eos_token_id # Importante para evitar advertencias y generar hasta EOS
        )
        
        generated_text = outputs[0]['generated_text']
        
        # Extraer solo la respuesta del modelo (después de "<start_of_turn>model\n")
        model_response_start_tag = "<start_of_turn>model\n"
        response_start_index = generated_text.rfind(model_response_start_tag) # rfind para encontrar la última ocurrencia

        if response_start_index != -1:
            model_answer = generated_text[response_start_index + len(model_response_start_tag):].strip()
        else:
            # Si el tag no se encuentra (raro si el prompt está bien formateado), mostrar todo
            model_answer = generated_text 

        print(f"\nRespuesta del Modelo:\n{model_answer}\n")
        print("-" * 50)

    except Exception as e:
        print(f"Error durante la generación: {e}")

print("Prueba finalizada.")