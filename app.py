# import docs
from utils.docs import getDocument
from utils.prompts import returnPrompt
# libraries
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import json
import os

# --- Configuración (igual que antes) ---
model_name = "meta-llama/Llama-3.1-8B-Instruct" # "meta-llama/Llama-3.1-8B-Instruct"
# ... (resto de configuración de modelo, hardware, RAG, directorios) ...
quantization_config = None
torch_dtype = torch.bfloat16
embedder_name = "sentence-transformers/all-mpnet-base-v2"
n_docs_to_retrieve = 5 # <<< Renombrado para claridad: cuántos chunks recuperar >>>
tokenizer_max_length = 4096
max_new_tokens = 100
output_dir = "kd_data_llama3_3B" # <<< Nombre de directorio cambiado
os.makedirs(output_dir, exist_ok=True)
output_jsonl_path = os.path.join(output_dir, "metadata.jsonl")
logits_dir = os.path.join(output_dir, "logits")
os.makedirs(logits_dir, exist_ok=True)


# --- Carga de Modelos y Datos ---

# Modelo de embeddings
print(f"Cargando embedder: {embedder_name}")
embedder = SentenceTransformer(embedder_name)

# # <<< INICIO: Modificación - Procesamiento de Documentos con Chunking >>>
# print("Procesando documentos con chunking...")
# original_docs = chunk_by_list(getDocument()) # Obtiene la lista de documentos (en tu caso, parece ser solo uno)
# if not original_docs:
#      raise ValueError("returnDocument() no devolvió ningún documento.")
# if not all(isinstance(doc, str) for doc in original_docs):
#     raise TypeError("returnDocument() debe devolver una lista de strings.")

# all_chunks = []
# for i, doc_text in enumerate(original_docs):
#     doc_chunks = chunk_document_by_article(doc_text)
#     print(f"  Documento {i+1}: Encontrados {len(doc_chunks)} chunks.")
#     all_chunks.extend(doc_chunks)

# if not all_chunks:
#      raise ValueError("No se pudieron extraer chunks de los documentos.")

# print(f"Total de chunks creados: {len(all_chunks)}")
# # 'all_chunks' es ahora la lista de textos que vamos a embeber e indexar.
# # <<< FIN: Modificación - Procesamiento de Documentos con Chunking >>>


# <<< INICIO: Modificación - Crear Embeddings y FAISS para Chunks >>>
print("Codificando chunks y creando índice FAISS...")
all_chunks = getDocument()
# Ahora codificamos los chunks, no los documentos completos
embs = embedder.encode(all_chunks, convert_to_numpy=True, show_progress_bar=True)

# Check if embeddings were created successfully
if embs is None or len(embs) == 0:
    raise ValueError("La codificación de los chunks no produjo embeddings.")
if len(embs) != len(all_chunks):
     raise ValueError(f"Discrepancia: {len(all_chunks)} chunks, pero {len(embs)} embeddings creados.")

index = faiss.IndexFlatIP(embs.shape[1])
index.add(embs)
print("Índice FAISS creado basado en chunks.")
# <<< FIN: Modificación - Crear Embeddings y FAISS para Chunks >>>

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
)
model.eval()
print("Modelo LLM cargado.")

# --- Pipeline RAG y Generación de Datos KD ---

print(f"Iniciando generación de datos KD. Guardando en: {output_dir}")
with open(output_jsonl_path, "w", encoding="utf-8") as f_out:
    data_id_counter = 0
    for prompt_info in returnPrompt():
        data_id_counter += 1
        current_data_id = f"data_{data_id_counter:05d}"

        # <<< INICIO: Modificación - Recuperar Chunks Relevantes >>>
        # 1. Recuperar Contexto (ahora recuperamos chunks)
        q_embs = embedder.encode([prompt_info["pregunta"]], convert_to_numpy=True)
        # Buscamos sobre el índice de chunks
        scores, I = index.search(q_embs, k=n_docs_to_retrieve)

        retrieved_chunk_indices = I[0]

        # Deduplicar índices de chunks (aunque menos probable, es buena práctica)
        unique_chunk_indices = list(dict.fromkeys(retrieved_chunk_indices))

        # Construir contexto usando el texto de los chunks únicos recuperados
        # ¡Importante! Usamos 'all_chunks' aquí
        retrieved_context = "\n\n---\n\n".join([all_chunks[i] for i in unique_chunk_indices]) # Separador opcional
        # <<< FIN: Modificación - Recuperar Chunks Relevantes >>>


        # 2. Construir Prompt para el Profesor
        system_message = f"""Eres un asistente AI experto y preciso. Responde la pregunta basándote estrictamente en el siguiente contexto (separado por '---'). Si la respuesta no está en el contexto, dilo explícitamente. Sé conciso.
Contexto:
{retrieved_context}""" # Ahora 'retrieved_context' contiene chunks relevantes
        user_message = prompt_info["pregunta"]

        full_prompt_for_teacher = (
             f"<|begin_of_text|>"
             f"<|start_header_id|>system<|end_header_id|>\n\n"
             f"{system_message}<|eot_id|>"
             f"<|start_header_id|>user<|end_header_id|>\n\n"
             f"{user_message}<|eot_id|>"
             f"<|start_header_id|>assistant<|end_header_id|>\n\n"
         )

        # 3. Tokenizar (sin cambios)
        inputs = tokenizer(
            full_prompt_for_teacher,
            return_tensors="pt",
            padding=False,
            truncation=True,
            # max_length=tokenizer_max_length,
        ).to(model.device)
        # ... (resto del código de tokenización) ...
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # 4. Generar Respuesta y Logits (sin cambios en la llamada a generate)
        with torch.no_grad():
            outputs = model.generate(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 max_new_tokens=max_new_tokens,
                 do_sample=False,
                 output_scores=True,
                 return_dict_in_generate=True,
                 pad_token_id=tokenizer.pad_token_id,
                 eos_token_id=tokenizer.eos_token_id
             )

        # 5. Procesar Salida para KD (sin cambios lógicos aquí)
        generated_ids_all = outputs.sequences[0]
        prompt_length = input_ids.shape[-1]
        target_token_ids = generated_ids_all[prompt_length:]
        teacher_logits_list = outputs.scores
        # ... (resto del código de verificación de longitud y apilado de logits) ...
        num_generated_tokens = len(target_token_ids)
        num_logit_steps = len(teacher_logits_list)

        if num_generated_tokens == 0:
             print(f"Advertencia ID {current_data_id}: No se generaron tokens para la pregunta: {prompt_info['pregunta']}")
             continue

        if num_logit_steps != num_generated_tokens:
            print(f"Advertencia ID {current_data_id}: Discrepancia de longitud - Logits: {num_logit_steps}, Tokens: {num_generated_tokens}. Se truncará a la longitud menor.")
            min_len = min(num_logit_steps, num_generated_tokens)
            target_token_ids = target_token_ids[:min_len]
            teacher_logits_list = teacher_logits_list[:min_len]
            if min_len == 0:
                 print(f"Advertencia ID {current_data_id}: Longitud mínima 0 tras truncar.")
                 continue

        teacher_logits_tensor = torch.stack(teacher_logits_list).squeeze(1).to(device='cpu', dtype=torch.float32)


        # 6. Guardar Datos para KD (sin cambios lógicos aquí, pero el contexto guardado será el de los chunks)
        logits_filename = f"{current_data_id}_logits.pt"
        logits_filepath = os.path.join(logits_dir, logits_filename)
        torch.save(teacher_logits_tensor, logits_filepath)

        metadata = {
            "id": current_data_id,
            "pregunta": prompt_info["pregunta"],
            "respuesta_esperada_original": prompt_info.get("respuesta", "N/A"),
            # <<< CAMBIO: Guardamos el contexto basado en chunks >>>
            # "contexto_recuperado_chunks": retrieved_context,
            "respuesta_generada_profesor_ids": target_token_ids.tolist(),
            "respuesta_generada_profesor_texto": tokenizer.decode(target_token_ids, skip_special_tokens=True),
            "logits_profesor_path": logits_filepath
        }

        f_out.write(json.dumps(metadata, ensure_ascii=False) + "\n")

        if data_id_counter % 10 == 0:
            print(f"Procesados {data_id_counter} ejemplos...")
            # <<< DEBUG Opcional: Imprime la respuesta generada para ver si mejora >>>
            # print(f"  Pregunta: {prompt_info['pregunta']}")
            # print(f"  Contexto Chunks: {retrieved_context[:500]}...") # Imprime inicio del contexto
            # print(f"  Respuesta Generada: {metadata['respuesta_generada_profesor_texto']}")


print(f"Proceso completado. Metadatos guardados en: {output_jsonl_path}")
print(f"Tensores de logits guardados en: {logits_dir}")