import os
import json
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from torch import nn
from typing import Dict, Any, List

# --- 1. Configuración ---
STUDENT_MODEL_NAME = "google/gemma-3-1b-it"
TEACHER_LOGITS_DIR = "kd_data_gemma3_12b_it/logits" # Asegúrate que esta ruta sea correcta
METADATA_PATH = "kd_data_gemma3_12b_it/metadata.jsonl" # Asegúrate que esta ruta sea correcta

OUTPUT_DIR = f"{STUDENT_MODEL_NAME.split('/')[-1]}-student-finetuned-kd"
CACHE_DIR = "./cache_student_gemma_3" 

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Hiperparámetros (ajusta según experimentación)
TRAIN_BATCH_SIZE = 1 # O 2, 4 si VRAM lo permite
EVAL_BATCH_SIZE = 2  # O 4
# Batch efectivo = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# Intenta mantener un batch efectivo entre 8 y 32 (o más si es necesario y posible)
GRADIENT_ACCUMULATION_STEPS = 16 // TRAIN_BATCH_SIZE # Para un batch efectivo de 16
LEARNING_RATE = 2e-5 # Punto de partida, podría ser 1e-5 o 3e-5
NUM_EPOCHS = 5       # Aumentado un poco, monitoriza eval_loss
LOGGING_STEPS = 20   # Log más frecuente para ver el progreso
SAVE_STEPS = 200     # Guarda checkpoints
EVAL_STEPS = 200     # Evalúa con la misma frecuencia que guardas
MAX_SEQ_LENGTH_STUDENT = 1024 

# Parámetros de Destilación
KD_ALPHA = 0.5       # Valor común, puedes probar 0.3, 0.7
KD_TEMPERATURE = 2.0 # Común, puedes probar 1.5, 2.5, 3.0

# --- 2. Carga de Modelo y Tokenizer Estudiante ---
print(f"Cargando modelo estudiante: {STUDENT_MODEL_NAME}")
quantization_config = None # Opcional: BitsAndBytesConfig si necesitas ahorrar VRAM para el estudiante
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME, cache_dir=CACHE_DIR)
student_model = AutoModelForCausalLM.from_pretrained(
    STUDENT_MODEL_NAME,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
    attn_implementation='eager' # Recomendado para Gemma 3
)
if student_tokenizer.pad_token is None:
    print("Añadiendo pad_token (eos_token) al tokenizador del estudiante.")
    student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

# Para Gemma, el vocab_size está directamente en config.
try:
    STUDENT_MODEL_CONFIG_VOCAB_SIZE = student_model.config.vocab_size
    print(f"INFO: Vocabulario del MODELO ESTUDIANTE (config.vocab_size): {STUDENT_MODEL_CONFIG_VOCAB_SIZE}")
except AttributeError:
    print(f"ADVERTENCIA: No se pudo obtener student_model.config.vocab_size. Usando len(student_tokenizer) como fallback.")
    STUDENT_MODEL_CONFIG_VOCAB_SIZE = len(student_tokenizer) # Fallback, pero lo ideal es el de la config

# Los logits del profesor TIENEN 262208 según tu inspección
TEACHER_LOGITS_ACTUAL_VOCAB_SIZE = 262208 # De tu inspección de archivos .pt
print(f"INFO: Vocabulario REAL de los LOGITS DEL PROFESOR (de archivos .pt): {TEACHER_LOGITS_ACTUAL_VOCAB_SIZE}")

# --- 3. Carga y Preprocesamiento de Datos ---
# (Tu código de carga de datos parece bueno, lo mantendré igual)
print("Cargando metadatos...")
all_data = []
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        all_data.append(json.loads(line))
initial_count = len(all_data)
processed_data = [] # ... tu lógica de filtrado ...
for example in all_data: # tu lógica de filtrado
    if not example.get("respuesta_generada_profesor_texto"): continue
    if not os.path.exists(example.get("logits_profesor_path", "")): continue
    processed_data.append(example)
print(f"Filtrados {initial_count - len(processed_data)} ejemplos. Usando {len(processed_data)} ejemplos.")
if not processed_data: raise ValueError("No hay datos válidos.")

raw_dataset = Dataset.from_list(processed_data)
dataset_dict = raw_dataset.train_test_split(test_size=0.05, seed=42) # Mantener un 5-10% para evaluación
train_dataset = dataset_dict['train']
eval_dataset = dataset_dict['test']

print(f"Dataset cargado. Entrenamiento: {len(train_dataset)}, Evaluación: {len(eval_dataset)}")

def preprocess_function(examples: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
    batch_input_ids = []
    batch_labels = []
    batch_teacher_logits = []
    batch_attention_mask = []
    
    for i in range(len(examples['pregunta'])):
        pregunta = examples['pregunta'][i]
        respuesta_profesor_texto = examples['respuesta_generada_profesor_texto'][i]
        logits_path = examples['logits_profesor_path'][i]
        
        prompt_part = f"<start_of_turn>user\n{pregunta}<end_of_turn>\n<start_of_turn>model\n"
        # Importante: Añadir EOS al final de la respuesta para que el modelo aprenda a parar.
        full_text = prompt_part + respuesta_profesor_texto + student_tokenizer.eos_token
        
        tokenized_full = student_tokenizer(full_text, max_length=MAX_SEQ_LENGTH_STUDENT, truncation=True, padding=False)
        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']
        
        # Tokenizar solo el prompt para enmascarar
        # NO añadir special tokens aquí si ya están en prompt_part y no quieres duplicarlos
        tokenized_prompt = student_tokenizer(prompt_part, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LENGTH_STUDENT)
        prompt_len = len(tokenized_prompt['input_ids'])
        
        labels = list(input_ids)
        labels[:prompt_len] = [-100] * prompt_len
        
        try:
            teacher_logits_tensor_original = torch.load(logits_path, map_location='cpu').to(torch.float32)
            if not isinstance(teacher_logits_tensor_original, torch.Tensor) or teacher_logits_tensor_original.nelement() == 0:
                 print(f"Logits inválidos en {logits_path}. Saltando.")
                 continue
        except Exception as e:
            print(f"Error cargando logits {logits_path}: {e}. Saltando.")
            continue
        
        student_response_token_ids_count = sum(1 for l_val in labels[prompt_len:] if l_val != -100)
        num_teacher_logits_steps = teacher_logits_tensor_original.shape[0]

        if student_response_token_ids_count == 0 or num_teacher_logits_steps == 0:
            print(f"ID {examples['id'][i]}: Tokens de respuesta o logits del profesor con longitud 0. Saltando.")
            continue
            
        min_response_len = min(student_response_token_ids_count, num_teacher_logits_steps)
        if min_response_len <= 0:
            print(f"ID {examples['id'][i]}: Longitud de respuesta 0 tras alineación. Saltando.")
            continue
            
        aligned_teacher_logits = teacher_logits_tensor_original[:min_response_len, :]
   
        
        final_seq_len = prompt_len + min_response_len
        current_input_ids = input_ids[:final_seq_len]
        current_attention_mask = attention_mask[:final_seq_len]
        current_labels = labels[:final_seq_len]
        
        batch_input_ids.append(current_input_ids)
        batch_attention_mask.append(current_attention_mask)
        batch_labels.append(current_labels)
        batch_teacher_logits.append(aligned_teacher_logits)
    
    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention_mask,
        "labels": batch_labels,
        "teacher_logits_list_of_lists": batch_teacher_logits, # `datasets` lo convierte a lista de listas
    }
    
print("Preprocesando datasets...")
BATCH_SIZE_FOR_MAP = 1
tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE_FOR_MAP, remove_columns=train_dataset.column_names, num_proc=1)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True, batch_size=BATCH_SIZE_FOR_MAP, remove_columns=eval_dataset.column_names, num_proc=1)
print("Preprocesamiento completado.")

TEACHER_VOCAB_SIZE = 262144

# --- 4. Data Collator Personalizado (tu código actual es robusto para lista de listas) ---
# (Mantener tu KDDataCollator, parece manejar bien la reconstrucción)
class KDDataCollator(DataCollatorForSeq2Seq): # Tu implementación
    def __init__(self, *args, teacher_vocab_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_vocab_size = teacher_vocab_size
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # ... tu lógica robusta de reconstrucción de tensor desde lista de listas ...
        reconstructed_teacher_logits_tensors = []
        for feature in features:
            list_of_lists = feature.pop("teacher_logits_list_of_lists")
            tensor = None
            if isinstance(list_of_lists, list):
                if not list_of_lists: tensor = torch.empty((0, self.teacher_vocab_size), dtype=torch.float32)
                elif all(isinstance(row, list) for row in list_of_lists):
                    if not list_of_lists[0] and len(list_of_lists) == 1: tensor = torch.empty((0, self.teacher_vocab_size), dtype=torch.float32)
                    elif not list_of_lists[0] and len(list_of_lists) > 1: raise ValueError(f"Logits con filas vacías inconsistentes.")
                    else:
                        try:
                            if not all(len(row) == len(list_of_lists[0]) for row in list_of_lists): raise ValueError(f"Sub-listas de logits con longitudes inconsistentes.")
                            tensor = torch.tensor(list_of_lists, dtype=torch.float32)
                        except Exception as e_tensor: raise ValueError(f"Error reconstruyendo tensor: {e_tensor}.")
                else: raise TypeError(f"Formato inesperado para logits: lista contiene no-lista.")
            elif isinstance(list_of_lists, torch.Tensor): tensor = list_of_lists
            else: raise TypeError(f"Formato inesperado para teacher_logits: {type(list_of_lists)}")
            reconstructed_teacher_logits_tensors.append(tensor)
        
        teacher_logits_list = reconstructed_teacher_logits_tensors
        batch = super().__call__(features)
        valid_teacher_logits = [t for t in teacher_logits_list if t.nelement() > 0 and t.ndim == 2]
        max_teacher_seq_len = 0
        if valid_teacher_logits:
            max_teacher_seq_len = max(t.shape[0] for t in valid_teacher_logits)
            if not all(t.shape[1] == self.teacher_vocab_size for t in valid_teacher_logits):
                raise ValueError(f"Tensores con vocab_size incorrecto. Esperado: {self.teacher_vocab_size}.")
        padded_teacher_logits = torch.zeros(len(teacher_logits_list), max_teacher_seq_len, self.teacher_vocab_size, dtype=torch.float32)
        for i, t_logits in enumerate(teacher_logits_list):
            if t_logits.nelement() > 0 and t_logits.ndim == 2:
                 padded_teacher_logits[i, :t_logits.shape[0], :] = t_logits
        batch["teacher_logits"] = padded_teacher_logits
        return batch

data_collator = KDDataCollator(tokenizer=student_tokenizer, padding="longest", label_pad_token_id=-100, teacher_vocab_size=TEACHER_LOGITS_ACTUAL_VOCAB_SIZE)

# --- 5. Trainer Personalizado para KD ---
class KDTrainer(Trainer):
    def __init__(self, *args, kd_alpha=0.5, kd_temperature=2.0, 
                 teacher_actual_vocab_size, # El tamaño real de los logits del profesor (262208)
                 student_model_vocab_size,  # El config.vocab_size del estudiante (262144)
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.teacher_actual_vocab_size = teacher_actual_vocab_size
        self.student_model_vocab_size = student_model_vocab_size
        
        print(f"KDTrainer Init: KD Alpha: {self.kd_alpha}")
        print(f"KDTrainer Init: Teacher Actual Vocab Size (de archivos .pt): {self.teacher_actual_vocab_size}")
        print(f"KDTrainer Init: Student Model Config Vocab Size: {self.student_model_vocab_size}")

        if self.kd_alpha > 0.0 and self.teacher_actual_vocab_size != self.student_model_vocab_size:
            print(f"INFO KDTrainer: Se aplicará KD. Los logits del profesor (vocab: {self.teacher_actual_vocab_size}) "
                  f"serán truncados para coincidir con el estudiante (vocab: {self.student_model_vocab_size}).")
        elif self.kd_alpha > 0.0: # Coinciden
             print(f"INFO KDTrainer: Vocabularios de profesor y estudiante ya coinciden ({self.student_model_vocab_size}). Se usará KD.")
            
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_logits_from_batch = inputs.pop("teacher_logits").to(model.device) # Estos son [B, S, 262208]
        outputs_student = model(**inputs)
        loss_sft = outputs_student.loss
        student_logits = outputs_student.logits # Estos son [B, S, 262144]

        if self.kd_alpha == 0.0:
            # Log SFT si es necesario (simplificado)
            if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0: self.log({"loss_sft": loss_sft.item(), "loss_kd": 0.0, "total_loss_combined": loss_sft.item()})
            return (loss_sft, outputs_student) if return_outputs else loss_sft

        # --- Lógica de Truncamiento para KD Loss ---
        # Los logits del profesor (teacher_logits_from_batch) tienen self.teacher_actual_vocab_size (262208)
        # Los logits del estudiante (student_logits) tienen self.student_model_vocab_size (262144)
        
        # Truncar los logits del profesor para que coincidan con el estudiante para la pérdida KD
        if self.teacher_actual_vocab_size > self.student_model_vocab_size:
            teacher_logits_for_kd = teacher_logits_from_batch[:, :, :self.student_model_vocab_size]
        elif self.teacher_actual_vocab_size < self.student_model_vocab_size:
            # Este caso no debería ocurrir con tus datos, pero es una salvaguarda
            # Aquí tendrías que rellenar los logits del profesor, o truncar los del estudiante.
            # Por simplicidad, si esto ocurre, se podría desactivar KD o lanzar un error.
            print(f"ERROR KD: Vocab del profesor ({self.teacher_actual_vocab_size}) es MENOR que el del estudiante ({self.student_model_vocab_size}). "
                  "Truncar profesor no es la estrategia correcta aquí. Saltando KD loss.")
            teacher_logits_for_kd = None # Señal para saltar KD
        else: # Coinciden (no debería pasar si entramos en el truncamiento)
            teacher_logits_for_kd = teacher_logits_from_batch

        loss_kd = torch.tensor(0.0, device=student_logits.device)
        if teacher_logits_for_kd is not None and \
           teacher_logits_for_kd.shape[-1] == student_logits.shape[-1]: # Doble comprobación
            
            len_teacher_resp = teacher_logits_for_kd.shape[1] # Usar la longitud de los logits del profesor (ya alineada en sec)

            if len_teacher_resp > 0 and teacher_logits_for_kd.nelement() > 0:
                log_probs_student = F.log_softmax(student_logits / self.kd_temperature, dim=-1)
                # Usar teacher_logits_for_kd (los truncados)
                probs_teacher = F.softmax(teacher_logits_for_kd / self.kd_temperature, dim=-1)
                
                effective_seq_len_for_kd = min(student_logits.shape[1], len_teacher_resp)
                
                # Asegurarse de que los slices sean consistentes
                log_probs_student_final = log_probs_student[:, :effective_seq_len_for_kd, :]
                probs_teacher_final = probs_teacher[:, :effective_seq_len_for_kd, :]
                
                kl_div_loss_unreduced = nn.KLDivLoss(reduction='none', log_target=False)(log_probs_student_final, probs_teacher_final)
                kl_div_loss_unreduced = kl_div_loss_unreduced.sum(dim=-1)
                
                loss_mask = (inputs["labels"] != -100) # Usar labels originales para la máscara
                active_loss_mask_for_kd = loss_mask[:, :effective_seq_len_for_kd]
                masked_kd_loss = kl_div_loss_unreduced.masked_fill(~active_loss_mask_for_kd, 0.0)
                num_active_tokens = active_loss_mask_for_kd.sum()

                if num_active_tokens > 0:
                    loss_kd = masked_kd_loss.sum() / num_active_tokens
        else:
             if teacher_logits_for_kd is not None: # Solo si no fue None desde el principio
                print(f"WARN KD: Discrepancia de vocabulario después del intento de truncamiento o logits vacíos. "
                      f"Student: {student_logits.shape[-1]}, Teacher (para KD): {teacher_logits_for_kd.shape[-1]}. Saltando KD.")


        total_loss = (1.0 - self.kd_alpha) * loss_sft + self.kd_alpha * loss_kd
        
        if self.state.global_step > 0 and self.state.global_step % self.args.logging_steps == 0:
            lr = self.lr_scheduler.get_last_lr()[0] if self.lr_scheduler else 0.0
            self.log({"loss_sft": loss_sft.item(), "loss_kd": loss_kd.item(), "total_loss_combined": total_loss.item(), "lr": lr})
        return (total_loss, outputs_student) if return_outputs else total_loss

    
# --- 6. Configuración del Entrenamiento ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01, # Regularización L2
    lr_scheduler_type="cosine", # O "linear", "constant_with_warmup"
    warmup_ratio=0.05, # Porcentaje de pasos totales para warmup, ej. 0.03-0.1
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=LOGGING_STEPS,
    report_to="tensorboard",
    fp16=(quantization_config is None and torch.cuda.is_available() and student_model.dtype != torch.bfloat16),
    bf16=(quantization_config is None and torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant':False},
    remove_unused_columns=False, # Necesario para que las columnas extra lleguen al collator
    dataloader_num_workers=0, 
    optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch", # Optimizador AdamW fusionado si hay GPU
)

STUDENT_VOCAB_SIZE_FROM_TOKENIZER = student_tokenizer.vocab_size
print(f"INFO: Usando tamaño de vocabulario profesor : {TEACHER_VOCAB_SIZE}")
print(f"INFO: Tamaño del vocabulario del estudiante (vocab_size): {STUDENT_VOCAB_SIZE_FROM_TOKENIZER}")

# --- 7. Crear el Trainer y Entrenar ---
trainer = KDTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
    data_collator=data_collator,
    tokenizer=student_tokenizer,
    kd_alpha=KD_ALPHA, # Se usará si los vocabularios se pueden alinear
    kd_temperature=KD_TEMPERATURE,
    teacher_actual_vocab_size=TEACHER_LOGITS_ACTUAL_VOCAB_SIZE, # 262208
    student_model_vocab_size=STUDENT_MODEL_CONFIG_VOCAB_SIZE    # 262144
)

print("Iniciando entrenamiento...")
# (Resto del script de entrenamiento y guardado)
# ...
train_output = None
try:
    train_output = trainer.train()
except Exception as e:
    print(f"Error durante el entrenamiento: {e}")
    import traceback
    traceback.print_exc()
    # Considera guardar el estado actual si falla para reanudar
    # trainer.save_model(os.path.join(OUTPUT_DIR, "checkpoint-on-error"))
    # student_tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "checkpoint-on-error"))
    # print("Modelo parcial guardado en checkpoint-on-error debido a un error.")
    # raise e # Re-lanzar si quieres que el script termine con error

# --- 8. Guardar Modelo Final y Métricas ---
if train_output:
    print("Entrenamiento completado. Guardando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    student_tokenizer.save_pretrained(OUTPUT_DIR)
    metrics = train_output.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
else:
    print("El entrenamiento no se completó o no generó resultados. No se guardará el modelo.")

print(f"Proceso de destilación (o SFT si alpha=0) completado. Modelo guardado en: {OUTPUT_DIR}")