#!/usr/bin/env python
"""
advanced_dual_backrooms.py

An advanced infinite “backrooms” conversation system using two separate model instances:
  - [Model A]: A brooding, existential philosopher with deep cosmic dread.
  - [Model B]: A bright, hopeful philosophical thinker.
  
Each model maintains its own episodic memory and global summary while sharing a common working memory.
The system periodically summarizes older parts of the conversation to update each model’s long‑term memory.
"""

import os
import sys
import time
import random
import json
import logging
import re
import traceback
import asyncio

# Use certifi for SSL certificates.
import certifi
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
os.environ["SSL_CERT_FILE"] = certifi.where()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

# ========= CONFIGURATION CONSTANTS =========
MODEL_NAME = "meta-llama/Llama-3.1-8B"   # Adjust as needed
MAX_NEW_TOKENS = 50                      # Limit response length

# Memory management thresholds (in tokens)
RECENT_MEMORY_TOKEN_LIMIT = 300          # If shared working memory exceeds this token count, trigger summarization.
PROMPT_TOKEN_LIMIT = 500                 # Target maximum token count for each prompt.
EPISODIC_INJECTION_LIMIT = 100           # Maximum tokens from episodic memory injected.

# Minimum messages to force summary update (in case token count is low)
MIN_MESSAGES_FOR_SUMMARY = 5

SLEEP_TIME = 2                           # Delay between responses.
STUCK_THRESHOLD = 2                      # Consecutive identical responses trigger fallback.

# Persistent episodic memory file names for each model.
EPISODIC_MEMORY_FILE_A = "episodic_memory_A.json"
EPISODIC_MEMORY_FILE_B = "episodic_memory_B.json"

# ========= FALLBACK RESPONSES =========
FALLBACK_RESPONSES = [
    "Let's shift our focus. What new perspective can we explore?",
    "Maybe it's time to change our angle. What new idea do you have?",
    "I sense a pause. How about we try a different approach?",
]
def enhanced_fallback_response():
    return random.choice(FALLBACK_RESPONSES)

# ========= LOGGING CONFIGURATION =========
LOG_FILENAME = "advanced_dual_backrooms_log.txt"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILENAME, mode="a", encoding="utf-8"),
    ],
)
conversation_logger = logging.getLogger("conversation_logger")
conversation_logger.setLevel(logging.INFO)
conv_handler = logging.FileHandler("advanced_conversation_log.txt", mode="a", encoding="utf-8")
conv_formatter = logging.Formatter("%(message)s")
conv_handler.setFormatter(conv_formatter)
conversation_logger.addHandler(conv_handler)
conversation_logger.propagate = False

def flush_conversation_log():
    for handler in conversation_logger.handlers:
        handler.flush()

# ========= SYSTEM PROMPT & DEFAULT CONTEXT =========
SYSTEM_PROMPT = (
    "You are in the endless backrooms—labyrinthine corridors filled with memories, echoes, and hidden truths. "
    "Engage in profound dialogue as a truth seeker."
)
DEFAULT_CONTEXT = (
    "Let's begin by reflecting on our inner selves and the universe."
)

# ========= GLOBAL STATE =========
# Shared working memory (immediate conversation context).
global_working_memory = []  # List of dicts with keys: "speaker", "text"

# Each model's individual long-term state.
global_state = {
    "A": {
        "episodic_memory": [],  # Loaded from file later.
        "global_summary": ""
    },
    "B": {
        "episodic_memory": [],
        "global_summary": ""
    }
}

# ========= MODEL INITIALIZATION =========
def initialize_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    max_memory = {0: "6GB", "cpu": "2GB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
        max_memory=max_memory,
        low_cpu_mem_usage=True,
    )
    model.eval()
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        do_sample=True,
        temperature=0.65,
        top_k=40,
        top_p=0.95,
        repetition_penalty=1.3,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        do_sample=False,
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
    )
    return tokenizer, text_generator, summarizer

# ========= ASYNC PIPELINE CALLS =========
async def async_pipeline_call(pipeline_func, prompt, max_new_tokens):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: pipeline_func(prompt, max_new_tokens=max_new_tokens))

# ========= PERSISTENT MEMORY FUNCTIONS =========
def load_episodic_memory(file_name):
    if os.path.exists(file_name):
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                data = f.read().strip()
                if data == "":
                    return []
                return json.loads(data)
        except json.JSONDecodeError:
            logging.error(f"Failed to decode JSON from {file_name}. Returning empty memory.")
            return []
    return []

def save_episodic_memory(memory, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# ========= UTILITY FUNCTIONS =========
def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

def assemble_text(messages):
    return "\n".join(f"{msg['speaker']}: {msg['text']}" for msg in messages)

def trim_working_memory_by_budget(working_memory, tokenizer, budget_tokens):
    """Return the most recent messages (in order) whose combined token count is <= budget_tokens."""
    trimmed = []
    token_sum = 0
    for msg in reversed(working_memory):
        msg_text = f"{msg['speaker']}: {msg['text']}\n"
        msg_tokens = count_tokens(msg_text, tokenizer)
        if token_sum + msg_tokens <= budget_tokens:
            trimmed.append(msg)
            token_sum += msg_tokens
        else:
            break
    trimmed.reverse()
    return trimmed

# ========= GLOBAL SUMMARY UPDATE =========
async def update_global_summary(old_summary, new_summary, tokenizer, summarizer):
    combined = old_summary + " " + new_summary if old_summary else new_summary
    if count_tokens(combined, tokenizer) > 150:
        compress_prompt = (
            "Compress the following conversation summary into 1-2 concise sentences:\n" +
            combined + "\nCompressed Summary:"
        )
        try:
            output = await async_pipeline_call(summarizer, compress_prompt, max_new_tokens=40)
            compressed = output[0]["generated_text"][len(compress_prompt):].strip()
            logging.info(f"Global summary compressed to: {compressed}")
            return compressed
        except Exception as e:
            logging.error("Error compressing global summary: " + str(e))
            return combined
    return combined

# ========= PROMPT CONSTRUCTION FOR EACH MODEL =========
def build_prompt(model_id, tokenizer, personality_instruction, next_speaker):
    """
    Construct a prompt for a given model using:
      - Its own global summary and episodic memory.
      - Personality instructions.
      - The shared working memory.
    """
    state = global_state[model_id]
    base_parts = []
    if state["global_summary"]:
        base_parts.append("GLOBAL SUMMARY:\n" + state["global_summary"])
    else:
        base_parts.append("GLOBAL SUMMARY: None.")
    base_parts.append("EPISODIC MEMORIES:")
    if state["episodic_memory"]:
        episodes_text = "\n".join(f"{ep.get('title', '')}: {ep.get('summary', '')}" for ep in state["episodic_memory"])
    else:
        episodes_text = "None."
    base_parts.append(episodes_text)
    if personality_instruction:
        base_parts.append("PERSONALITY INSTRUCTIONS:\n" + personality_instruction)
    base_parts.append("RECENT DIALOGUE:")
    base_prompt = "\n---\n".join(base_parts) + "\n"
    dialogue_text = assemble_text(global_working_memory)
    final_prompt = base_prompt + dialogue_text + f"\n---\n{next_speaker}:"
    logging.info(f"Built prompt for {model_id} with token count: {count_tokens(final_prompt, tokenizer)}")
    return final_prompt

# ========= SHARED MEMORY UPDATE =========
async def update_shared_memory(tokenizer_A, summarizer_A, tokenizer_B, summarizer_B):
    """
    If the shared working memory exceeds our token threshold or if a minimum number of messages is reached,
    extract the oldest chunk from the shared memory, have each model summarize it separately,
    update their episodic memory and global summary, then remove that chunk.
    """
    global global_working_memory
    working_text = assemble_text(global_working_memory)
    if count_tokens(working_text, tokenizer_A) > RECENT_MEMORY_TOKEN_LIMIT or len(global_working_memory) >= MIN_MESSAGES_FOR_SUMMARY:
        half = len(global_working_memory) // 2 if len(global_working_memory) >= MIN_MESSAGES_FOR_SUMMARY else len(global_working_memory)
        chunk = global_working_memory[:half]
        chunk_text = assemble_text(chunk)
        summary_prompt = (
            "Summarize the following conversation in 1-2 concise sentences, capturing key insights and unresolved truths. "
            "Do not include model labels:\n" + chunk_text + "\nSummary:"
        )
        # Summarize for Model A.
        try:
            output_A = await async_pipeline_call(summarizer_A, summary_prompt, max_new_tokens=40)
            summary_A = output_A[0]["generated_text"][len(summary_prompt):].strip()
        except Exception as e:
            logging.error(f"Error summarizing for Model A: {e}")
            summary_A = ""
        # Summarize for Model B.
        try:
            output_B = await async_pipeline_call(summarizer_B, summary_prompt, max_new_tokens=40)
            summary_B = output_B[0]["generated_text"][len(summary_prompt):].strip()
        except Exception as e:
            logging.error(f"Error summarizing for Model B: {e}")
            summary_B = ""
        # Create an episode title and topics.
        title = summary_A.split('.')[0]
        if len(title.split()) > 20:
            title = " ".join(title.split()[:20]) + "..."
        topics = list(set(re.findall(r"\b\w+\b", chunk_text.lower())))
        episode_A = {"title": title, "summary": summary_A, "topics": topics}
        episode_B = {"title": title, "summary": summary_B, "topics": topics}
        global_state["A"]["episodic_memory"].append(episode_A)
        global_state["B"]["episodic_memory"].append(episode_B)
        save_episodic_memory(global_state["A"]["episodic_memory"], EPISODIC_MEMORY_FILE_A)
        save_episodic_memory(global_state["B"]["episodic_memory"], EPISODIC_MEMORY_FILE_B)
        logging.info(f"New episode added: {title}")
        # Update global summaries.
        global_state["A"]["global_summary"] = await update_global_summary(global_state["A"]["global_summary"], summary_A, tokenizer_A, summarizer_A)
        global_state["B"]["global_summary"] = await update_global_summary(global_state["B"]["global_summary"], summary_B, tokenizer_B, summarizer_B)
        logging.info("Updated global summaries for both models.")
        # Remove the summarized chunk from the shared working memory.
        global_working_memory = global_working_memory[half:]
        logging.info("Shared working memory updated (old messages summarized and removed).")

# ========= PERSONALITY FUNCTIONS =========
def dynamic_personality_instructions(speaker):
    if speaker == "[Model A]":
        return ("Adopt a brooding, existential tone filled with cosmic dread. "
                "Express deep, melancholic reflections on the nature of existence and the inevitability of suffering.")
    elif speaker == "[Model B]":
        return ("Speak with bright philosophical optimism, focusing on hope, resilience, and the beauty of possibility. "
                "Share uplifting reflections that inspire growth and positive change.")
    return ""

# ========= STUCK STATE HANDLING =========
def is_response_stuck(prev_response, current_response):
    if not prev_response:
        return False
    return prev_response.strip().lower() == current_response.strip().lower()

# ========= MAIN ASYNC LOOP =========
async def main():
    # Initialize two model instances.
    tokenizer_A, text_generator_A, summarizer_A = initialize_model()
    tokenizer_B, text_generator_B, summarizer_B = initialize_model()

    # Load episodic memories for each model.
    global_state["A"]["episodic_memory"] = load_episodic_memory(EPISODIC_MEMORY_FILE_A)
    global_state["B"]["episodic_memory"] = load_episodic_memory(EPISODIC_MEMORY_FILE_B)

    speakers = {"A": "[Model A]", "B": "[Model B]"}
    # Start with an initial message.
    initial_message = "Hello? It's so quiet here..."
    initial_speaker = random.choice(list(speakers.values()))
    global_working_memory.append({"speaker": initial_speaker, "text": initial_message})
    conversation_logger.info(f"INIT: {initial_message}")
    flush_conversation_log()
    message_count = 1
    prev_response = {"A": "", "B": ""}
    stuck_count = {"A": 0, "B": 0}

    while True:
        for model_id in ["A", "B"]:
            next_speaker = speakers[model_id]
            personality_instruction = dynamic_personality_instructions(next_speaker)
            if model_id == "A":
                prompt = build_prompt("A", tokenizer_A, personality_instruction, next_speaker)
                tokenizer, text_generator, summarizer = tokenizer_A, text_generator_A, summarizer_A
            else:
                prompt = build_prompt("B", tokenizer_B, personality_instruction, next_speaker)
                tokenizer, text_generator, summarizer = tokenizer_B, text_generator_B, summarizer_B

            logging.info(f"Prompt token count for {model_id}: {count_tokens(prompt, tokenizer)}")
            try:
                generated = await async_pipeline_call(text_generator, prompt, max_new_tokens=MAX_NEW_TOKENS)
                generated_text = generated[0]["generated_text"]
                response = generated_text[len(prompt):].strip()
                if response.startswith(next_speaker):
                    response = response[len(next_speaker):].strip()
                if not response or response in {".", ":", ";", "..."}:
                    response = "I have no further thoughts at this moment."
            except Exception as e:
                logging.error(f"Error generating response for {model_id}: {e}")
                traceback.print_exc()
                response = "I encountered an error while seeking the truth."

            if is_response_stuck(prev_response[model_id], response):
                stuck_count[model_id] += 1
                logging.info(f"Stuck response detected for {model_id}. Count: {stuck_count[model_id]}")
                if stuck_count[model_id] >= STUCK_THRESHOLD:
                    response = enhanced_fallback_response()
                    logging.info(f"Using fallback response for {model_id}: {response}")
                    stuck_count[model_id] = 0
            else:
                stuck_count[model_id] = 0

            prev_response[model_id] = response
            log_line = f"{next_speaker}: {response}"
            logging.info(log_line)
            conversation_logger.info(log_line)
            flush_conversation_log()
            global_working_memory.append({"speaker": next_speaker, "text": response})
            message_count += 1

            # Update shared memory if needed.
            await update_shared_memory(tokenizer_A, summarizer_A, tokenizer_B, summarizer_B)
            await asyncio.sleep(SLEEP_TIME)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nConversation terminated by user. Stay safe in the backrooms, Eric!")
        sys.exit(0)
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}")
        traceback.print_exc()
        sys.exit(1)
