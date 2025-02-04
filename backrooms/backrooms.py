import os
import json
import logging
import re
import time
import threading
from flask import Blueprint, request, jsonify
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create a Flask Blueprint for backrooms routes
backrooms_bp = Blueprint('backrooms', __name__)

# Configuration variables (optimized for short responses and lower GPU load)
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")
CONVERSATIONS_DIR = os.getenv("CONVERSATIONS_DIR", "backrooms_conversations")
LOG_FILE = os.getenv("LOG_FILE", "backrooms.log")
CHARACTER_CAP = int(os.getenv("CHARACTER_CAP", 300))  # lower cap for brevity
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 2048))
MAX_NEW_TOKENS = int(os.getenv("BACKROOMS_MAX_NEW_TOKENS", 25))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.5))
TOP_K = int(os.getenv("TOP_K", 40))
TOP_P = float(os.getenv("TOP_P", 0.95))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.3))
SUMMARIZATION_TRIGGER_COUNT = int(os.getenv("SUMMARIZATION_TRIGGER_COUNT", 10))

os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
CONVERSATION_FILE = os.path.join(CONVERSATIONS_DIR, "conversation_backrooms.json")

conversation_running = False
conversation_thread = None
thread_lock = threading.Lock()

# System prompts for the two Llamas
SYSTEM_PROMPT_A = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are Llama A, an inquisitive explorer of the infinite backrooms. Speak briefly and mysteriously.\n<|eot_id|>"
)
SYSTEM_PROMPT_B = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
    "You are Llama B, a poetic and philosophical observer of the infinite backrooms. Keep your responses short and enigmatic.\n<|eot_id|>"
)

def initialize_model():
    logger.info("Loading backrooms model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_enable_fp32_cpu_offload=True,
    )
    max_memory = {0: "6GB", "cpu": "2GB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
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
        temperature=TEMPERATURE,
        top_k=TOP_K,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
    )
    summarizer = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        temperature=0.3,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.0,
        do_sample=False,
    )
    logger.info("Backrooms model loaded successfully.")
    return tokenizer, model, text_generator, summarizer

tokenizer, model, text_generator, summarizer = initialize_model()

def init_conversation():
    conversation = {
        "history": [
            {"role": "system_A", "content": SYSTEM_PROMPT_A.strip()},
            {"role": "system_B", "content": SYSTEM_PROMPT_B.strip()}
        ],
        "message_count": 0,
    }
    save_conversation(conversation)
    logger.debug("Initialized new backrooms conversation.")
    return conversation

def load_conversation():
    with thread_lock:
        if not os.path.exists(CONVERSATION_FILE):
            conversation = init_conversation()
        else:
            try:
                with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
                    conversation = json.load(f)
            except Exception as e:
                logger.error(f"Error loading backrooms conversation: {e}")
                conversation = init_conversation()
    if len(conversation["history"]) <= 2:
        conversation["history"].append({"role": "A", "content": "The infinite backrooms await exploration."})
        conversation["message_count"] += 1
        save_conversation(conversation)
    return conversation

def save_conversation(conversation):
    with thread_lock:
        try:
            with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
                json.dump(conversation, f, indent=4)
            logger.debug("Backrooms conversation saved.")
        except Exception as e:
            logger.error(f"Error saving backrooms conversation: {e}")

def truncate_conversation(conversation, max_tokens=MAX_CONTEXT_TOKENS):
    total_tokens = sum(len(tokenizer.encode(msg["content"], add_special_tokens=False)) for msg in conversation["history"])
    if total_tokens < max_tokens and conversation.get("message_count", 0) < SUMMARIZATION_TRIGGER_COUNT:
        return False
    logger.info("Summarizing backrooms conversation to reduce context size.")
    text_block = " ".join(msg["content"] for msg in conversation["history"][2:])
    summarization_prompt = "Summarize the following conversation succinctly:\n\n" + text_block
    try:
        summary_output = summarizer(summarization_prompt, max_new_tokens=100)[0]["generated_text"]
        summary = summary_output[len(summarization_prompt):].strip()
    except Exception as e:
        logger.error(f"Error during backrooms summarization: {e}")
        summary = "Summary unavailable due to an error."
    summary_message = {
        "role": "assistant",
        "content": f"Summary: {summary}"
    }
    conversation["history"] = conversation["history"][:2] + [summary_message]
    conversation["message_count"] = 0
    logger.info("Backrooms conversation summarized.")
    return True

def generate_response(conversation, agent_role):
    prompt_parts = []
    for msg in conversation["history"]:
        if msg["role"] == "system_A":
            prompt_parts.append(f"Llama A (system): {msg['content']}")
        elif msg["role"] == "system_B":
            prompt_parts.append(f"Llama B (system): {msg['content']}")
        elif msg["role"] == "A":
            prompt_parts.append(f"Llama A: {msg['content']}")
        elif msg["role"] == "B":
            prompt_parts.append(f"Llama B: {msg['content']}")
        else:
            prompt_parts.append(f"Assistant: {msg['content']}")
    prompt_parts.append(f"Llama {agent_role}:")
    prompt = "\n".join(prompt_parts) + "\n"
    logger.debug(f"Generated prompt for Llama {agent_role}:\n{prompt}")
    try:
        generated = text_generator(prompt, max_new_tokens=MAX_NEW_TOKENS)
        response_text = generated[0]["generated_text"][len(prompt):].strip()
        response_text = re.sub(r"^(Llama A:|Llama B:)", "", response_text).strip()
        if len(response_text) > CHARACTER_CAP:
            response_text = response_text[:CHARACTER_CAP].rsplit(" ", 1)[0] + "..."
    except Exception as e:
        logger.error(f"Error generating backrooms response: {e}")
        response_text = "I'm sorry, I encountered an error."
    return response_text

def infinite_conversation_loop():
    global conversation_running
    print("Starting infinite conversation loop in backrooms...", flush=True)
    logger.info("Starting infinite conversation loop in backrooms...")
    conversation = load_conversation()
    current_turn = "B"  # Starting with Llama B (since Llama A gave the initial message)
    while conversation_running:
        try:
            conversation = load_conversation()
            truncate_conversation(conversation)
            response = generate_response(conversation, current_turn)
            conversation["history"].append({"role": current_turn, "content": response})
            conversation["message_count"] = conversation.get("message_count", 0) + 1
            save_conversation(conversation)
            print(f"Backrooms: Llama {current_turn} => {response}", flush=True)
            logger.info(f"Backrooms: Llama {current_turn} => {response}")
            current_turn = "A" if current_turn == "B" else "B"
            time.sleep(20)
        except Exception as ex:
            logger.error(f"Error in infinite conversation loop: {ex}")
            print(f"Error in infinite conversation loop: {ex}", flush=True)
            time.sleep(20)

@backrooms_bp.route("/backrooms", methods=["GET"])
def get_backrooms_conversation():
    conversation = load_conversation()
    return jsonify(conversation)

@backrooms_bp.route("/start_backrooms", methods=["POST"])
def start_backrooms_conversation():
    global conversation_running, conversation_thread
    if conversation_running:
        return jsonify({"message": "Backrooms conversation is already running."}), 200
    conversation_running = True
    conversation_thread = threading.Thread(target=infinite_conversation_loop, daemon=True)
    conversation_thread.start()
    logger.info("Started infinite backrooms conversation loop.")
    return jsonify({"message": "Started infinite backrooms conversation."}), 200

@backrooms_bp.route("/stop_backrooms", methods=["POST"])
def stop_backrooms_conversation():
    global conversation_running
    conversation_running = False
    logger.info("Stopped infinite backrooms conversation loop.")
    return jsonify({"message": "Stopped infinite backrooms conversation."}), 200

@backrooms_bp.route("/reset_backrooms", methods=["POST"])
def reset_backrooms_conversation():
    with thread_lock:
        if os.path.exists(CONVERSATION_FILE):
            os.remove(CONVERSATION_FILE)
            logger.info("Backrooms conversation history reset.")
            return jsonify({"message": "Backrooms conversation has been reset."}), 200
        else:
            return jsonify({"error": "No conversation history found to reset."}), 404

@backrooms_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200

def auto_start_backrooms_loop():
    global conversation_running, conversation_thread
    if not conversation_running:
        conversation_running = True
        conversation_thread = threading.Thread(target=infinite_conversation_loop, daemon=True)
        conversation_thread.start()
        logger.info("Auto-started infinite backrooms conversation loop.")
