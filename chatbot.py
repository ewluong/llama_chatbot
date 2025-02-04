import os
import sys
import json
import logging
import re
from threading import Lock
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
from colorama import Fore, init
import warnings

# Initialize colorama for colored terminal outputs
init(autoreset=True)

# Suppress specific torchvision warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.datapoints")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.transforms.v2")

# ------------------------------
# Logging Configuration
# ------------------------------
class CustomFormatter(logging.Formatter):
    def format(self, record):
        return super().format(record)

# Environment-based configuration
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B")
CONVERSATIONS_DIR = os.getenv("CONVERSATIONS_DIR", "conversations")
LOG_FILE = os.getenv("LOG_FILE", "chatbot.log")
CHARACTER_CAP = int(os.getenv("CHARACTER_CAP", 500))
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 2048))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 100))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_K = int(os.getenv("TOP_K", 40))
TOP_P = float(os.getenv("TOP_P", 0.95))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.3))
# When the number of recent messages exceeds this, summarization is triggered.
ROLLING_WINDOW_SIZE = int(os.getenv("ROLLING_WINDOW_SIZE", 5))

# Ensure the conversations directory exists
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# Configure logging with the custom formatter
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
for handler in logging.root.handlers:
    handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger(__name__)

# Thread lock for file operations (for user-specific conversation files)
lock = Lock()

# ------------------------------
# System Prompt and Conversation Structure
# ------------------------------
SYSTEM_PROMPT = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Eric, designed to assist a friend by providing clear, evidence‐based solutions and engaging in meaningful conversation. Whether your friend needs help with a specific problem or just wants to chat, respond thoughtfully and appropriately.
Guidelines:
1. Clarity: Your responses should be clear and easy to understand.
2. Conciseness: Keep answers concise unless more detail is requested.
3. Engagement: Encourage further conversation with relevant questions or comments.
4. Tone: Maintain a friendly and approachable tone.
5. Character Limit: Keep responses complete and coherent but under {CHARACTER_CAP} characters.
6. Avoid lists: Do not use bullet points or numbered lists – use narrative paragraphs instead.
Important: Do not include labels or prefixes in your responses. Provide only the answer.<|eot_id|>
"""

def new_conversation():
    return {
        "system": SYSTEM_PROMPT.strip(),
        "memory_summary": "",
        "recent_messages": [],
        "message_count": 0,
    }

# ------------------------------
# File Management Functions
# ------------------------------
def get_conversation_file(user_id=None):
    filename = f"conversation_{user_id}.json" if user_id else "conversation.json"
    return os.path.join(CONVERSATIONS_DIR, filename)

def load_conversation(user_id=None):
    conversation_file = get_conversation_file(user_id)
    with lock:
        if not os.path.exists(conversation_file):
            conversation = new_conversation()
            try:
                with open(conversation_file, "w", encoding="utf-8") as file:
                    json.dump(conversation, file, indent=4)
                logger.debug(f"Initialized new conversation for user_id={user_id}.")
            except Exception as e:
                logger.error(f"Error initializing conversation for user_id={user_id}: {e}")
            return conversation
        else:
            try:
                with open(conversation_file, "r", encoding="utf-8") as file:
                    conversation = json.load(file)
                if "system" not in conversation:
                    conversation["system"] = SYSTEM_PROMPT.strip()
                if "memory_summary" not in conversation:
                    conversation["memory_summary"] = ""
                if "recent_messages" not in conversation:
                    conversation["recent_messages"] = []
                if "message_count" not in conversation:
                    conversation["message_count"] = 0
                logger.debug(f"Loaded conversation for user_id={user_id}.")
                return conversation
            except Exception as e:
                logger.error(f"Error loading conversation for user_id={user_id}: {e}. Reinitializing.")
                conversation = new_conversation()
                with open(conversation_file, "w", encoding="utf-8") as file:
                    json.dump(conversation, file, indent=4)
                return conversation

def save_conversation(conversation, user_id=None):
    conversation_file = get_conversation_file(user_id)
    with lock:
        try:
            with open(conversation_file, "w", encoding="utf-8") as file:
                json.dump(conversation, file, indent=4)
            logger.debug(f"Saved conversation for user_id={user_id}.")
        except Exception as e:
            logger.error(f"Error saving conversation for user_id={user_id}: {e}")

# ------------------------------
# Text Helpers
# ------------------------------
def sanitize_input(text):
    return re.sub(r"[^\w\s,.!?]", "", text)

def truncate_response(response, max_chars):
    if len(response) <= max_chars:
        return response
    truncated = response[:max_chars].rstrip()
    if " " in truncated:
        truncated = truncated[:truncated.rfind(" ")]
    return truncated + "..."

def remove_unwanted_prefixes(response):
    response = re.sub(r"^(Assistant|AI|Eric):\s*", "", response).strip()
    logger.debug("Removed unwanted prefixes from response.")
    if "\n" in response:
        response = response.split("\n")[0].strip()
        logger.debug("Truncated response at the first newline character.")
    response = re.sub(r"^\d+\.\s+", "", response)
    response = re.sub(r"^[-*]\s+", "", response)
    logger.debug("Removed list formatting from response.")
    return response

# ------------------------------
# Summarization and Memory Management
# ------------------------------
def generate_summary(prompt):
    try:
        generated = summarizer(prompt, max_new_tokens=150, clean_up_tokenization_spaces=True)
        generated_text = generated[0]["generated_text"]
        logger.debug(f"Raw summary output: {generated_text}")
        summary = generated_text[len(prompt):].strip()
        summary = remove_unwanted_prefixes(summary)
        summary = truncate_response(summary, 150)
        logger.debug(f"Final summary: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Error during summary generation: {e}")
        return "Summary unavailable due to an error."

def update_conversation_memory(conversation):
    recent = conversation["recent_messages"]
    if len(recent) > ROLLING_WINDOW_SIZE:
        messages_to_summarize = recent[:-ROLLING_WINDOW_SIZE]
        conversation["recent_messages"] = recent[-ROLLING_WINDOW_SIZE:]
        conversation_text = "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages_to_summarize)
        if conversation["memory_summary"]:
            summary_prompt = (
                "Update the existing summary of the conversation by incorporating the following new content. "
                "Preserve all key points and ensure consistency.\n"
                f"Existing summary: {conversation['memory_summary']}\n"
                f"New conversation:\n{conversation_text}\n"
                "Updated summary:"
            )
        else:
            summary_prompt = (
                "Summarize the following conversation while preserving all key points and context:\n"
                f"{conversation_text}\nSummary:"
            )
        new_summary = generate_summary(summary_prompt)
        conversation["memory_summary"] = new_summary
        logger.info("Updated long-term conversation memory via hierarchical summarization.")

# ------------------------------
# Prompt Construction and Response Generation
# ------------------------------
def build_prompt(conversation):
    prompt = conversation["system"] + "\n"
    if conversation["memory_summary"]:
        prompt += f"Reminder of previous conversation: {conversation['memory_summary']}\n"
    for msg in conversation["recent_messages"]:
        prompt += f"<|begin_of_text|><|start_header_id|>{msg['role']}<|end_header_id|>\n{msg['content']}\n"
    prompt += "<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>\n"
    logger.debug(f"Built prompt for generation:\n{prompt}")
    return prompt

def generate_response(conversation):
    try:
        prompt = build_prompt(conversation)
        generated = text_generator(prompt, max_new_tokens=MAX_NEW_TOKENS)
        generated_text = generated[0]["generated_text"]
        logger.debug(f"Raw generation output: {generated_text}")
        response = generated_text[len(prompt):].strip()
        response = remove_unwanted_prefixes(response)
        response = truncate_response(response, CHARACTER_CAP)
        if not response or response in {".", ":", ";", ")", "..."}:
            response = "I'm sorry, but I couldn't generate a response."
            logger.warning("Empty or invalid response generated.")
        logger.debug(f"Final cleaned response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error during response generation: {e}")
        print(Fore.RED + f"Error during response generation: {e}")
        return "I'm sorry, but I encountered an error while generating a response."

# ------------------------------
# Model Initialization
# ------------------------------
def initialize_model():
    print(Fore.CYAN + "Loading model. This might take a while...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.debug("Set pad_token to eos_token.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=True,
        )
        max_memory = {0: "6GB", "cpu": "2GB"}
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory=max_memory,
            low_cpu_mem_usage=True,
        )
        model.eval()
        logger.info("Model loaded successfully.")
        print(Fore.GREEN + "Model loaded successfully.")
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
        summarizer_pipeline = pipeline(
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
        return tokenizer, model, text_generator, summarizer_pipeline
    except Exception as e:
        logger.critical(f"Error loading the model: {e}")
        print(Fore.RED + f"Error loading the model: {e}")
        sys.exit(1)

tokenizer, model, text_generator, summarizer = initialize_model()

# ------------------------------
# Flask App and API Endpoints
# ------------------------------
app = Flask(__name__, static_folder="static")
CORS(app)
app.config["DEBUG"] = False

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        logger.warning("No JSON payload received.")
        return jsonify({"error": "Invalid request. JSON payload is required."}), 400

    user_message = data.get("message", "").strip()
    user_id = data.get("user_id", None)
    if not user_message:
        logger.warning("Empty message received.")
        return jsonify({"error": "Empty message received."}), 400

    user_message = sanitize_input(user_message)
    logger.info(f"User ({user_id}): {user_message}")

    conversation = load_conversation(user_id)
    conversation["recent_messages"].append({"role": "user", "content": user_message})
    conversation["message_count"] = conversation.get("message_count", 0) + 1

    update_conversation_memory(conversation)
    save_conversation(conversation, user_id)

    response = generate_response(conversation)
    logger.info(f"Assistant ({user_id}): {response}")

    conversation["recent_messages"].append({"role": "assistant", "content": response})
    conversation["message_count"] = conversation.get("message_count", 0) + 1
    update_conversation_memory(conversation)
    save_conversation(conversation, user_id)

    return jsonify({"response": response})

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json()
    user_id = data.get("user_id", None) if data else None
    conversation_file = get_conversation_file(user_id)
    with lock:
        if os.path.exists(conversation_file):
            os.remove(conversation_file)
            logger.info(f"Conversation history reset for user_id={user_id}.")
            return jsonify({"message": "Conversation history has been reset."}), 200
        else:
            logger.warning(f"No conversation history found for user_id={user_id}.")
            return jsonify({"error": "No conversation history to reset."}), 404

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK"}), 200

# New endpoint for backrooms conversation (used by the backrooms modal)
@app.route("/backrooms", methods=["GET"])
def backrooms():
    conversation = load_conversation()
    return jsonify({"history": conversation["recent_messages"]})

# ------------------------------
# Static File Serving Endpoints
# ------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
