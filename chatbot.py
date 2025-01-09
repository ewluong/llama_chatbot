import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import sys
import os
import logging
from colorama import Fore, Style, init

# Initialize colorama for colored terminal outputs
init(autoreset=True)

# Configure logging
LOG_FILE = "chatbot.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define constants
MODEL_NAME = "meta-llama/Llama-3.2-3B"  # Ensure this is the correct and available model
CONVERSATION_FILE = "conversation.json"
CHARACTER_CAP = 500  # Maximum characters per response
MAX_CONTEXT_TOKENS = 2048  # Adjust based on model's capacity (commonly 2048 for LLaMA models)

# Initial system prompt without markdown or example dialogues
SYSTEM_PROMPT = """
You are an AI language model designed to assist users by providing clear, evidence-based solutions and engaging in meaningful conversations. Whether the user seeks help with a specific problem or just wants to chat, respond thoughtfully and appropriately.

Guidelines:
1. Clarity: Ensure responses are clear and easy to understand.
2. Conciseness: Keep answers concise unless detailed explanations are requested.
3. Engagement: Encourage ongoing conversation with relevant questions or comments.
4. Tone: Maintain a friendly and approachable tone.

Important: Do not include any labels or prefixes (e.g., "Assistant:", "AI Response 1:") in your responses. Provide only the answer.
"""

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your website

def initialize_conversation():
    """
    Initialize the conversation history.
    If the conversation file exists, load it. Otherwise, start with the system prompt.
    """
    if os.path.exists(CONVERSATION_FILE):
        with open(CONVERSATION_FILE, "r", encoding="utf-8") as file:
            try:
                conversation = json.load(file)
                logging.debug("Loaded existing conversation history.")
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON from conversation file. Starting fresh.")
                conversation = {"history": [{"role": "system", "content": SYSTEM_PROMPT.strip()}]}
    else:
        conversation = {"history": [{"role": "system", "content": SYSTEM_PROMPT.strip()}]}
        with open(CONVERSATION_FILE, "w", encoding="utf-8") as file:
            json.dump(conversation, file, indent=4)
        logging.debug("Initialized new conversation history with system prompt.")
    return conversation

def save_conversation(conversation):
    """
    Save the conversation history to a JSON file.
    """
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as file:
        json.dump(conversation, file, indent=4)
    logging.debug("Saved updated conversation history.")

def count_tokens(tokenizer, conversation):
    """
    Count the number of tokens in the conversation using the tokenizer.
    """
    tokens = 0
    for message in conversation["history"]:
        # Exclude special tokens added by the tokenizer
        tokens += len(tokenizer.encode(message["content"], add_special_tokens=False))
    logging.debug(f"Current token count: {tokens}")
    return tokens

def truncate_conversation(tokenizer, conversation, max_tokens=MAX_CONTEXT_TOKENS):
    """
    Truncate the conversation history to fit within the model's context window.
    Removes the oldest user-assistant pair while preserving the system prompt.
    """
    while count_tokens(tokenizer, conversation) > max_tokens:
        if len(conversation["history"]) > 2:
            removed_message = conversation["history"].pop(1)  # Remove the second message
            logging.debug(f"Truncated message: {removed_message}")
        else:
            logging.warning("Cannot truncate conversation further. Context limit may be exceeded.")
            break
    return conversation

def remove_unwanted_prefixes(response):
    """
    Remove any prefixes like "Assistant:", "AI Response:", etc., from the response.
    Also truncate the response at the first newline character.
    """
    prefixes = ["Assistant:", "AI Response:", "Bot:", "AI:"]
    for prefix in prefixes:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
            logging.debug(f"Removed prefix '{prefix}' from response.")
    # Truncate at the first newline character
    if "\n" in response:
        response = response.split("\n")[0].strip()
        logging.debug("Truncated response at the first newline character.")
    return response

def generate_response(model, tokenizer, conversation):
    """
    Generate a response from the model based on the conversation history.
    """
    try:
        # Prepare the input by concatenating all messages
        prompt = ""
        for message in conversation["history"]:
            if message["role"] == "system":
                prompt += f"{message['content']}\n"
            elif message["role"] == "user":
                prompt += f"User: {message['content']}\n"
            elif message["role"] == "assistant":
                prompt += f"{message['content']}\n"

        logging.debug(f"Prompt for generation:\n{prompt}")

        # Tokenize input with attention_mask
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
            padding=False,
            add_special_tokens=False
        )
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=150,          # Adjust as needed
                do_sample=True,
                temperature=0.7,             # Controls randomness
                top_k=50,                     # Limits sampling to top K tokens
                top_p=0.95,                   # Nucleus sampling
                repetition_penalty=1.2,       # Penalizes repetition
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.debug(f"Raw model response: {response}")

        # Extract only the new part of the response
        response = response[len(prompt):].strip()
        logging.debug(f"Extracted response: {response}")

        # Enforce character caps
        if len(response) > CHARACTER_CAP:
            response = response[:CHARACTER_CAP].rstrip() + "..."
            logging.debug("Response truncated to character cap.")

        # Remove any unintended prefixes and truncate at first newline
        response = remove_unwanted_prefixes(response)
        logging.debug(f"Final response after cleaning: {response}")

        return response

    except Exception as e:
        logging.error(f"Error during response generation: {e}")
        print(Fore.RED + f"Error during response generation: {e}")
        return "I'm sorry, but I encountered an error while generating a response."

# Load the tokenizer and model once when the server starts
print(Fore.CYAN + "Loading model. This might take a while...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Set pad_token to eos_token if not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.debug("pad_token was None. Set pad_token to eos_token.")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",                     # Automatically map layers to available devices
        torch_dtype=torch.float16,            # Use half-precision for efficiency
        load_in_8bit=True,                     # Enable 8-bit quantization if supported
        # llm_int8_enable_fp32_cpu_offload=True  # Uncomment if you face memory issues
    )
    model.eval()  # Set model to evaluation mode
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading the model: {e}")
    print(Fore.RED + f"Error loading the model: {e}")
    sys.exit(1)

print(Fore.GREEN + "Model loaded and API is starting...")

# Initialize or load conversation history
conversation = initialize_conversation()

@app.route('/chat', methods=['POST'])
def chat():
    global conversation
    data = request.get_json()
    user_message = data.get('message', '').strip()

    if not user_message:
        return jsonify({'error': 'Empty message received.'}), 400

    logging.info(f"User: {user_message}")

    # Append user input to conversation
    conversation["history"].append({"role": "user", "content": user_message})

    # Truncate conversation to manage context length
    conversation = truncate_conversation(tokenizer, conversation, max_tokens=MAX_CONTEXT_TOKENS)

    # Generate response
    response = generate_response(model, tokenizer, conversation)
    logging.info(f"Assistant: {response}")

    # Append bot response to conversation
    conversation["history"].append({"role": "assistant", "content": response})

    # Save updated conversation
    save_conversation(conversation)

    return jsonify({'response': response})

if __name__ == "__main__":
    # Run the Flask app
    app.run(host='127.0.0.1', port=5000)
