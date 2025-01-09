# Chatbot Project 🚀

This project is a fully functional chatbot powered by the **Llama 3.2-3B** model and integrated with a dynamic frontend interface. The chatbot is designed to assist users with meaningful conversations and solutions in a user-friendly chatbox.

---

## Features 🌟

### Backend
- **Transformer-Based Chatbot:**
  - Uses the **Llama 3.2-3B** model for natural language processing.
  - Supports token truncation to fit within context limits.
  - Handles conversation history and session management.
- **Flask API:**
  - Provides a `/chat` endpoint for handling user queries.
  - Ensures responses are concise, clear, and meaningful.

### Frontend
- **Dynamic Chatbox Modal:**
  - Draggable, responsive, and styled for modern user experience.
  - Real-time user-bot interaction.
- **Customizable UI:**
  - Dark theme with color changes based on active sections.
  - Message animations for a seamless user experience.

---

## Setup Instructions 🛠️

### Prerequisites
1. **Python 3.8+**
2. **Virtual Environment** (optional but recommended)
3. **Node.js** (optional, if further frontend work is required)

---

### Backend Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repo-name/chatbot.git
   cd chatbot

2. **Create and Activate a Virtual Environment:**
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

3. **Install Dependencies:**

pip install -r requirements.txt

