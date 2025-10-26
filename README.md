# Local_CLI_Chatbot

A lightweight **local command-line chatbot** powered by a Hugging Face text generation model (e.g., `microsoft/DialoGPT-small`).  
It maintains a short-term conversational memory to create coherent multi-turn dialogue — all running locally with no API calls.

---

##  Features

- 💬 **Conversational Memory** – remembers the last few exchanges (configurable window)
- ⚙️ **Local Model Execution** – runs fully offline using Hugging Face `transformers`
- 🧩 **Modular Structure** – clean separation of logic across files
- 🧠 **Sliding Window Memory Buffer** – keeps recent context coherent
- 🖥️ **Command-Line Interface** – simple, responsive, and developer-friendly

---

##  Project Structure

Local_CLI_Chatbot/
│
├── model_loader.py # Loads model & tokenizer using Hugging Face pipeline
├── chat_memory.py # Manages short-term chat memory (sliding window)
├── main_chat.py # Main CLI loop (integration of model + memory)
├── requirements.txt # Python dependencies
└── README.md # Project documentation (this file)


---

## ⚙️ Setup Instructions

### 1️⃣ Create and Activate a Virtual Environment
```bash
python -m venv myenv
myenv\Scripts\activate      # On Windows

# OR
source myenv/bin/activate   # On macOS/Linux
```

2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
## Run the Chatbot
```
python main_chat.py --model Qwen/Qwen2-0.5B --memory 4 --max-new-tokens 80
```
---

### Example Output

<img width="427" height="138" alt="Image" src="https://github.com/user-attachments/assets/6dc73c2d-6663-4b78-88b1-ef0641df06fe" />



---

 ### Command Reference

| Command       | Description                     |
| ------------- | ------------------------------- |
| `/exit`       | Quit the chatbot                |
| `/clear`      | Clear short-term chat memory    |
| (any message) | Sends your message to the model |

---


## Customization

Change model:
Edit the --model argument to try a different Hugging Face model (e.g. distilgpt2 or TinyLlama/TinyLlama-1.1B-Chat-v1.0)

Change memory window:
--memory 5 keeps the last 5 turns of conversation.

Change response length:
Adjust --max-new-tokens for longer or shorter replies.

---
🧩 Author

Developed as part of a Local Chatbot Integration Task — demonstrating modular design, conversational memory, and local Hugging Face model usage.
