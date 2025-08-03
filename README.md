# MediBot AI

A powerful medical assistant web app combining deep learning symptom-based disease prediction and LLM-powered Q\&A with semantic search (RAG) over medical documents.

---

## Project Structure

```
MEDIBOT_AI/
├── app.py                  # Main Flask app entry point
├── dataset.csv             # Dataset for symptom-disease model
├── model/                  # Disease prediction model files
│   ├── model.h5            # Trained TensorFlow model for disease prediction
│   ├── labelencoder.pkl    # Encodes disease labels into numerical format
│   ├── symptom_list.pkl    # List of all input symptoms for model
│   └── predict.py          # Model prediction helper script
├── chatbot/                # RAG-powered chatbot modules
│   ├── chatbot.py          # Main chatbot handler
│   ├── embedder.py         # Embedding utilities
│   ├── gemini_utils.py     # Gemini API helpers
│   ├── pinecone_utils.py   # Pinecone vector store utilities
│   ├── medquad.csv         # Knowledge dataset for RAG
│   └── test.py             # Test script for chatbot
├── static/                 # Static assets
│   ├── css/
│   │   ├── chatbot_result.css
│   │   ├── result.css
│   │   └── style.css
│   └── js/
├── templates/              # HTML templates
│   ├── index.html          # Home page (Symptom Checker)
│   ├── result.html         # Symptom prediction result page
│   ├── chatbot.html        # Chatbot interface
│   └── chatbot_result.html # Chatbot response display page
├── env/                    # Virtual environment (optional)
├── chatbot_env/            # Chatbot-specific environment (optional)
├── .env                    # Environment variables
├── requirements.txt        # Python dependencies
├── train.py                # Model training script
├── temp.py                 # Experimental or temporary scripts
├── README.md               # Project documentation (you are here)
```

---

## Features

* 🎯 **Disease Prediction:** Input symptoms to predict possible diseases using a TensorFlow model.
* 💬 **Medical Chatbot:** Ask medical questions and get LLM-powered responses.
* 🔍 **RAG Integration:** Combines Pinecone vector search + Gemini API for accurate, context-aware Q\&A.
* 🎨 **Clean UI:** HTML + CSS powered web interface with static asset separation.

---

## Installation

```bash
# Create & activate virtual environment (recommended)
python -m venv env
source env/bin/activate  # For Linux/macOS
env\Scripts\activate     # For Windows

# Install dependencies
pip install -r requirements.txt
```

Add your API keys and config variables inside `.env`.

---

## Run the App

```bash
python app.py
```

Open `http://localhost:5000/` in your browser.

---

## File Highlights

* `app.py` → Flask routes for symptom checker & chatbot.
* `chatbot/` → All modules for Pinecone + Gemini based RAG chatbot.
* `model/` → Pre-trained deep learning disease prediction model.
* `templates/` + `static/` → Frontend files (HTML + CSS).

---

## Usage Flow

1. **Symptom Checker**

   * Enter symptoms → Get predicted disease(s).

2. **Medical Chatbot**

   * Ask health-related questions → Get answers powered by LLM + knowledge base.

---

## Requirements

* Python >= 3.8
* Flask
* TensorFlow
* Pinecone
* sentence-transformers
* Gemini API SDK (or Google Generative Language API)

(Full list in `requirements.txt`)

---

## Notes

This project is for research & educational purposes only. Not intended for real medical diagnosis.

---

## License

MIT License
