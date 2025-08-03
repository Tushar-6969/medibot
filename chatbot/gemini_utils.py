# to get query + conetxt res from gemini llm 
# chatbot/gemini_utils.py

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load API key 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

def get_gemini_response(user_query, context):
    
    try:
        for i in range(50):
            print("my contex is ")
        print(context)
        print("my query is ",user_query)    
        prompt = f"""
You are a helpful medical assistant. Use the provided context to answer the user's medical question accurately and concisely.

Context:
{context}

User Query:
{user_query}

Answer:"""

        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        print("[❌ ERROR in generate_gemini_response]:", e)
        return "Sorry, Gemini couldn't process the request."
