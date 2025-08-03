
#takes user queery -> searches Pinecone for context -> 
# -> sends everything to the Gemini API -> returns the final answer.
# chatbot/chatbot.py

import traceback
from sentence_transformers import SentenceTransformer
from .pinecone_utils import query_pinecone
from .gemini_utils import get_gemini_response  

# Load embedding model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

def chatbot_handler(user_query):
    try:
        print("====== TESTING Chatbot (Medical Q&A) ======")
        print(" User Query:", user_query)

        # Convert query to embedding
        embedded_query = model.encode([user_query])[0].tolist()

        # Query Pinecone
        context_chunks = query_pinecone(embedded_query, top_k=5, namespace="medquad")

        if not context_chunks:
            print("[⚠️] No context found.")
            return "Sorry, I couldn't find any relevant information."

        # Prepare context
        context = "\n\n".join(context_chunks[:3])
        print("[ Retrieved Context]:\n", context)

        # Generate final response using Gemini
        final_answer = get_gemini_response(user_query, context)
        print("[ Gemini Final Answer]:\n", final_answer)

        return final_answer

    except Exception as e:
        print("[❌ ERROR in chatbot_handler]:", e)
        traceback.print_exc()
        return "Sorry, something went wrong while processing your query."
