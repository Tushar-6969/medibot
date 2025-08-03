# to test pinecone working + gemini response 
# chatbot/test.py

from chatbot import chatbot_handler

if __name__ == "__main__":
    test_query = "What are the symptoms of dengue?"
    print("====== TESTING Chatbot (Medical Q&A) ======")
    response = chatbot_handler(test_query)
    print("[✅] Chatbot response:\n", response)
