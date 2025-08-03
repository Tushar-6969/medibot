# 1. connects to pinecone 
# 2. retrieve relvent query from pinecone 
# chatbot/pinecone_utils.py

import os
import traceback
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "medibot")

# Create Pinecone client instance
pc = Pinecone(api_key=PINECONE_API_KEY)

def init_pinecone():
    try:
        print("[DEBUG] Initializing Pinecone...")

        index_names = pc.list_indexes().names()
        if PINECONE_INDEX not in index_names:
            raise Exception(f"Index '{PINECONE_INDEX}' not found. Available: {index_names}")

        print(f"[DEBUG] Pinecone index '{PINECONE_INDEX}' is available.")

    except Exception as e:
        print("[ERROR in init_pinecone]:", e)
        traceback.print_exc()
        raise

def query_pinecone(embedded_query, top_k=3, namespace=None):
    try:
        print("[DEBUG] Querying Pinecone...")
        index = pc.Index(PINECONE_INDEX)

        response = index.query(
            vector=embedded_query,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace if namespace else None
        )

        if not isinstance(response, dict):
            response = response.to_dict()
        print("[DEBUG] Raw Pinecone response:", response)

        matches = response.get("matches", [])
        print("[DEBUG] Extracted matches:", matches)

        context_chunks = [
            match.get("metadata", {}).get("answer", "")
            for match in matches
            if isinstance(match, dict) and "metadata" in match
        ]
        print("[DEBUG] Extracted context_chunks:", context_chunks)
        return context_chunks

    except Exception as e:
        print("[ERROR in query_pinecone]:", e)
        traceback.print_exc()
        return []
