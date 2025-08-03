# to convert knowledge base in to vector and uploading in pinocone sdk(3+) 
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
from dotenv import load_dotenv

print("[DEBUG] Initializing components...")

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

if not all([PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME]):
    raise ValueError("❌ Missing environment variables!")

print("[DEBUG] Initializing Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)
print(f"[DEBUG] Pinecone index '{INDEX_NAME}' is ready.")

# Load CSV
df = pd.read_csv("medquad.csv")
print(f"[DEBUG] Loaded {len(df)} rows from medquad.csv")

# Combine question and context
def build_text(row):
    question = str(row.get("question", "")).strip()
    context = str(row.get("context", "")).strip()
    if context:
        return f"Question: {question} [CTX] {context}"
    return question

texts = [build_text(row) for _, row in df.iterrows()]

# Delete old vectors
all_ids = [f"med-{i}" for i in range(len(texts))]
chunk_size = 1000
for i in tqdm(range(0, len(all_ids), chunk_size), desc="Deleting old vectors"):
    index.delete(ids=all_ids[i:i + chunk_size])

# Load SentenceTransformer model
print("[DEBUG] Loading SentenceTransformer model...")
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
print("[DEBUG] Model loaded successfully.")

# Upload in batches
batch_size = 64
for i in tqdm(range(0, len(texts), batch_size), desc="Uploading to Pinecone"):
    batch_texts = texts[i:i + batch_size]
    embeddings = model.encode(batch_texts).tolist()
    ids = [f"med-{i + j}" for j in range(len(batch_texts))]

    vectors = []
    for j, text in enumerate(batch_texts):
        row = df.iloc[i + j]
        metadata = {
            "text": text,
            "question": str(row.get("question", "")).strip(),
            "context": str(row.get("context", "")).strip(),
            "answer": str(row.get("answer", "")).strip()
        }
        vectors.append({
            "id": ids[j],
            "values": embeddings[j],
            "metadata": metadata
        })

    # index.upsert(vectors=vectors)
    index.upsert(vectors=vectors, namespace="medquad")


print(" All vectors uploaded with full metadata.")
