# Use Python 3.10 to create a fresh virtual environment
py -3.10 -m venv chatbot_env
chatbot_env\Scripts\activate

# Install correct versions
pip install --upgrade pip
pip install sentence-transformers==2.2.2 transformers==4.30.2




---
emb.py=
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm
from dotenv import load_dotenv

# Load env variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = os.getenv("INDEX_NAME")

if not all([PINECONE_API_KEY, PINECONE_ENV, INDEX_NAME]):
    raise ValueError("Missing environment variables!")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index(INDEX_NAME)

# Load data
df = pd.read_csv("medquad.csv")

# Build combined text
def build_text(row):
    if "context" in row and pd.notna(row["context"]):
        return f"{row['question']} [CTX] {row['context']}"
    return row["question"]

texts = [build_text(row) for _, row in df.iterrows()]

# Delete old vectors in chunks
all_ids = [f"med-{i}" for i in range(len(texts))]
chunk_size = 1000
for i in tqdm(range(0, len(all_ids), chunk_size), desc="Deleting old vectors"):
    index.delete(ids=all_ids[i:i+chunk_size])

# Load embedding model
model = SentenceTransformer("paraphrase-MiniLM-L3-v2")

# Upload with metadata
batch_size = 64
for i in tqdm(range(0, len(texts), batch_size), desc="Uploading to Pinecone"):
    batch_texts = texts[i:i+batch_size]
    embeddings = model.encode(batch_texts).tolist()
    ids = [f"med-{i+j}" for j in range(len(batch_texts))]

    # Create vector list with metadata
    vectors = [
        {
            "id": ids[j],
            "values": embeddings[j],
            "metadata": {
                "text": batch_texts[j]
            }
        }
        for j in range(len(batch_texts))
    ]

    # Upsert in chunks of 1000
    for j in range(0, len(vectors), 1000):
        index.upsert(vectors=vectors[j:j+1000])

print("✅ All embeddings uploaded with metadata!")

