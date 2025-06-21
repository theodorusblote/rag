import os
from datetime import datetime
import uuid  # create unique IDs for vector-store rows

import chromadb
from openai import OpenAI

from rag.settings import OPENAI_API_KEY
from .emb_utils import Embedder

# Initialise global helpers
embedder = Embedder()  # Spins up MiniLM model

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Suppress HuggingFace tokenizer warnings

SYSTEM_PROMPT = (
    "You are Monum, a helpful assistant. Use the context provided to answer as a helpful coach."
)

# Persistent vector store
collection = chromadb.PersistentClient(path="chroma_db").get_or_create_collection(name="memory")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


def save(text: str, role: str):
    emb = embedder.embed([text])[0]  # 384-dim float list, unit-norm
    doc_id = uuid.uuid4().hex  # 32-char, collision-safe primary key
    meta = {"role": role, "ts": datetime.now().isoformat()}
    collection.add(
        documents=[text],
        embeddings=[emb],
        ids=[doc_id],
        metadatas=[meta])


def search(query: str, k: int = 4):
    q_emb = embedder.embed([query])[0]  # embed the query
    res = collection.query(
        query_embeddings=[q_emb],  # must be list-of-lists
        n_results=k)  # top-k
    return res["documents"][0]  # return top-k results (list[str])


def chat():
    print("Hi, I'm Monum (type '/exit' to quit)")
    while True:
        if (user_input := input(": ")).lower() == "/exit":
            print("Session ended")
            break
        save(user_input, "user")
        context_docs = search(user_input)  # List[str] up to k
        context_block = "\n".join(context_docs)  # newline-separated passage
        prompt = [
            {
                "role": "system", "content": SYSTEM_PROMPT + "\n\n" + context_block
            },
            {
                "role": "user", "content": user_input
            }
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=prompt
        ).choices[0].message.content
        print(response)
        save(response, "assistant")


if __name__ == "__main__":
    try:
        chat()
    except KeyboardInterrupt:
        print("\nSession ended")
