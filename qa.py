# qa.py

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import requests

load_dotenv()



df = pd.read_pickle("quran_final.pkl")
index = faiss.read_index("quran.index")
model = SentenceTransformer("all-MiniLM-L6-v2")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
print("🔑 Loaded GROQ API Key:", GROQ_API_KEY)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def get_top_k_context(query, k=3):
    embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([embedding]).astype("float32"), k)
    return [df.iloc[i]["text_to_embed"] for i in indices[0]]

def ask_question(question):
    context_passages = get_top_k_context(question)
    context = "\n\n---\n\n".join(context_passages)

    prompt = (
        f"You are an Islamic assistant. Use the following Quranic context to answer the user's question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "You are a knowledgeable Islamic assistant helping users learn from the Quran."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    
    try:
        res_json = response.json()
        # Print the entire response for debugging
        print("🔍 Debug: Full response JSON from GROQ API:", res_json)
        
        if "choices" not in res_json:
            print("❌ Error: 'choices' key not found in the response.")
            return "⚠️ Sorry, something went wrong while getting a response from the model."

        return res_json["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("❌ Exception while parsing response:", e)
        print("🔁 Raw response text:", response.text)
        return "⚠️ Failed to parse the model response."

