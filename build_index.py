# build_index.py

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

df = pd.read_pickle("quran.pkl")

df["text_to_embed"] = (
    "Surah: " + df["surah_number"].astype(str) + ", Ayah: " + df["ayah_number"].astype(str) + "\n"
    + "Arabic: " + df["sentence"] + "\n"
    + "Translation: " + df["translation"] + "\n"
    + "Tafsir: " + df["en-tafsir-mokhtasar-html"]
)

texts = df["text_to_embed"].tolist()

print("📦 Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # Fast and good

embeddings = model.encode(texts, show_progress_bar=True)

# Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Save everything
faiss.write_index(index, "quran.index")
df.to_pickle("quran_final.pkl")
print("✅ Saved quran.index and quran_final.pkl")
