from datasets import load_dataset
import pandas as pd

# Load the dataset from Hugging Face
dataset = load_dataset("M-AI-C/quran-en-tafssirs")
data = dataset["train"]

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Keep only needed columns
df = df[["en-ahmedali", "ayah", "sorah", "sentence", "en-tafsir-mokhtasar-html"]]

# Rename for consistency
df.rename(columns={"en-ahmedali": "translation", "sorah": "surah_number", "ayah": "ayah_number", "combined_text": "en-tafsir-mokhtasar-html"}, inplace=True)

# Save to disk
df.to_csv("quran_dataset.csv", index=False)
df.to_pickle("quran.pkl")

print("✅ Quran dataset saved to data/quran_dataset.csv and data/quran.pkl")
