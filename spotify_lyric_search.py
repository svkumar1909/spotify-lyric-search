import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------- LOAD DATA ----------------
df = pd.read_csv("spotify_songs.csv")
df = df[["artist", "song", "text"]].dropna().reset_index(drop=True)

# ---------------- TEXT CLEANING ----------------
def preprocess_lyrics(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)

df["processed_lyrics"] = df["text"].apply(preprocess_lyrics)

# ---------------- TRAIN / TEST SPLIT ----------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# ---------------- VECTORIZE ----------------
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=7000
)

X_train = vectorizer.fit_transform(train_df["processed_lyrics"])
X_test = vectorizer.transform(test_df["processed_lyrics"])

# ---------------- PREDICTION FUNCTION ----------------
def search_song(lyrics_snippet, top_k=3):
    snippet_clean = preprocess_lyrics(lyrics_snippet)
    snippet_vec = vectorizer.transform([snippet_clean])

    similarity_scores = cosine_similarity(snippet_vec, X_train)[0]
    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "song": train_df.iloc[idx]["song"],
            "artist": train_df.iloc[idx]["artist"],
            "similarity": round(similarity_scores[idx], 3)
        })

    return results

# ---------------- DEMO ----------------
query = "hello darkness my old friend"
results = search_song(query)

print(f"\nInput Lyrics: \"{query}\"\n")
for r in results:
    print(f"{r['song']} - {r['artist']} (score: {r['similarity']})")
