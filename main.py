from fastapi import FastAPI
from pydantic import BaseModel
import requests
import random
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import quote_plus
from sklearn.feature_extraction.text import CountVectorizer
from janome.tokenizer import Tokenizer



app = FastAPI()

# Sentence‑BERT モデルを一度だけロードして再利用
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")

class BookInfo(BaseModel):
    title: str
    description: str | None = None

class BooksRequest(BaseModel):
    books: list[BookInfo]
  


@app.post("/api/v1/recommend")
def recommend_books(payload: BooksRequest):
    texts = [f"{b.title} {b.description or ''}" for b in payload.books]
    full_text = " ".join(texts)
    
    # 1. 形態素解析で名詞のみ抽出
    t = Tokenizer()
    words = [token.surface for token in t.tokenize(full_text) if token.part_of_speech.startswith('名詞')]
    words = list(set(words))
    if len(words) < 2:
        return {"message": "特徴語が抽出できません"}

    # 2. 全体ベクトル化
    doc_vector = model.encode([full_text])[0]

    # 3. 各単語をベクトル化→コサイン類似度
    word_vectors = model.encode(words)
    sim_scores = cosine_similarity([doc_vector], word_vectors)[0]

    # 4. 上位5語
    top_idx = sim_scores.argsort()[-5:][::-1]
    keywords = [words[i] for i in top_idx]

    # 5. Google Books API検索
    query = "+".join(keywords)
    print(query)
    url = (
        "https://www.googleapis.com/books/v1/volumes?"
        f"q={quote_plus(query)}"
        "&maxResults=10"
        "&language=ja"
    )
    response = requests.get(url)
    if response.status_code != 200:
        return {"message": "Google Books APIリクエスト失敗", "detail": response.text}

    try:
        data = response.json()
    except Exception as e:
        return {"message": f"JSONデコード失敗: {e}", "detail": response.text}

    if "items" not in data or not data["items"]:
        return {"message": "おすすめの本が見つかりませんでした。"}

    # 一番上の本を返す
    recommended = data["items"][0]
    info = recommended.get("volumeInfo", {})

    return {
        "recommended_title": info.get("title"),
        "authors": info.get("authors"),
        "description": info.get("description"),
        "thumbnail": info.get("imageLinks", {}).get("thumbnail"),
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )