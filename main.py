from fastapi import FastAPI
from pydantic import BaseModel
import requests
import random
import uvicorn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from urllib.parse import quote_plus


app = FastAPI()

# Sentence‑BERT モデルを一度だけロードして再利用
model = SentenceTransformer("all-MiniLM-L6-v2")

class BookInfo(BaseModel):
    title: str
    description: str | None = None

class BooksRequest(BaseModel):
    books: list[BookInfo]


@app.post("/recommend")
def recommend_books(payload: BooksRequest):
    """複数の本情報（タイトルと説明）を受け取り、
    Google Books API の候補とベクトル類似度を比較して
    最も近い一冊を返す。
    """
    # --- 1) ユーザ入力ベクトルをまとめて生成 -----------------------------
    texts = [f"{b.title} {b.description or ''}" for b in payload.books]
    user_vectors = model.encode(texts)  # shape = (N_user, dim)

    # --- 2) Google Books API で候補を最大 10 件取得 ------------------------
    #     タイトルと説明を空白区切りで単純連結して検索文字列にする
    query_text = " ".join(texts)
    url = (
        "https://www.googleapis.com/books/v1/volumes?"
        f"q={quote_plus(query_text)}"
        "&maxResults=10"
        "&language=ja"
    )

    response = requests.get(url)
    data = response.json()

    if "items" not in data:
        return {"message": "おすすめの本が見つかりませんでした。"}

    # --- 3) 候補ごとにベクトルを計算し、コサイン類似度 ----------------
    items = data["items"]
    candidate_texts = [
        f"{it.get('volumeInfo', {}).get('title', '')} "
        f"{it.get('volumeInfo', {}).get('description', '')}"
        for it in items
    ]
    cand_vectors = model.encode(candidate_texts)   # shape = (M, dim)

    # --- 4) 類似度行列を計算し，各候補に対して「ユーザ本の中で最も近いスコア」を採用 ---
    sim_matrix = cosine_similarity(user_vectors, cand_vectors)  # (N_user, M)
    sims = np.max(sim_matrix, axis=0)   # shape = (M,)

    # --- 5) スコア 0.6 以上に絞り，その中で最大スコアの候補を選択 ---------------------
    candidate_idxs = np.where(sims >= 0.6)[0]
    if candidate_idxs.size == 0:
        # しきい値を満たす候補がなければ全体から最大値
        best_idx = int(np.argmax(sims))
    else:
        best_idx = int(candidate_idxs[np.argmax(sims[candidate_idxs])])
    recommended = items[best_idx]
    info = recommended.get("volumeInfo", {})

    return {
        "recommended_title": info.get("title"),
        "authors": info.get("authors"),
        "description": info.get("description"),
        "thumbnail": info.get("imageLinks", {}).get("thumbnail"),
        "similarity_score": float(sims[best_idx]),
    }

# --- Run the FastAPI app locally -----------------------------
# This allows you to start the server with `python main.py`
# or continue to use `uvicorn main:app --reload` if you prefer.
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,   # auto‑reload on code changes (development use)
    )