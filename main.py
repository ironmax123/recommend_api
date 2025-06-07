from fastapi import FastAPI
from pydantic import BaseModel
import requests
import random
import uvicorn


app = FastAPI()

class BookRequest(BaseModel):
    title: str
    description: str

@app.post("/recommend")
def recommend_book(book: BookRequest):
    keywords = f"{book.title} {book.description}".split()
    rand_keyword = random.choice(keywords)

    url = f"https://www.googleapis.com/books/v1/volumes?q={rand_keyword}&maxResults=40&language=ja"

    response = requests.get(url)
    data = response.json()

    if "items" in data:
        recommended = random.choice(data["items"])
        return {
            "recommended_title": recommended["volumeInfo"].get("title"),
            "authors": recommended["volumeInfo"].get("authors"),
            "description": recommended["volumeInfo"].get("description"),
            "info_link": recommended["volumeInfo"].get("infoLink")
        }

    return {"message": "おすすめの本が見つかりませんでした。"}

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