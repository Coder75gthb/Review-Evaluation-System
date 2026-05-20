from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import predict_single
import pandas as pd
import io

app = FastAPI(title="Product Review Analyser")

# =========================================
# CORS — allows Vercel frontend to call
# =========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ReviewInput(BaseModel):
    text: str

# =========================================
# ENDPOINT 1 — Single review
# =========================================
@app.post("/predict")
def predict(review: ReviewInput):
    result = predict_single(review.text)
    return result

# =========================================
# ENDPOINT 2 — Health check
# =========================================
@app.get("/")
def health():
    return {"status": "running"}

# =========================================
# ENDPOINT 3 — Bulk CSV upload
# =========================================
@app.post("/predict-bulk")
async def predict_bulk(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    if 'text' not in df.columns:
        return {"error": "CSV must have a 'text' column"}

    results = []
    for text in df['text'].tolist()[:100]:  # cap at 100
        result = predict_single(str(text))
        result['text'] = text
        results.append(result)

    # Summary stats
    sentiments = [r['sentiment'] for r in results]
    pos_pct = round(sentiments.count('Positive') / len(sentiments) * 100)
    neg_pct = round(sentiments.count('Negative') / len(sentiments) * 100)
    neu_pct = round(sentiments.count('Neutral')  / len(sentiments) * 100)

    return {
        "total_reviews": len(results),
        "summary": {
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct":  neu_pct
        },
        "predictions": results
    }