from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import predict_single
import pandas as pd
import io
import os
from groq import Groq

app = FastAPI(title="Product Review Analyser")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# =========================================
# GROQ CLIENT
# =========================================
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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
# ENDPOINT 3 — Dynamic insights
# =========================================
@app.post("/insights")
def get_insights(review: ReviewInput):
    result = predict_single(review.text)
    sentiment = result['sentiment']
    rating    = result['predicted_rating']
    probs     = result['probabilities']

    prompt=f"""A customer wrote this review: "{review.text}"

The AI model predicted:
- Sentiment: {sentiment}
- Predicted Rating: {rating}/5
- Positive probability: {probs['positive']}%
- Neutral probability: {probs['neutral']}%
- Negative probability: {probs['negative']}%

You are an expert customer sentiment analyst. Your job is to understand
what a customer TRULY means, not just what they literally say.

Before classifying anything, reason through these questions internally:
- What is the customer's overall emotional state?
- What would this customer tell a friend about this product?
- If you were the product owner, what would concern or please you here?
- Are there any indirect, implied, or hedged messages hidden in polite language?

Using that reasoning, extract:

1. POSITIVE_THEMES: Aspects the customer is genuinely satisfied with
2. NEUTRAL_THEMES: Aspects the customer feels indifferent or conflicted about
3. NEGATIVE_THEMES: Aspects the customer is dissatisfied with, even if expressed politely or indirectly
4. POSITIVE_SUMMARY: One sentence capturing genuine praise
5. NEUTRAL_SUMMARY: One sentence capturing mixed feelings
6. NEGATIVE_SUMMARY: One sentence capturing real concerns, stated OR implied
7. RECOMMENDATION: One actionable business recommendation

If no aspects exist for a category, write "None identified".
Respond in exactly this format, nothing else:
POSITIVE_THEMES: theme1, theme2, theme3
NEUTRAL_THEMES: theme1, theme2, theme3
NEGATIVE_THEMES: theme1, theme2, theme3
POSITIVE_SUMMARY: sentence here
NEUTRAL_SUMMARY: sentence here
NEGATIVE_SUMMARY: sentence here
RECOMMENDATION: sentence here"""


    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300
    )

    raw = response.choices[0].message.content.strip()

    lines = {}
    for line in raw.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            lines[key.strip()] = val.strip()

    return {
        "sentiment":         sentiment,
        "rating":            rating,
        "positive_themes":   [t.strip() for t in lines.get('POSITIVE_THEMES', 'Quality').split(',')],
        "neutral_themes":    [t.strip() for t in lines.get('NEUTRAL_THEMES',  'Features').split(',')],
        "negative_themes":   [t.strip() for t in lines.get('NEGATIVE_THEMES', 'Issues').split(',')],
        "positive_summary":  lines.get('POSITIVE_SUMMARY', ''),
        "neutral_summary":   lines.get('NEUTRAL_SUMMARY',  ''),
        "negative_summary":  lines.get('NEGATIVE_SUMMARY', ''),
        "recommendation":    lines.get('RECOMMENDATION',   '')
    }

# =========================================
# ENDPOINT 4 — Bulk CSV
# =========================================
@app.post("/predict-bulk")
async def predict_bulk(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

    if 'text' not in df.columns:
        return {"error": "CSV must have a 'text' column"}

    results = []
    for text in df['text'].tolist()[:100]:
        result = predict_single(str(text))
        result['text'] = text
        results.append(result)

    sentiments  = [r['sentiment'] for r in results]
    pos_pct     = round(sentiments.count('Positive') / len(sentiments) * 100)
    neg_pct     = round(sentiments.count('Negative') / len(sentiments) * 100)
    neu_pct     = round(sentiments.count('Neutral')  / len(sentiments) * 100)
    avg_rating  = round(sum(r['predicted_rating'] for r in results) / len(results), 1)

    sample_texts = df['text'].tolist()[:20]

    bulk_prompt = f"""You are a product review analyst.

Here are {len(results)} product reviews with these results:
- {pos_pct}% Positive, {neu_pct}% Neutral, {neg_pct}% Negative
- Average predicted rating: {avg_rating}/5

Sample reviews:
{chr(10).join(['- ' + str(t)[:100] for t in sample_texts])}

Provide insights in exactly this format:
POSITIVE_THEMES: theme1, theme2, theme3
NEUTRAL_THEMES: theme1, theme2, theme3
NEGATIVE_THEMES: theme1, theme2, theme3
POSITIVE_SUMMARY: sentence here
NEUTRAL_SUMMARY: sentence here
NEGATIVE_SUMMARY: sentence here
RECOMMENDATION: sentence here"""

    bulk_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": bulk_prompt}],
        temperature=0.3,
        max_tokens=300
    )

    bulk_raw = bulk_response.choices[0].message.content.strip()
    bulk_lines = {}
    for line in bulk_raw.split('\n'):
        if ':' in line:
            key, val = line.split(':', 1)
            bulk_lines[key.strip()] = val.strip()

    return {
        "total_reviews": len(results),
        "summary": {
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct":  neu_pct,
            "avg_rating":   avg_rating
        },
        "insights": {
            "positive_themes":  [t.strip() for t in bulk_lines.get('POSITIVE_THEMES', 'Quality').split(',')],
            "neutral_themes":   [t.strip() for t in bulk_lines.get('NEUTRAL_THEMES',  'Features').split(',')],
            "negative_themes":  [t.strip() for t in bulk_lines.get('NEGATIVE_THEMES', 'Issues').split(',')],
            "positive_summary": bulk_lines.get('POSITIVE_SUMMARY', ''),
            "neutral_summary":  bulk_lines.get('NEUTRAL_SUMMARY',  ''),
            "negative_summary": bulk_lines.get('NEGATIVE_SUMMARY', ''),
            "recommendation":   bulk_lines.get('RECOMMENDATION',   '')
        },
        "predictions": results
    }
