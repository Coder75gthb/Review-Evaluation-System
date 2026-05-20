import torch
import pickle
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cpu")  # HuggingFace free tier = CPU

# =========================================
# LOAD ALL MODELS ONCE AT STARTUP
# =========================================
print("Loading models...")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model1 = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)
model1.load_state_dict(
    torch.load('models/best_model1.pt', map_location=device)
)
model1.eval()

model2 = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2
)
model2.config.hidden_dropout_prob = 0.15
model2.config.attention_probs_dropout_prob = 0.15
model2.load_state_dict(
    torch.load('models/best_model2.pt', map_location=device)
)
model2.eval()

xgb     = pickle.load(open('models/final_xgb.pkl',    'rb'))
scaler  = pickle.load(open('models/final_scaler.pkl', 'rb'))
tfidf   = pickle.load(open('models/final_tfidf.pkl',  'rb'))
svd     = pickle.load(open('models/final_svd.pkl',    'rb'))

print("✅ All models loaded!")

# =========================================
# PREDICT SINGLE REVIEW
# =========================================
def predict_single(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        probs1 = torch.softmax(
            model1(**inputs).logits, dim=1
        )
        probs2 = torch.softmax(
            model2(**inputs).logits, dim=1
        )

    neg_prob     = probs1[0][0].item()
    neutral_prob = probs2[0][0].item()
    pos_prob     = probs2[0][1].item()

    # Sentiment decision
    if neg_prob >= 0.65:
        sentiment = 0
        sentiment_label = "Negative"
        probs = [neg_prob, 0.0, 0.0]
    elif neutral_prob >= 0.30:
        sentiment = 1
        sentiment_label = "Neutral"
        probs = [0.0, neutral_prob, pos_prob]
    elif (pos_prob - neutral_prob) >= 0.45:
        sentiment = 2
        sentiment_label = "Positive"
        probs = [0.0, neutral_prob, pos_prob]
    else:
        sentiment = 1
        sentiment_label = "Neutral"
        probs = [0.0, neutral_prob, pos_prob]

    # Rating prediction
    neg_p, neu_p, pos_p = probs
    rating_est = (neg_p * 1.5) + (neu_p * 3.0) + (pos_p * 4.5)
    pos_neg_margin = pos_p - neg_p
    confidence = max(probs)

    sent_features = np.array([[
        neg_p, neu_p, pos_p,
        rating_est,
        pos_neg_margin,
        pos_p - neu_p,
        confidence
    ]])

    tfidf_feat = tfidf.transform([text])
    svd_feat   = svd.transform(tfidf_feat)

    X_combined = np.hstack([svd_feat, sent_features])
    X_scaled   = scaler.transform(X_combined)

    rating = float(np.clip(xgb.predict(X_scaled)[0], 1, 5))

    return {
        "sentiment": sentiment_label,
        "sentiment_score": sentiment,
        "predicted_rating": round(rating, 1),
        "confidence": round(confidence * 100, 1),
        "probabilities": {
            "negative": round(neg_prob * 100, 1),
            "neutral":  round(neutral_prob * 100, 1),
            "positive": round(pos_prob * 100, 1)
        }
    }