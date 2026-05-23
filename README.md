# Review Intelligence

An AI powered product review evaluation system that predicts sentiment, estimates star ratings, and generates actionable business insights using BERT and LLM technology.

## Live Demo
Frontend: https://v0-product-review-dashboard-amber.vercel.app/
Backend API: https://huggingface.co/spaces/legend75/product-review-analyser

## What It Does
Upload any product review (single or bulk CSV) and get:
- Sentiment prediction (Positive, Neutral, Negative)
- Predicted star rating (1 to 5)
- Confidence score with probability breakdown
- AI generated insights on what customers love, dislike, and feel neutral about
- Actionable business recommendations
- Downloadable report

## Tech Stack
- NLP Model: 2 stage BERT (bert-base-uncased)
- Rating Prediction: XGBoost with BERT CLS embeddings
- LLM Insights: Groq (llama-3.1-8b-instant)
- Backend: FastAPI deployed on HuggingFace Spaces
- Frontend: Next.js with Tailwind CSS deployed on Vercel

## Model Performance
- Sentiment Classification Macro F1: 0.73
- Rating Prediction RMSE: 0.71
- Overall Accuracy: 87%
- Training Data: 37,559 product reviews (10:1 class imbalance handled)

## Architecture
User Input
|
FastAPI Backend (HuggingFace)
|
Stage 1: BERT (Negative vs Non-negative)
|
Stage 2: BERT (Neutral vs Positive)
|
XGBoost Rating Predictor (BERT embeddings + TF-IDF)
|
Groq LLM Insights Generator
|
Next.js Frontend (Vercel)

## Project Structure
app/
main.py         FastAPI endpoints
model.py        BERT and XGBoost inference
Dockerfile
requirements.txt

## Key Features
- 2 stage hierarchical BERT pipeline for handling class imbalance
- Focal loss to handle 10:1 positive to neutral ratio
- CLS token embeddings combined with TF-IDF SVD features
- Dynamic LLM powered insights per review or per CSV batch
- Single review and bulk CSV evaluation modes

