# =============================================================
# train_model.py — Run once locally BEFORE deploying to Render
# =============================================================
# Usage: python train_model.py
#
# This script trains the TF-IDF + Logistic Regression model on
# the CLINC150 dataset and saves it to futo_model.pkl.
# Render will then just LOAD the pkl — no training at startup,
# which keeps memory usage well under the 512MB free tier limit.
#
# Steps:
#   1. Make sure data_full.csv is in the same folder as this file
#   2. Run: python train_model.py
#   3. Confirm futo_model.pkl was created
#   4. Commit futo_model.pkl to your GitHub repo
#   5. Deploy to Render — done!
# =============================================================

import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

CSV_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_full.csv")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "futo_model.pkl")


def train_and_save():
    print("=" * 55)
    print("  FUTO Chatbot — Model Trainer")
    print("=" * 55)

    # ── Load dataset ─────────────────────────────────────────
    print("\n[1/4] Loading CLINC150 dataset …")
    df       = pd.read_csv(CSV_PATH)
    train_df = df[(df["split"] == "train") & (df["intent"].astype(str) != "0")].copy()
    val_df   = df[df["split"] == "val"].copy()

    print(f"      Train samples : {len(train_df):,}")
    print(f"      Val   samples : {len(val_df):,}")
    print(f"      Unique intents: {train_df['intent'].nunique()}")

    # ── Encode labels ─────────────────────────────────────────
    print("\n[2/4] Encoding intent labels …")
    le = LabelEncoder()
    le.fit(train_df["intent"].astype(str))

    X_train = train_df["text"].astype(str).tolist()
    y_train = le.transform(train_df["intent"].astype(str))

    # ── Build & train pipeline ────────────────────────────────
    print("\n[3/4] Training TF-IDF + Logistic Regression …")
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),       # unigrams + bigrams
            min_df=2,                 # ignore very rare words
            sublinear_tf=True,        # log-scale TF
            strip_accents="unicode",
        )),
        ("clf", LogisticRegression(
            C=4.0,
            max_iter=500,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)
    print("      Training complete!")

    # ── Validate ──────────────────────────────────────────────
    acc = 0.0
    try:
        val_texts   = val_df["text"].astype(str).tolist()
        val_intents = val_df["intent"].astype(str).tolist()
        X_v = [x for x, s in zip(val_texts, val_intents) if s in le.classes_]
        y_v = le.transform([s for s in val_intents if s in le.classes_])
        acc = pipeline.score(X_v, y_v)
        print(f"      Validation accuracy: {acc:.2%}")
    except Exception as e:
        print(f"      Validation skipped: {e}")

    # ── Save to pkl ───────────────────────────────────────────
    print(f"\n[4/4] Saving model to {MODEL_PATH} …")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "pipeline"      : pipeline,
            "label_encoder" : le,
            "accuracy"      : acc,
        }, f)

    size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    print(f"      File size: {size_mb:.1f} MB")
    print("\n✅ Done! Commit futo_model.pkl to your repo and deploy.")
    print("=" * 55)


if __name__ == "__main__":
    train_and_save()