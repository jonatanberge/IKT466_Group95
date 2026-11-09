import re
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

train_df = pd.read_csv(BASE_DIR / "data/splits/train.csv")
val_df = pd.read_csv(BASE_DIR / "data/splits/val.csv")
test_df = pd.read_csv(BASE_DIR / "data/splits/test.csv")

train_df = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

y_train = train_df["sentiment"]
y_test = test_df["sentiment"]

TOKEN_LENGTHS = [50, 100, 250, 500]


def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower())


def truncate_review(text, max_tokens):
    tokens = tokenize(text)
    return " ".join(tokens[:max_tokens])


results = []

for max_tokens in TOKEN_LENGTHS:

    print(f"\nMaximum tokens: {max_tokens}")

    # Truncate training and validation reviews
    train_texts = train_df["review"].apply(
        lambda text: truncate_review(text, max_tokens)
    )

    test_texts = test_df["review"].apply(
        lambda text: truncate_review(text, max_tokens)
    )

    # Create a new TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2)
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    # Train Naive Bayes
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train, y_train)

    # Evaluate
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, average="macro"
    )
    recall = recall_score(
        y_test, predictions, average="macro"
    )
    f1 = f1_score(
        y_test, predictions, average="macro"
    )

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Macro F1:  {f1:.4f}")

    results.append({
    "max_tokens": max_tokens,
    "accuracy": accuracy,
    "precision_macro": precision,
    "recall_macro": recall,
    "f1_macro": f1
})
    
# Save experiment results
results_dir = BASE_DIR / "results" / "naive_bayes"
results_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    results_dir / "token_length_test_results.csv",
    index=False
)

print("\nResults saved successfully!")