
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

# --------------------------------------------------
# 1. Load datasets
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

train_df = pd.read_csv(BASE_DIR / "data/splits/train.csv")
val_df = pd.read_csv(BASE_DIR / "data/splits/val.csv")

y_train = train_df["sentiment"]
y_val = val_df["sentiment"]

# --------------------------------------------------
# 2. Tokenization and truncation
# --------------------------------------------------

MAX_TOKENS = 500


def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower())


def truncate_review(text, max_tokens):
    tokens = tokenize(text)
    return " ".join(tokens[:max_tokens])


print("Preparing reviews...")

train_texts = train_df["review"].apply(
    lambda text: truncate_review(text, MAX_TOKENS)
)

val_texts = val_df["review"].apply(
    lambda text: truncate_review(text, MAX_TOKENS)
)

# --------------------------------------------------
# 3. Hyperparameters to test
# --------------------------------------------------

ALPHA_VALUES = [0.1, 1.0, 2.0, 5.0]
MAX_FEATURES_VALUES = [5000, 10000, 20000]
NGRAM_RANGES = [(1, 1), (1, 2)]

results = []

# --------------------------------------------------
# 4. Hyperparameter tuning
# --------------------------------------------------

for max_features in MAX_FEATURES_VALUES:
    for ngram_range in NGRAM_RANGES:

        print(
            f"\nTF-IDF: max_features={max_features}, "
            f"ngram_range={ngram_range}"
        )

        # Fit TF-IDF only on training data
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)

        for alpha in ALPHA_VALUES:

            # Train Naive Bayes
            model = MultinomialNB(alpha=alpha)
            model.fit(X_train, y_train)

            # Predict validation labels
            predictions = model.predict(X_val)

            # Calculate evaluation metrics
            accuracy = accuracy_score(y_val, predictions)
            precision = precision_score(
                y_val, predictions, average="macro"
            )
            recall = recall_score(
                y_val, predictions, average="macro"
            )
            f1 = f1_score(
                y_val, predictions, average="macro"
            )

            # Save results for this configuration
            results.append({
                "max_tokens": MAX_TOKENS,
                "alpha": alpha,
                "max_features": max_features,
                "ngram_range": str(ngram_range),
                "accuracy": accuracy,
                "precision_macro": precision,
                "recall_macro": recall,
                "f1_macro": f1
            })

            print(
                f"Alpha={alpha:<4} | "
                f"Accuracy={accuracy:.4f} | "
                f"Macro F1={f1:.4f}"
            )

# --------------------------------------------------
# 5. Find the best configuration
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="f1_macro",
    ascending=False
)

print("\n--- TOP 5 CONFIGURATIONS ---")

print(
    results_df[
        [
            "alpha",
            "max_features",
            "ngram_range",
            "accuracy",
            "f1_macro"
        ]
    ].head(5).to_string(index=False)
)

best = results_df.iloc[0]

print("\n--- BEST CONFIGURATION ---")
print(f"Max tokens:   {int(best['max_tokens'])}")
print(f"Alpha:        {best['alpha']}")
print(f"Max features: {int(best['max_features'])}")
print(f"N-gram range: {best['ngram_range']}")
print(f"Accuracy:     {best['accuracy']:.4f}")
print(f"Macro F1:     {best['f1_macro']:.4f}")

# --------------------------------------------------
# 6. Save results to CSV
# --------------------------------------------------

results_dir = BASE_DIR / "results" / "naive_bayes"
results_dir.mkdir(parents=True, exist_ok=True)

output_path = results_dir / "hyperparameter_tuning.csv"

results_df.to_csv(output_path, index=False)

print(f"\nResults saved to: {output_path}")
