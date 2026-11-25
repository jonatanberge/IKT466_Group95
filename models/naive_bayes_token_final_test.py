import re
import pandas as pd
import matplotlib.pyplot as plt
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

TOKEN_LENGTHS = [50, 100, 250, 500]
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
ALPHA = 1.0


def tokenize(text):
    return re.findall(r"\b\w+\b|[^\w\s]", str(text).lower())


def truncate_review(text, max_tokens):
    tokens = tokenize(text)
    return " ".join(tokens[:max_tokens])


def fit_final_naive_bayes(train, test, max_tokens):
    train_texts = train["review"].map(lambda text: truncate_review(text, max_tokens))
    test_texts = test["review"].map(lambda text: truncate_review(text, max_tokens))

    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    model = MultinomialNB(alpha=ALPHA)
    model.fit(x_train, train["sentiment"])
    return model, vectorizer, x_test


def main():
    train_df = pd.read_csv(BASE_DIR / "data/splits/train.csv")
    val_df = pd.read_csv(BASE_DIR / "data/splits/val.csv")
    test_df = pd.read_csv(BASE_DIR / "data/splits/test.csv")

    train_df = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    y_test = test_df["sentiment"]

    results = []

    for max_tokens in TOKEN_LENGTHS:
        print(f"\nMaximum tokens: {max_tokens}")
        model, _, x_test = fit_final_naive_bayes(train_df, test_df, max_tokens)
        # Evaluate
        predictions = model.predict(x_test)

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
        results_dir / "naive_bayes_test_results.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))
    plt.plot(
        results_df["max_tokens"],
        results_df["f1_macro"],
        marker="o",
        linewidth=2,
        label="Macro F1"
    )

    plt.title("Naive Bayes Performance by Token Length")
    plt.xlabel("Maximum token length")
    plt.ylabel("Macro F1-score")
    plt.ylim(0, 1)
    plt.xticks(results_df["max_tokens"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        results_dir / "naive_bayes_token_length_plot.png",
        dpi=300
    )
    plt.close()

    print("\nResults and plot saved successfully!")


if __name__ == "__main__":
    main()
