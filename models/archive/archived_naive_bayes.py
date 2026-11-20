import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt

# Find the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the prepared datasets
train_df = pd.read_csv(BASE_DIR / "data/splits/train.csv")
val_df = pd.read_csv(BASE_DIR / "data/splits/val.csv")

print("Training samples:", len(train_df))
print("Validation samples:", len(val_df))

print("\nExample review:")
print(train_df["review"].iloc[0])

print("\nSentiment:")
print(train_df["sentiment"].iloc[0])

#TF-IDF is the feature extraction technique, it prepares the text for the classifies Naive Bayes.

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=10000
)

# Convert reviews into numerical features
X_train = vectorizer.fit_transform(train_df["review"])
X_val = vectorizer.transform(val_df["review"])

print("\nTF-IDF transformation complete!")
print("Training matrix shape:", X_train.shape)
print("Validation matrix shape:", X_val.shape)


# Naive Bayes Classifier

# Extract sentiment labels
y_train = train_df["sentiment"]
y_val = val_df["sentiment"]

# Create the Naive Bayes classifier
model = MultinomialNB(alpha=1.0)

# Train the model
model.fit(X_train, y_train)

print("\nNaive Bayes training complete!")

# Predict sentiments for validation reviews
predictions = model.predict(X_val)

print("\nFirst 10 predictions:")
print(predictions[:10])

print("\nFirst 10 actual labels:")
print(y_val.iloc[:10].values)


# Evaluate the model

# Calculate validation accuracy
accuracy = accuracy_score(y_val, predictions)

print("\nValidation accuracy:", accuracy)

# Print detailed classification metrics
print("\nClassification report:")
print(classification_report(
    y_val,
    predictions,
    target_names=["Negative", "Positive"]
))

# Calculate confusion matrix
cm = confusion_matrix(y_val, predictions)

print("\nConfusion matrix:")
print(cm)


# Analyze incorrect predictions

# Create a copy of the validation dataset
errors_df = val_df.copy()

# Add model predictions
errors_df["predicted"] = predictions

# Keep only incorrect predictions
errors_df = errors_df[
    errors_df["sentiment"] != errors_df["predicted"]
]

print("\nNumber of incorrect predictions:", len(errors_df))

# Display the first 5 incorrect predictions
for i, (_, row) in enumerate(errors_df.head(5).iterrows(), start=1):
    print(f"\n--- Incorrect prediction {i} ---")
    print("Actual sentiment:", row["sentiment"])
    print("Predicted sentiment:", row["predicted"])
    print("Review:", row["review"][:500])


# Review length

# Create a DataFrame for review length analysis
length_df = val_df.copy()

# Add predictions
length_df["predicted"] = predictions

# Count words in each validation review
length_df["word_count"] = length_df["review"].str.split().str.len()

# Calculate length thresholds from training data
train_lengths = train_df["review"].str.split().str.len()

short_threshold = train_lengths.quantile(0.25)
long_threshold = train_lengths.quantile(0.75)

print("\nReview length thresholds:")
print("Short:", short_threshold)
print("Long:", long_threshold)


# Categorize reviews based on word count
def categorize_length(word_count):
    if word_count <= short_threshold:
        return "Short"
    elif word_count <= long_threshold:
        return "Medium"
    else:
        return "Long"


length_df["length_category"] = length_df["word_count"].apply(
    categorize_length
)


print("\nNaive Bayes accuracy by review length:")

for category in ["Short", "Medium", "Long"]:
    subset = length_df[
        length_df["length_category"] == category
    ]

    accuracy = accuracy_score(
        subset["sentiment"],
        subset["predicted"]
    )

    print(
        f"{category}: "
        f"{accuracy:.4f} "
        f"({len(subset)} reviews)"
    )


categories = ["Short", "Medium", "Long"]
accuracies = []

for category in categories:
    subset = length_df[
        length_df["length_category"] == category
    ]

    accuracy = accuracy_score(
        subset["sentiment"],
        subset["predicted"]
    )

    accuracies.append(accuracy)

plt.figure(figsize=(8, 5))

plt.bar(categories, accuracies)

plt.xlabel("Review Length")
plt.ylabel("Accuracy")
plt.title("Naive Bayes Accuracy by Review Length")

plt.ylim(0, 1)

plt.tight_layout()
plt.show()