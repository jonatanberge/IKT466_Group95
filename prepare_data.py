import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


df = pd.read_csv("IMDB Dataset.csv")

print("Original dataset size:", len(df))


# Check whether any identical review has conflicting labels
conflicts = df.groupby("review")["sentiment"].nunique()
conflict_count = (conflicts > 1).sum()

if conflict_count:
    raise ValueError(
        f"Found {conflict_count} reviews with conflicting labels. "
        "Inspect these before splitting."
    )

df = df.drop_duplicates(subset=["review"]).copy()

print("Dataset size after removing duplicates:", len(df))


df["sentiment"] = df["sentiment"].map({
    "negative": 0,
    "positive": 1
})

assert df["sentiment"].notna().all()


# First split: 60% training, 40% temporary
train_df, temp_df = train_test_split(
    df,
    test_size=0.4,
    random_state=42,
    stratify=df["sentiment"]
)

# Second split: 20% validation, 20% testing
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["sentiment"]
)


output_dir = Path("data/splits")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "val.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)


print("\nDataset splits:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nTraining class distribution:")
print(train_df["sentiment"].value_counts())

print("\nValidation class distribution:")
print(val_df["sentiment"].value_counts())

print("\nTesting class distribution:")
print(test_df["sentiment"].value_counts())

print("\nFiles saved in data/splits/")