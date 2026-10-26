import pandas as pd

df = pd.read_csv("IMDB Dataset.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSentiment distribution:")
print(df["sentiment"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate reviews:")
print(df["review"].duplicated().sum())

df["review_length"] = df["review"].str.split().str.len()

print("\nReview length statistics:")
print(df["review_length"].describe())