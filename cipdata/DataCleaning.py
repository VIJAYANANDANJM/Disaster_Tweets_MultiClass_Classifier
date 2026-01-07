import pandas as pd
import re

# =========================
# LOAD PROCESSED DATA
# =========================
df = pd.read_csv("processed/disaster_text_only.csv")

# =========================
# CLEANING FUNCTION
# =========================
def clean_tweet(text):
    text = str(text)

    # Remove RT
    text = re.sub(r'^RT\s+', '', text)

    # Remove mentions
    text = re.sub(r'@\w+', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove non-ASCII chars
    text = text.encode("ascii", "ignore").decode()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# =========================
# APPLY CLEANING
# =========================
df["clean_text"] = df["text"].apply(clean_tweet)

# Drop empty texts after cleaning
df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)

print("✅ Samples after cleaning:", len(df))
print("\n🔹 Example BEFORE:\n", df["text"].iloc[0])
print("\n🔹 Example AFTER:\n", df["clean_text"].iloc[0])


df[["clean_text", "label"]].to_csv(
    "processed/disaster_text_clean.csv",
    index=False
)

print("✅ Saved cleaned dataset to cipdata/processed/disaster_text_clean.csv")
