import pandas as pd
import re
df = pd.read_csv("../Processed_Data_Set/disaster_text_only.csv")
def clean_tweet(text):
    text = str(text)
    text = re.sub(r'^RT\s+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = text.encode("ascii", "ignore").decode()
    text = re.sub(r'\s+', ' ', text).strip()
    return text
df["clean_text"] = df["text"].apply(clean_tweet)
df = df[df["clean_text"].str.len() > 5].reset_index(drop=True)
print("Samples after cleaning:", len(df))
print("\nExample BEFORE:\n", df["text"].iloc[0])
print("\nExample AFTER:\n", df["clean_text"].iloc[0])
df[["clean_text", "label"]].to_csv(
    "../Processed_Data_Set/disaster_text_clean.csv",
    index=False
)
print("Saved cleaned dataset to Data_Set/Processed_Data_Set/disaster_text_clean.csv")
