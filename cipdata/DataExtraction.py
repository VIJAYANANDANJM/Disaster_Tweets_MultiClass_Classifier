import pandas as pd
import glob
import os

# =========================
# LABEL MAPPING (FINAL)
# =========================
label_map = {
    "injured_or_dead_people": 0,
    "affected_individuals": 0,  # merged category
    "infrastructure_and_utility_damage": 1,
    "vehicle_damage": 1,        # merged as per paper
    "not_humanitarian": 2,
    "other_relevant_information": 3,
    "rescue_volunteering_or_donation_effort": 4
}

texts = []
labels = []

# =========================
# LOAD ALL TSV FILES
# =========================
files = glob.glob("*.tsv")

for file in files:
    df = pd.read_csv(file, sep="\t", header=None)

    # Drop header row
    df = df.iloc[1:].reset_index(drop=True)

    tweet_texts = df.iloc[:, 12]   # tweet text
    class_labels = df.iloc[:, 6]   # humanitarian label

    for text, label in zip(tweet_texts, class_labels):
        if pd.notna(text) and pd.notna(label):
            label = label.strip()
            if label in label_map:
                texts.append(text.strip())
                labels.append(label_map[label])

print("✅ Total usable samples:", len(texts))
print("🔹 Sample text:", texts[0])
print("🔹 Sample label:", labels[0])


processed_df = pd.DataFrame({
    "text": texts,
    "label": labels
})

os.makedirs("cipdata/processed", exist_ok=True)
processed_df.to_csv(
    "cipdata/processed/disaster_text_only.csv",
    index=False
)

print("✅ Saved cleaned dataset to cipdata/processed/disaster_text_only.csv")

