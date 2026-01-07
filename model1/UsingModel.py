import torch
from transformers import AutoTokenizer
from model import DeLTran15   # your model class file

# -------------------------------
# 1. CONFIG
# -------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "deltran15_minilm_fp32.pt"
TOKENIZER_PATH = "modeltokenizer"

# -------------------------------
# 2. LOAD TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
model = DeLTran15(MODEL_NAME)
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu")
)
model.eval()

# -------------------------------
# 4. LABEL MAP
# -------------------------------
label_map = {
    0: "affected_individuals",
    1: "infrastructure_and_utility_damage",
    2: "not_humanitarian",
    3: "other_relevant_information",
    4: "rescue_volunteering_or_donation"
}

# -------------------------------
# 5. PREDICTION FUNCTION
# -------------------------------
def predict(text):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(
            enc["input_ids"],
            enc["attention_mask"]
        )
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    return pred, probs.squeeze().tolist()

# -------------------------------
# 6. TEST
# -------------------------------
if __name__ == "__main__":
    while(True):
        text = input("Enter text to classify (or 'exit' to quit): ")
        label, confidence = predict(text)

        print("Predicted label:", label)
        print("Class name:", label_map[label])
        print("Confidence:", confidence)
        if text.lower() == 'exit':
            break
    
