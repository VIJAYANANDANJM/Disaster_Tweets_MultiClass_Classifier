import torch
import os
import sys
from transformers import AutoTokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Explainable_AI import explain_prediction
from Model import DeLTran15   

# -------------------------------
# 1. CONFIG
# -------------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "deltran15_minilm_fp32.pt"
TOKENIZER_PATH = "Model_Tokenizer"

MODEL_FILE = os.path.join(current_dir, MODEL_PATH)
TOKENIZER_DIR = os.path.join(current_dir, TOKENIZER_PATH)

# -------------------------------
# 2. LOAD TOKENIZER
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

# -------------------------------
# 3. LOAD MODEL
# -------------------------------
model = DeLTran15(MODEL_NAME, local_files_only=True)
model.load_state_dict(
    torch.load(MODEL_FILE, map_location="cpu")
)
model.to("cpu")
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
        if probs.device.type == "meta":
            raise RuntimeError("Model output is on the meta device; weights were not materialized correctly.")
        pred = int(torch.argmax(probs, dim=1).detach().cpu().item())

    return pred, probs.squeeze(0).detach().cpu().tolist()

# -------------------------------
# 6. TEST
# -------------------------------
if __name__ == "__main__":
    while(True):
        text = input("Enter text to classify (or 'exit' to quit): ")
        label, confidence = predict(text)

        explanation, probs = explain_prediction(
        model,
        tokenizer,
        text,
        target_class=label)
        SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]"}
        print("\nTop influential tokens:")
        for token, score in sorted(explanation, key=lambda x: x[1], reverse=True)[:10]:
            if token not in SPECIAL_TOKENS:
                print(token, ":", round(float(score), 4))

        print("Predicted label:", label)
        print("Class name:", label_map[label])
        print("Confidence:", confidence)
        if text.lower() == 'exit':
            break
    
