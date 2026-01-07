import torch
from transformers import AutoTokenizer
from model import DeLTran15

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("modeltokenizer")

# Load model
model = DeLTran15(MODEL_NAME)
model.load_state_dict(
    torch.load("deltran15_minilm_fp32.pt", map_location="cpu")
)
model.eval()
