# dashboard/utils.py
#from model1.UsingModel import predict, label_map
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model1.UsingModel import predict, label_map
def classify_tweet(text):
    """Wrapper to return structured prediction result."""
    label, scores = predict(text)
    confidence = max(scores)
    return {
        "text": text,
        "label": label_map[label],
        "confidence": confidence
    }