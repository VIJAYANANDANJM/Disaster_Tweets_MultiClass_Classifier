"""
Model Inference Module
Handles loading the trained model and providing predictions, XAI explanations, and actionable info extraction.
"""
import torch
import sys
import os
from transformers import AutoTokenizer

# Add project root to path to import Trained_Model modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from Trained_Model.Model import DeLTran15
from Trained_Model.Explainable_AI import explain_prediction
from Trained_Model.Actionable_Info import extract_actionable_info
from Dashboard.config import MODEL_NAME, MODEL_PATH, TOKENIZER_PATH, LABEL_MAP, ACTIONABLE_LABELS


class ModelInference:
    """Wrapper class for model inference, XAI, and actionable info extraction."""
    
    def __init__(self):
        """Initialize and load the model."""
        self.model = None
        self.tokenizer = None
        self.loaded = False
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Get absolute paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, MODEL_PATH)
            tokenizer_path = os.path.join(project_root, TOKENIZER_PATH)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            
            # Load model
            self.model = DeLTran15(MODEL_NAME)
            self.model.load_state_dict(
                torch.load(model_path, map_location="cpu")
            )
            self.model.eval()
            self.loaded = True
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.loaded = False
    
    def predict(self, text):
        """
        Predict the class for given text.
        
        Args:
            text: Input text string
            
        Returns:
            tuple: (predicted_label_id, confidence_scores_list, label_name)
        """
        if not self.loaded:
            return None, None, None
        
        try:
            enc = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            with torch.no_grad():
                logits = self.model(
                    enc["input_ids"],
                    enc["attention_mask"]
                )
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
            
            confidence_scores = probs.squeeze().tolist()
            label_name = LABEL_MAP.get(pred, "unknown")
            
            return pred, confidence_scores, label_name
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None, None
    
    def get_explanation(self, text, target_class=None):
        """
        Get XAI explanation with token scores.
        
        Args:
            text: Input text string
            target_class: Optional target class ID (uses predicted if None)
            
        Returns:
            tuple: (explanation_list, probability_array)
            explanation_list: List of (token, score) tuples
        """
        if not self.loaded:
            return None, None
        
        try:
            explanation, probs = explain_prediction(
                self.model,
                self.tokenizer,
                text,
                target_class=target_class
            )
            return explanation, probs
        except Exception as e:
            print(f"Error in explanation: {e}")
            return None, None
    
    def get_actionable_info(self, text, label_id):
        """
        Extract actionable information from text.
        
        Args:
            text: Input text string
            label_id: Predicted label ID
            
        Returns:
            dict: Actionable information or None if label is not actionable
        """
        if label_id not in ACTIONABLE_LABELS:
            return None
        
        try:
            return extract_actionable_info(text)
        except Exception as e:
            print(f"Error extracting actionable info: {e}")
            return None
    
    def classify_tweet(self, text):
        """
        Complete classification pipeline: prediction + XAI + actionable info.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Complete classification results
        """
        pred_id, confidences, label_name = self.predict(text)
        
        if pred_id is None:
            return None
        
        explanation, probs = self.get_explanation(text, target_class=pred_id)
        actionable_info = self.get_actionable_info(text, pred_id)
        
        return {
            "predicted_label_id": pred_id,
            "predicted_label": label_name,
            "confidence_scores": confidences,
            "explanation": explanation,
            "actionable_info": actionable_info
        }

