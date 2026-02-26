"""
Token Highlighting Module
Handles highlighting tokens in text based on XAI scores with color gradients.
"""
import re
from typing import List, Tuple


class TokenHighlighter:
    """Handles token-level highlighting with color gradients."""
    
    def __init__(self, min_color="#FFFFFF", max_color="#FF0000"):
        """
        Initialize highlighter.
        
        Args:
            min_color: Color for lowest importance (hex)
            max_color: Color for highest importance (hex)
        """
        self.min_color = min_color
        self.max_color = max_color
    
    def hex_to_rgb(self, hex_color):
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def rgb_to_hex(self, rgb):
        """Convert RGB tuple to hex color."""
        return '#%02x%02x%02x' % tuple(int(c) for c in rgb)
    
    def interpolate_color(self, ratio):
        """
        Interpolate between min and max colors based on ratio.
        
        Args:
            ratio: Value between 0 and 1
            
        Returns:
            str: Hex color string
        """
        ratio = max(0, min(1, ratio))  # Clamp between 0 and 1
        
        min_rgb = self.hex_to_rgb(self.min_color)
        max_rgb = self.hex_to_rgb(self.max_color)
        
        r = int(min_rgb[0] + (max_rgb[0] - min_rgb[0]) * ratio)
        g = int(min_rgb[1] + (max_rgb[1] - min_rgb[1]) * ratio)
        b = int(min_rgb[2] + (max_rgb[2] - min_rgb[2]) * ratio)
        
        return self.rgb_to_hex((r, g, b))
    
    def normalize_scores(self, scores):
        """
        Normalize scores to 0-1 range.
        
        Args:
            scores: List of score values
            
        Returns:
            list: Normalized scores
        """
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [0.5] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]
    
    def create_highlighted_text(self, text: str, explanation: List[Tuple[str, float]], 
                                special_tokens: set = None):
        """
        Create highlighted text with token scores.
        
        Args:
            text: Original text
            explanation: List of (token, score) tuples from XAI
            special_tokens: Set of special tokens to skip
            
        Returns:
            list: List of (text_segment, color, score) tuples for rendering
        """
        if special_tokens is None:
            special_tokens = {"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
        
        # Filter out special tokens and get scores
        token_score_map = {}
        scores_list = []
        
        # Build token to score mapping, handling subword tokens
        current_token = ""
        for token, score in explanation:
            if token in special_tokens:
                continue
            
            if token.startswith('##'):
                # Subword token - append to current token
                current_token += token[2:]
            else:
                # New token
                if current_token:
                    clean_token = current_token.lower().strip()
                    if clean_token and clean_token not in token_score_map:
                        token_score_map[clean_token] = score
                        scores_list.append(score)
                current_token = token.lower().strip()
                if current_token and current_token not in token_score_map:
                    token_score_map[current_token] = score
                    scores_list.append(score)
        
        # Handle last token
        if current_token and current_token not in token_score_map:
            token_score_map[current_token] = explanation[-1][1] if explanation else 0.0
            scores_list.append(token_score_map[current_token])
        
        if not scores_list or not token_score_map:
            return [(text, "#000000", 0.0)]
        
        # Normalize scores
        min_score = min(scores_list)
        max_score = max(scores_list)
        if max_score != min_score:
            token_score_map = {
                k: (v - min_score) / (max_score - min_score)
                for k, v in token_score_map.items()
            }
        else:
            token_score_map = {k: 0.5 for k in token_score_map}
        
        # Find tokens in text and create segments
        segments = []
        text_lower = text.lower()
        processed_ranges = []  # List of (start, end) tuples
        
        # Sort tokens by length (longest first) to avoid partial matches
        sorted_tokens = sorted(token_score_map.keys(), key=len, reverse=True)
        
        # Find all token positions
        token_positions = {}  # token -> list of (start, end) positions
        for token in sorted_tokens:
            positions = []
            start = 0
            while True:
                pos = text_lower.find(token, start)
                if pos == -1:
                    break
                # Check if it's a word boundary match
                if (pos == 0 or not text[pos-1].isalnum()) and \
                   (pos + len(token) >= len(text) or not text[pos + len(token)].isalnum()):
                    positions.append((pos, pos + len(token)))
                start = pos + 1
            if positions:
                token_positions[token] = positions
        
        # Build segments by processing text character by character
        i = 0
        while i < len(text):
            # Find the highest scoring token at this position
            best_token = None
            best_score = -1
            best_end = i + 1
            
            for token, positions in token_positions.items():
                for start, end in positions:
                    if start == i:
                        score = token_score_map[token]
                        if score > best_score:
                            best_score = score
                            best_token = token
                            best_end = end
            
            if best_token is not None:
                # Add highlighted token
                token_text = text[i:best_end]
                color = self.interpolate_color(best_score)
                segments.append((token_text, color, best_score))
                i = best_end
            else:
                # Find next token start or end of text
                next_token_start = len(text)
                for positions in token_positions.values():
                    for start, end in positions:
                        if start > i and start < next_token_start:
                            next_token_start = start
                
                # Add plain text segment
                plain_text = text[i:next_token_start]
                if plain_text:
                    segments.append((plain_text, "#000000", 0.0))
                i = next_token_start
        
        # If no segments were created, return original text
        if not segments:
            return [(text, "#000000", 0.0)]
        
        return segments

