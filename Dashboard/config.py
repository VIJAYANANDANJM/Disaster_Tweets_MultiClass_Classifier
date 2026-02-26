"""
Configuration file for the Dashboard application.
Store your Twitter API credentials here.
"""
import os

# Twitter/X API v2 Credentials
# Get these from https://developer.twitter.com/en/portal/dashboard
TWITTER_API_KEY = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.getenv("TWITTER_API_SECRET", "")
TWITTER_ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN", "")
TWITTER_ACCESS_TOKEN_SECRET = os.getenv("TWITTER_ACCESS_TOKEN_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# Model paths (relative to project root)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_PATH = "Trained_Model/deltran15_minilm_fp32.pt"
TOKENIZER_PATH = "Trained_Model/Model_Tokenizer"

# Label mapping
LABEL_MAP = {
    0: "affected_individuals",
    1: "infrastructure_and_utility_damage",
    2: "not_humanitarian",
    3: "other_relevant_information",
    4: "rescue_volunteering_or_donation"
}

# Label display names (more user-friendly)
LABEL_DISPLAY_NAMES = {
    0: "Affected Individuals",
    1: "Infrastructure Damage",
    2: "Not Humanitarian",
    3: "Other Information",
    4: "Rescue/Donation"
}

# Label colors for UI
LABEL_COLORS = {
    0: "#FF6B6B",  # Red - urgent
    1: "#FFA500",  # Orange - important
    2: "#95A5A6",  # Gray - not relevant
    3: "#3498DB",  # Blue - informational
    4: "#2ECC71"   # Green - actionable
}

# Actionable labels (show actionable info for these)
ACTIONABLE_LABELS = {0, 1, 4}

# Tweet fetch settings
MAX_TWEETS_PER_FETCH = 50
TWEET_FETCH_INTERVAL = 60  # seconds

