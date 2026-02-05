# Twitter Disaster Classification Dashboard

A modern desktop application for classifying and analyzing disaster-related tweets using a fine-tuned transformer model with explainable AI (XAI) features.

## Features

- 🔐 **Twitter API Integration**: Login with your Twitter API credentials to fetch tweets
- 🤖 **AI Classification**: Automatically classify tweets into 5 disaster-related categories
- 🎨 **Token Highlighting**: Visualize which tokens are most important for classification using color gradients
- 📊 **Actionable Information Extraction**: Extract locations, people counts, needs, damage types, and time mentions
- 🔍 **Filtering**: Filter tweets by classification category
- 💻 **Modern UI**: Beautiful, dark-themed desktop interface built with CustomTkinter

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download spaCy model (optional, for better location extraction):
```bash
python -m spacy download en_core_web_sm
```

3. Get Twitter API credentials:
   - Go to https://developer.twitter.com/en/portal/dashboard
   - Create a new app and get your API keys
   - You'll need:
     - API Key
     - API Secret
     - Access Token
     - Access Token Secret

## Usage

Run the dashboard:
```bash
python run_dashboard.py
```

1. **Login**: Enter your Twitter API credentials in the login screen
2. **Fetch Tweets**: Click "Fetch Tweets" to load tweets from your timeline
3. **View Classifications**: Tweets are automatically classified and displayed with color-coded labels
4. **Filter**: Use filter buttons to show only specific categories
5. **View Details**: Click on any tweet to see:
   - Full classification details
   - Token-level importance highlighting
   - Extracted actionable information

## Classification Categories

1. **Affected Individuals** (Red) - Tweets about people affected by disasters
2. **Infrastructure Damage** (Orange) - Tweets about damaged infrastructure
3. **Not Humanitarian** (Gray) - Non-relevant tweets
4. **Other Information** (Blue) - Other relevant information
5. **Rescue/Donation** (Green) - Tweets about rescue efforts or donations

## Token Highlighting

When viewing a tweet's details, tokens are highlighted with color gradients:
- **White/Light**: Low importance tokens
- **Red/Dark**: High importance tokens (most influential for classification)

## Actionable Information

For actionable categories (Affected Individuals, Infrastructure Damage, Rescue/Donation), the system extracts:
- 📍 **Locations**: Geographic locations mentioned
- 👥 **People Counts**: Number of people affected (injured, missing, etc.)
- 🆘 **Needs**: Required resources (food, water, medicine, etc.)
- 💥 **Damage Types**: Types of damage mentioned
- ⏰ **Time Mentions**: Temporal information

## Project Structure

```
Dashboard/
├── __init__.py
├── config.py              # Configuration and settings
├── dashboard.py           # Main application UI
├── model_inference.py     # Model loading and inference
├── twitter_api.py         # Twitter API integration
├── token_highlighter.py   # Token highlighting logic
└── README.md
```

## Configuration

Edit `Dashboard/config.py` to:
- Set default API credentials (or use environment variables)
- Customize label colors
- Adjust tweet fetch settings

## Troubleshooting

- **Model not loading**: Ensure `Trained_Model/deltran15_minilm_fp32.pt` exists
- **Twitter API errors**: Verify your API credentials are correct
- **Import errors**: Make sure all dependencies are installed
- **spaCy errors**: Install the English model: `python -m spacy download en_core_web_sm`

## Notes

- The dashboard uses Twitter API v1.1 for tweet fetching
- Rate limits apply based on your Twitter API tier
- Model inference runs on CPU by default (GPU support can be added)

