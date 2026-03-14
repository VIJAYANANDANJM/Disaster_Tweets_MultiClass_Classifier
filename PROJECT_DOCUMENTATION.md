# Project Documentation — Disaster Tweet Multi-Class Classification System

> This document provides an exhaustive, in-depth explanation of every component in the system. It is intended for developers, reviewers, and academics who need to understand how the entire project works from data ingestion to final visualization.

---

## Table of Contents

1. [Project Purpose & Motivation](#1-project-purpose--motivation)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Data Pipeline — From Raw Tweets to Clean Data](#3-data-pipeline--from-raw-tweets-to-clean-data)
4. [The DeLTran15 Model — In Depth](#4-the-deltran15-model--in-depth)
5. [Explainable AI (XAI) — How It Works Internally](#5-explainable-ai-xai--how-it-works-internally)
6. [Actionable Information Extraction — NLP Pipeline](#6-actionable-information-extraction--nlp-pipeline)
7. [Model Inference Module — The Complete Pipeline](#7-model-inference-module--the-complete-pipeline)
8. [Backend Server — Node.js/Express/MongoDB](#8-backend-server--nodejsexpressmongodb)
9. [Desktop Dashboard — Python/CustomTkinter UI](#9-desktop-dashboard--pythoncustomtkinter-ui)
10. [Human-in-the-Loop (HITL) Verification](#10-human-in-the-loop-hitl-verification)
11. [Geospatial Temporal Aggregation — Complete Reference](#11-geospatial-temporal-aggregation--complete-reference)
12. [Configuration System](#12-configuration-system)
13. [Threading & Concurrency Model](#13-threading--concurrency-model)
14. [File-by-File Reference](#14-file-by-file-reference)

---

## 1. Project Purpose & Motivation

### The Real-World Problem

During natural disasters (earthquakes, floods, hurricanes, wildfires), millions of tweets are posted on social media. Emergency responders need to answer critical questions **within minutes**:

- **Where** are people trapped?
- **What** infrastructure is damaged?
- **What** supplies do people need?
- **Who** is offering help?

Manual reading of thousands of tweets is **impossible** at disaster scale. A binary "disaster / not disaster" classifier doesn't tell responders what **kind** of information the tweet contains. They need **multi-class categorization**.

### Our Solution

This system provides:

1. **5-Class Classification**: Categorizes tweets into 5 humanitarian categories so responders know exactly what type of information each tweet contains.
2. **Explainable AI**: Shows **why** the model made each prediction through token-level visual highlighting, building trust with human operators.
3. **Actionable Intelligence Extraction**: Automatically extracts locations, people counts, needs, damage types, and time references from classified tweets.
4. **Geospatial Aggregation**: Groups tweets by location and produces **situation reports per location**, not just per tweet.
5. **Human-in-the-Loop**: Low-confidence predictions are flagged for human review, creating a feedback loop.

---

## 2. High-Level Architecture

### Design Philosophy: Privacy-First

The system follows a **decoupled architecture** where:
- **Data storage** is centralized (MongoDB — either local or Atlas cloud)
- **AI inference** runs entirely on the user's local machine
- **No external AI APIs** are called (no OpenAI, no Google, no cloud ML)

This means sensitive tweet data and classification results **never leave the user's machine** for AI processing.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER'S LOCAL MACHINE                        │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Desktop Dashboard (Python)                     │   │
│  │  ┌──────────┐ ┌──────────────┐ ┌──────────────────────┐ │   │
│  │  │dashboard │ │model_inference│ │geospatial_aggregator │ │   │
│  │  │.py       │ │.py           │ │.py                   │ │   │
│  │  │          │ │              │ │                      │ │   │
│  │  │ Main UI  │ │ DeLTran15   │ │ LocationResolver     │ │   │
│  │  │ Tabs     │ │ XAI Engine  │ │ GeoSpatialAggregator │ │   │
│  │  │ HITL     │ │ Actionable  │ │ 10-Case Decision     │ │   │
│  │  │ Filters  │ │ Info Extract│ │ Severity Scoring     │ │   │
│  │  └────┬─────┘ └──────┬──────┘ └──────────────────────┘ │   │
│  │       │               │                                  │   │
│  │  ┌────┴───────────────┴──────────────────────────────┐  │   │
│  │  │         api_client.py (HTTP REST Client)           │  │   │
│  │  └────────────────────────┬───────────────────────────┘  │   │
│  └───────────────────────────│──────────────────────────────┘   │
│                              │ HTTP                             │
│  ┌───────────────────────────│──────────────────────────────┐   │
│  │  Trained Model Files      │                               │   │
│  │  ├── deltran15_minilm_fp32.pt (90 MB model weights)      │   │
│  │  ├── Model.py (DeLTran15 architecture)                    │   │
│  │  ├── Explainable_AI.py (gradient-based XAI)               │   │
│  │  ├── Actionable_Info.py (spaCy NER + regex)               │   │
│  │  └── Model_Tokenizer/ (HuggingFace tokenizer)            │   │
│  └───────────────────────────────────────────────────────────┘   │
└──────────────────────────────│───────────────────────────────────┘
                               │
                     HTTP REST API (localhost:5000)
                               │
┌──────────────────────────────│───────────────────────────────────┐
│  Backend Server (Node.js / Express)                              │
│  ├── POST /api/tweets/create       ← Store tweets               │
│  ├── GET  /api/tweets              → Fetch tweets (paginated)    │
│  ├── PUT  /api/tweets/:id/classify ← Store classification       │
│  ├── GET  /api/tweets/geo-aggregate→ Geo aggregation             │
│  └── MongoDB ← Persistent storage for all tweet data            │
└──────────────────────────────────────────────────────────────────┘
```

### Data Flow (End to End)

```
1. Tweet enters the system
   ├── Manual input by user in Dashboard text box
   ├── Fetched from MongoDB via "Refresh Database"
   └── Loaded from mock dataset (mock_geo_tweets.py)

2. Text goes to model_inference.py
   ├── Tokenized by HuggingFace tokenizer (all-MiniLM-L6-v2)
   ├── Forward pass through DeLTran15 → 5 probability scores
   ├── Softmax → pick highest probability class
   ├── Gradient backward pass → token importance scores (XAI)
   └── spaCy NER + regex → actionable info extraction

3. Results returned to Dashboard
   ├── Classification: label ID, label name, confidence scores
   ├── Explanation: list of (token, importance_score) tuples
   └── Actionable Info: locations, needs, damage, people, time

4. Dashboard renders results
   ├── Color-coded category label
   ├── Token highlighting (white → red heat map)
   ├── Actionable info cards
   └── HITL buttons if low-confidence

5. Results synced to MongoDB via api_client.py
   └── PUT /api/tweets/:id/classify sends classification back to backend

6. Geo Analysis (optional)
   ├── Fetches ALL tweets from MongoDB
   ├── Classifies any unclassified tweets
   ├── Groups by location → clusters
   ├── Runs 6-step pipeline per cluster
   └── Renders cluster reports in Geo Analysis view
```

---

## 3. Data Pipeline — From Raw Tweets to Clean Data

### Stage 1: Raw Data Collection

Raw disaster tweet datasets are stored in `Data_Set/Unprocessed_Data_Sets/`. These come from publicly available disaster tweet datasets and contain noisy, unprocessed text.

### Stage 2: Data Extraction (`Data_Extraction.py`)

This script reads raw source files and extracts the relevant columns (text, label) into a standardized format.

### Stage 3: Data Cleaning (`Data_Cleaning.py`)

The cleaning pipeline applies these transformations in order:

1. **URL Removal**: Strip all `http://` and `https://` links
2. **Special Character Removal**: Remove `@mentions`, `#hashtags` (keep the word), and other non-alphanumeric characters
3. **Contraction Expansion**: "don't" → "do not", "can't" → "cannot"
4. **Emoji Handling**: Convert emojis to their text descriptions
5. **Whitespace Normalization**: Collapse multiple spaces, strip leading/trailing whitespace
6. **Case Normalization**: Convert to lowercase for consistency

### Stage 4: Labeling

Data is manually categorized into the 5 classes:

| Label ID | Category | Description |
|---|---|---|
| 0 | Affected Individuals | People hurt, trapped, displaced, missing |
| 1 | Infrastructure & Utility Damage | Roads, bridges, power, water damaged |
| 2 | Not Humanitarian | Irrelevant content (sports, food, casual chat) |
| 3 | Other Relevant Information | Warnings, updates, general awareness |
| 4 | Rescue/Volunteering/Donation | Help offers, donations, volunteer coordination |

### Stage 5: Output

Clean, labeled CSVs are stored in `Data_Set/Processed_Data_Set/`, ready for model training.

---

## 4. The DeLTran15 Model — In Depth

### Why This Architecture?

We needed a model that:
- Understands **semantic meaning** in short texts (tweets are typically <280 characters)
- Is **computationally lightweight** enough to run locally without a GPU
- Can be **fine-tuned** on our specific 5-class disaster taxonomy
- Produces **embeddings** that can be used for gradient-based XAI

**`sentence-transformers/all-MiniLM-L6-v2`** fits all criteria: 22M parameters, 6 transformer layers, 384-dimension embeddings, optimized for semantic similarity in short texts.

### Architecture Definition

**File:** `Trained_Model/Model.py`

```python
class DeLTran15(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super().__init__()
        # Pre-trained transformer encoder (all-MiniLM-L6-v2)
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Regularization dropout (30% during training)
        self.dropout = nn.Dropout(0.3)
        
        # Classification head: 384-dim embedding → 5 classes
        self.classifier = nn.Linear(
            self.encoder.config.hidden_size,  # 384
            num_classes                        # 5
        )

    def forward(self, input_ids, attention_mask):
        # Step 1: Get transformer output for all tokens
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Step 2: Extract [CLS] token embedding (sentence representation)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        
        # Step 3: Apply dropout for regularization
        x = self.dropout(cls_embedding)
        
        # Step 4: Linear projection to 5 classes (logits)
        return self.classifier(x)
```

### How Classification Works (Step by Step)

```
Input: "30 people trapped after bridge collapse in Manila"

Step 1 — Tokenization:
  tokenizer("30 people trapped after bridge collapse in Manila")
  → input_ids: [101, 2382, 2111, 7416, 2044, 2958, 9253, 1999, 10023, 102]
  → attention_mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  (101 = [CLS], 102 = [SEP], others = word pieces)

Step 2 — Transformer Encoding:
  Each token is converted to a 384-dimensional embedding by the encoder.
  The [CLS] token's embedding captures the overall sentence meaning.
  
  [CLS] embedding: [0.234, -0.567, 0.891, ..., 0.123]  (384 numbers)

Step 3 — Dropout:
  During training, 30% of the embedding values are randomly zeroed.
  During inference (prediction), dropout is disabled.

Step 4 — Linear Classification:
  The 384-dim embedding is multiplied by a 384×5 weight matrix:
  logits = W × embedding + bias
  → logits: [1.2, 4.8, -0.5, 2.1, 0.9]

Step 5 — Softmax:
  Convert logits to probabilities (sum to 1.0):
  softmax([1.2, 4.8, -0.5, 2.1, 0.9])
  → [0.04, 0.85, 0.01, 0.06, 0.04]
  
  Label 1 (Infrastructure Damage) has 85% probability → PREDICTION
```

### Model File

**`Trained_Model/deltran15_minilm_fp32.pt`** (90 MB)

This file contains the learned weights (parameters) of the model after fine-tuning. It includes:
- The frozen pre-trained transformer encoder weights (from MiniLM)
- The fine-tuned classification head weights (learned on our disaster data)
- Saved in PyTorch's `state_dict` format (FP32 precision)

### Training Process

**File:** `Model_Build/Build.py`

1. Load clean CSVs from `Data_Set/Processed_Data_Set/`
2. Create train/validation/test splits
3. Tokenize all texts using `all-MiniLM-L6-v2` tokenizer
4. Fine-tune the model using cross-entropy loss and AdamW optimizer
5. Save best checkpoint to `deltran15_minilm_fp32.pt`

---

## 5. Explainable AI (XAI) — How It Works Internally

### The Problem with Black-Box AI

If the model says a tweet is "Infrastructure Damage" with 85% confidence, how does the operator know the model isn't just guessing? They need to see **which words** the model focused on to make that decision.

### Our Approach: Gradient-Based Token Importance

**File:** `Trained_Model/Explainable_AI.py`

We use a technique that measures how much each token's embedding "contributed" to the final prediction. The algorithm:

```python
def explain_prediction(model, tokenizer, text, target_class=None, max_len=128):
    model.eval()

    # 1. Tokenize the input
    enc = tokenizer(text, truncation=True, padding=True,
                    max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    # 2. Forward pass with hidden states output
    outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True   # ← Need hidden states for explanation
    )

    # 3. Get CLS embedding and compute logits
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    logits = model.classifier(cls_embedding)
    probs = torch.softmax(logits, dim=1)

    # 4. If no target class specified, use the predicted class
    if target_class is None:
        target_class = torch.argmax(probs, dim=1)

    # 5. Backward pass — compute gradients w.r.t. input embeddings
    model.zero_grad()
    logits[0, target_class].backward()

    # 6. Token importance = L2 norm of each token's embedding
    token_embeddings = outputs.last_hidden_state[0]
    token_importance = torch.norm(token_embeddings, dim=1)

    # 7. Map tokens to importance scores
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    explanation = list(zip(tokens, token_importance.detach().cpu().numpy()))

    return explanation, probs.detach().cpu().numpy()
```

### How the Math Works

For each token position `i`:
- The transformer produces a 384-dimensional embedding vector: `e_i = [e_i1, e_i2, ..., e_i384]`
- The **L2 norm** of this vector measures its "magnitude": `||e_i|| = sqrt(e_i1² + e_i2² + ... + e_i384²)`
- After the backward pass, the gradient signal flows through the embeddings
- Tokens with **higher norms** had more influence on the classification decision

### Visual Output in the Dashboard

**File:** `Dashboard/token_highlighter.py`

The importance scores are mapped to a **white → red color gradient**:
- **White** (low score): Token had little influence → "after", "the", "in"
- **Light pink**: Moderate influence → "Manila", "people"
- **Deep red** (high score): Token strongly influenced prediction → "trapped", "bridge", "collapse"

The dashboard renders each token as a colored label widget, creating a visual heat map that shows exactly what the model "looked at."

---

## 6. Actionable Information Extraction — NLP Pipeline

### Purpose

Once we know a tweet is about a disaster (labels 0, 1, or 4), the next question is: **what actionable intelligence is in the text?** This module extracts structured data from the unstructured tweet text.

**File:** `Trained_Model/Actionable_Info.py`

### Extraction Methods

#### 1. Location Extraction (spaCy NER + Regex Fallback)

**Primary method:** spaCy's `en_core_web_sm` model identifies named entities of type GPE (geopolitical entity), LOC (location), and FAC (facility).

```python
doc = spacy_nlp("Flooding in Manila Bay near Tondo district")
# Entities: Manila Bay (GPE), Tondo (GPE)
```

**Fallback method** (if spaCy is unavailable): Regex patterns catch:
- Preposition + Capitalized words: `"in Manila Bay"` → `"Manila Bay"`
- City, State format: `"Houston, TX"` → `"Houston, TX"`
- Highway format: `"Highway 59"`, `"I-45"` → extracted as locations

#### 2. People Count Extraction (Regex)

Patterns match common casualty phrasings:

```
"30 people trapped"           → {count: 30, status: "trapped"}
"at least 12 dead"            → {count: 12, status: "dead"}
"over 100 injured"            → {count: 100, status: "injured"}
"approximately 50 missing"    → {count: 50, status: "missing"}
```

#### 3. Needs Extraction (Keyword Matching)

A predefined set of need-related keywords is checked against the tweet:

```python
NEEDS_KEYWORDS = {
    "food", "water", "medicine", "medical", "shelter", "blankets",
    "clothes", "rescue", "ambulance", "evacuation", "supplies",
    "aid", "donation"
}
```

If the tweet text contains any of these words, they're extracted as identified needs.

#### 4. Damage Type Extraction (Keyword Matching)

```python
DAMAGE_KEYWORDS = {
    "bridge", "road", "highway", "power", "electricity", "collapsed",
    "flooded", "fire", "wildfire", "damaged", "destroyed", "outage",
    "blocked", "landslide"
}
```

#### 5. Time Mention Extraction (Regex)

```python
TIME_PATTERNS = [
    r"\bnow\b", r"\btoday\b", r"\byesterday\b", r"\btonight\b",
    r"\bimmediately\b", r"\burgently\b", r"\basap\b",
    r"\bthis\s+morning\b", r"\bthis\s+afternoon\b",
    r"\b\d+\s*(mins?|hours?)\s*ago\b"
]
```

### Complete Pipeline Example

```
Input: "30 people trapped after bridge collapse near Manila Bay, 
        need water and food NOW"

Output:
{
    "locations": ["Manila Bay"],
    "people_count": [{"count": 30, "status": "trapped"}],
    "needs": ["food", "water"],
    "damage_type": ["bridge", "collapsed"],
    "time_mentions": ["NOW"]
}
```

---

## 7. Model Inference Module — The Complete Pipeline

**File:** `Dashboard/model_inference.py`

This module is the **central orchestrator** that ties together the DeLTran15 model, XAI, and actionable info extraction into a single pipeline.

### Class: `ModelInference`

```
ModelInference
├── __init__()           → Initialize, set up paths
├── _load_model()        → Load DeLTran15 weights + tokenizer
├── predict(text)        → Run model → (label_id, scores[], label_name)
├── get_explanation(text) → Run XAI → [(token, score), ...]
├── get_actionable_info(text, label_id)  → Run NER → {locations, needs, ...}
├── classify_tweet(text)  → FULL PIPELINE → {prediction + XAI + actionable}
└── classify_tweet_cluster(tweets, location) → Geo cluster classification
```

### `classify_tweet(text)` — The Complete Pipeline

This is the method called for every tweet. Here's exactly what happens:

```python
def classify_tweet(self, text):
    # 1. Model prediction
    label_id, scores, label_name = self.predict(text)
    # label_id = 1, scores = [0.04, 0.85, 0.01, 0.06, 0.04]
    # label_name = "infrastructure_and_utility_damage"
    
    # 2. XAI explanation  
    explanation, probs = self.get_explanation(text)
    # explanation = [("[CLS]", 0.1), ("30", 0.3), ("people", 0.6),
    #                ("trapped", 0.9), ("after", 0.2), ("bridge", 0.8),
    #                ("collapse", 0.7), ("in", 0.1), ("manila", 0.5)]
    
    # 3. Actionable info (only for labels 0, 1, 4)
    actionable_info = self.get_actionable_info(text, label_id)
    # actionable_info = {"locations": ["Manila"], "needs": ["rescue"],
    #                     "damage_type": ["bridge", "collapsed"], ...}
    
    # 4. Return complete result
    return {
        "predicted_label_id": 1,
        "predicted_label": "infrastructure_and_utility_damage",
        "confidence_scores": [0.04, 0.85, 0.01, 0.06, 0.04],
        "explanation": [("[CLS]", 0.1), ("30", 0.3), ...],
        "actionable_info": {"locations": ["Manila"], ...}
    }
```

### Normalization

The module includes `_normalize_actionable_info()` which converts between snake_case (Python convention) and camelCase (JavaScript/MongoDB convention):

```python
# Python side:      damage_type, people_count, time_mentions
# Backend side:     damageType,  peopleCount,  timeMentions
```

This ensures data formats are consistent between the Python dashboard and the Node.js backend.

---

## 8. Backend Server — Node.js/Express/MongoDB

### Purpose

The backend is a **pure data storage and retrieval layer**. It does NOT run any AI inference — it stores tweets and their classification results in MongoDB, and provides REST API endpoints for the dashboard to interact with.

### Tweet Schema (`backend/models/Tweet.js`)

```javascript
const TweetSchema = new mongoose.Schema({
    // Core tweet data
    text: { type: String, required: true },
    author: String,
    tweetId: { type: String, unique: true, sparse: true },
    source: { type: String, enum: ["manual", "twitter_api", "database"] },
    
    // Classification (filled by dashboard after model inference)
    classification: {
        predictedLabelId: Number,     // 0-4
        predictedLabel: String,        // Human-readable label name
        confidenceScores: [Number],    // Array of 5 probabilities
        classifiedAt: Date             // When classification happened
    },
    
    // XAI explanation (token importance scores)
    explanation: [{
        token: String,
        score: Number
    }],
    
    // Actionable information (from NLP extraction)
    actionableInfo: {
        locations: [String],
        peopleCount: [{ count: Number, status: String }],
        needs: [String],
        damageType: [String],
        timeMentions: [String]
    },
    
    // HITL status
    status: {
        type: String,
        enum: ["pending", "verified", "unverified", "human_verified"],
        default: "pending"
    },
    humanLabel: Number,    // Human-corrected label (if HITL was used)
    
    // Location fields (for Geospatial Aggregation)
    location: { type: String, index: true },  // Resolved location (indexed)
    locationSource: String,       // "place_tag" | "user_profile" | "text_extraction"
    userProfileLocation: String,  // User's self-declared profile location
    placeTag: String,             // Twitter geo-tagged place name
    placeCountry: String,         // Country from Twitter place
    geoCoordinates: {             // Optional GPS coordinates
        type: { type: String, enum: ["Point"] },
        coordinates: [Number]
    }
}, { timestamps: true });
```

### API Endpoints (`backend/routes/tweets.js`)

| Method | Endpoint | Request Body/Params | Response | Purpose |
|---|---|---|---|---|
| `POST` | `/api/tweets/create` | `{text, author, placeTag, ...}` | Created tweet object | Store a new tweet with location fields |
| `GET` | `/api/tweets` | `?page=1&limit=100&query=...` | `{tweets[], totalPages}` | Fetch tweets (paginated) |
| `PUT` | `/api/tweets/:id/classify` | `{classification, explanation, actionableInfo, status}` | Updated tweet | Save model results for a specific tweet |
| `DELETE` | `/api/tweets/:id` | — | Success/failure | Delete a tweet |
| `GET` | `/api/tweets/geo-aggregate` | `?minClusterSize=5` | Aggregated clusters | Server-side MongoDB aggregation by location |
| `GET` | `/api/health` | — | `{status: "ok"}` | Health check for dashboard connection |

### Location Resolution in Create Route

When a tweet is created via `POST /create`, the backend applies **3-tier location priority**:

```javascript
// Tier 1: placeTag (most reliable — from Twitter geo)
if (req.body.placeTag) {
    location = req.body.placeTag;
    locationSource = "place_tag";
}
// Tier 2: userProfileLocation (user's self-declared)
else if (req.body.userProfileLocation) {
    location = req.body.userProfileLocation;
    locationSource = "user_profile";
}
// Tier 3: Falls back to empty (text extraction happens client-side)
```

---

## 9. Desktop Dashboard — Python/CustomTkinter UI

**File:** `Dashboard/dashboard.py` (~1270 lines)

### Class Hierarchy

```
dashboard.py
├── ConnectionFrame(CTkFrame)      — Backend connection check (polling pattern)
├── TweetInputFrame(CTkFrame)      — Manual tweet text input area
├── GeoAnalysisView(CTkFrame)      — 🌍 Geo Analysis tab (cluster view)
└── MainDashboard(CTk)             — Main application window
    ├── setup_ui()                 — Build the UI layout
    ├── setup_dashboard()          — Build dashboard content area
    ├── on_connection_success()    — Called when backend is connected
    ├── load_tweets_from_db()      — Fetch + classify tweets
    ├── classify_tweets_locally()  — Run model on each tweet
    ├── display_tweets()           — Render tweet list
    ├── show_tweet_detail()        — Show detail panel for selected tweet
    ├── handle_manual_tweet()      — Process manually entered tweet
    ├── filter_tweets()            — Filter by category label
    ├── show_geo_analysis()        — Switch to Geo Analysis view
    └── show_main_dashboard()      — Switch back to main view
```

### Application Lifecycle

```
1. python run_dashboard.py
   └── main() → creates MainDashboard()

2. MainDashboard.__init__()
   ├── Creates main_container frame
   ├── Creates ConnectionFrame (checks backend health)
   └── Creates dashboard_frame (hidden initially)

3. ConnectionFrame.check_backend_status()
   ├── Background thread calls api_client.check_health()
   ├── Main thread polls _check_result every 200ms
   ├── If connected → calls on_connection_success()
   └── If not → shows retry button

4. MainDashboard.on_connection_success()
   ├── Hides ConnectionFrame
   ├── Shows dashboard_frame
   └── Calls load_tweets_from_db()

5. load_tweets_from_db()
   ├── Background thread fetches tweets via API
   ├── Calls classify_tweets_locally(tweets)
   ├── Main thread polls _db_load_done every 300ms
   └── When done → calls display_tweets()

6. User interacts with dashboard
   ├── Clicks tweet → show_tweet_detail()
   ├── Enters text → handle_manual_tweet()
   ├── Clicks filter → filter_tweets()
   └── Clicks 🌍 → show_geo_analysis()
```

### Tweet Detail Panel

When a user clicks a tweet in the list, the detail panel shows:

1. **Full tweet text** with token-level XAI highlighting
2. **Category badge** with color-coded label
3. **Confidence meter** (percentage bar)
4. **Actionable information** (locations, needs, damage, people, time)
5. **HITL correction buttons** (5 category buttons for human verification)

---

## 10. Human-in-the-Loop (HITL) Verification

### Confidence Threshold

**File:** `Dashboard/config.py` → `CONFIDENCE_THRESHOLD = 0.7`

When the model classifies a tweet:
- **confidence ≥ 70%** → status = `"verified"` (auto-accepted)
- **confidence < 70%** → status = `"unverified"` (flagged for human review)

### HITL Queue

The Dashboard's "To Verify" button filters to show only `unverified` tweets. For each:

1. The model's prediction is shown with its confidence score
2. XAI highlighting shows which words influenced the prediction
3. A row of **5 category buttons** lets the human select the correct label
4. The human clicks the correct category
5. The tweet's status changes to `"human_verified"` with `humanLabel` set
6. Results are synced back to MongoDB

### Why HITL Matters

- **Catches errors**: The model has 87% accuracy, meaning ~13% of tweets may be misclassified. HITL catches these.
- **Builds training data**: Human-verified labels can be used to retrain/improve the model.
- **Builds trust**: Operators can verify the AI's work, building confidence in the system.

---

## 11. Geospatial Temporal Aggregation — Complete Reference

> **Full documentation with all 10 cases, severity scoring, and code references:**
> See [GEOSPATIAL_FEATURE.md](GEOSPATIAL_FEATURE.md)

### Summary

The geospatial module (`Dashboard/geospatial_aggregator.py`, 684 lines) groups tweets by geographic location and produces **per-location situation reports**.

### The Two Classes

**`LocationResolver`** — Resolves and normalizes tweet locations
- 3-tier priority: placeTag > userProfileLocation > text extraction
- Normalization: "Downtown Houston" → "houston", "North Manila" → "manila"

**`GeoSpatialAggregator`** — The 6-step analysis pipeline
1. **`cluster_tweets()`** — Group tweets by normalized location
2. **`compute_cluster_consensus()`** — Weighted 5-class distribution per cluster
3. **`_compute_temporal_pattern()`** — Detect bursts (≤6h) vs spreads (>72h)
4. **`determine_cluster_status()`** — Apply 10-case decision matrix
5. **`combine_actionable_info()`** — Merge intelligence from all tweets
6. **`generate_recommended_actions()`** — Map labels to concrete actions

### Quick Reference: 10-Case Decision Matrix

| Case | Condition | Severity | Action |
|---|---|---|---|
| 1. Insufficient Data | 1 tweet only | UNKNOWN | Ignore |
| 2. Suspicious Source | 1 author, many tweets | UNKNOWN | Flag |
| 3. Early Signal | <5 tweets, ≥2 authors | LOW | Monitor |
| 10. Location Uncertain | >60% remote authors | UNCERTAIN | Verify |
| 6. No Disaster | ≥80% Not Humanitarian | NONE | Dismiss |
| 7. Low Confidence | Avg conf <50% | UNCERTAIN | Human Review |
| 5. Ambiguous | Low agreement | MEDIUM | Human Review |
| 8. Active Event ⚡ | Burst + humanitarian | CRITICAL | IMMEDIATE |
| 9. Recovery Phase | Spread + recovery > rescue | MEDIUM | Shift Response |
| 4. Confirmed Event | Enough tweets + authors | Computed | Alert |

---

## 12. Configuration System

**File:** `Dashboard/config.py`

All configurable parameters are centralized in this file:

### Model Configuration
```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"    # Base transformer
MODEL_PATH = "Trained_Model/deltran15_minilm_fp32.pt"     # Weights file
TOKENIZER_PATH = "Trained_Model/Model_Tokenizer"          # Tokenizer directory
```

### Label System
```python
LABEL_MAP = {0: "affected_individuals", 1: "infrastructure...", ...}
LABEL_DISPLAY_NAMES = {0: "Affected Individuals", ...}
LABEL_COLORS = {0: "#FF6B6B", 1: "#FFA500", 2: "#95A5A6", 3: "#3498DB", 4: "#2ECC71"}
ACTIONABLE_LABELS = {0, 1, 4}   # Only these labels show actionable info
```

### HITL Configuration
```python
CONFIDENCE_THRESHOLD = 0.7   # Below this → flagged for human review
```

### Geo Configuration
```python
GEO_MIN_CLUSTER_SIZE = 5        # Min tweets for a "real" cluster
GEO_AGREEMENT_THRESHOLD = 0.35  # Min consensus for non-ambiguous
GEO_MIN_UNIQUE_AUTHORS = 3      # Min different authors
GEO_TEMPORAL_BURST_HOURS = 2    # Burst detection window
GEO_SEVERITY_THRESHOLDS = {     # Severity scoring points
    "tweet_count_high": 40, "tweet_count_medium": 15,
    "people_high": 100, "people_medium": 20
}
```

---

## 13. Threading & Concurrency Model

### The Problem

Python's `tkinter` (which CustomTkinter is built on) is **NOT thread-safe**. All UI widget updates MUST happen on the main thread. But network requests and model inference are slow operations that would freeze the UI if run on the main thread.

### Our Solution: Polling Pattern

Instead of calling `self.after()` from background threads (which crashes on Python 3.13), we use a **shared-variable + main-thread polling** pattern:

```python
# ❌ UNSAFE (crashes on Python 3.13):
def worker():
    result = slow_operation()
    self.after(0, lambda: self.label.configure(text=result))  # BOOM!

# ✅ SAFE (polling pattern):
def worker():
    result = slow_operation()
    self._result = result     # Write to shared variable
    self._done = True         # Signal completion

def _poll():
    if not self._done:
        self.after(300, self._poll)  # Poll again in 300ms
        return
    # Safe to update UI here — we're on the main thread
    self.label.configure(text=self._result)
```

### Where This Pattern Is Used

| Component | Thread Function | Poller | What It Does |
|---|---|---|---|
| `ConnectionFrame` | `check_thread()` | `_poll_result()` | Checks if backend is reachable |
| `GeoAnalysisView` | `worker()` (in load_clusters) | `_poll_geo()` | Classifies + aggregates tweets |
| `MainDashboard` | `load_thread()` | `_poll_db_load()` | Fetches + classifies DB tweets |

---

## 14. File-by-File Reference

| File | Lines | Size | Purpose |
|---|---|---|---|
| **Dashboard/** | | | |
| `dashboard.py` | ~1270 | 53 KB | Main UI: MainDashboard + GeoAnalysisView + ConnectionFrame + TweetInputFrame |
| `model_inference.py` | 291 | 10 KB | Model wrapper: predict, XAI, actionable info, cluster classification |
| `geospatial_aggregator.py` | 684 | 29 KB | Core geo logic: LocationResolver + GeoSpatialAggregator (6 steps, 10 cases) |
| `api_client.py` | ~302 | 12 KB | HTTP REST client: get_tweets, create_tweet, get_all_tweets_for_geo |
| `config.py` | 66 | 2 KB | All configuration parameters |
| `token_highlighter.py` | ~200 | 7 KB | Token-level color mapping for XAI visualization |
| `twitter_api.py` | ~200 | 7 KB | Twitter API v2 wrapper (optional, for fetching live tweets) |
| **Trained_Model/** | | | |
| `deltran15_minilm_fp32.pt` | — | 90 MB | Pre-trained model weights |
| `Model.py` | 22 | 0.7 KB | DeLTran15 class definition (encoder + classifier head) |
| `Explainable_AI.py` | 52 | 1.3 KB | Gradient-based XAI (backward pass + token norm) |
| `Actionable_Info.py` | 114 | 3.5 KB | spaCy NER + regex extraction pipeline |
| `Main.py` | ~60 | 2.4 KB | Standalone CLI classifier |
| **backend/** | | | |
| `models/Tweet.js` | ~70 | — | MongoDB schema definition (with 6 location fields) |
| `routes/tweets.js` | ~220 | — | CRUD endpoints + geo-aggregate |
| `server.js` | ~30 | — | Express server entry point |
| **Root** | | | |
| `run_dashboard.py` | 10 | 0.2 KB | Entry point: `from Dashboard.dashboard import main; main()` |
| `mock_geo_tweets.py` | ~1500 | 59 KB | Mock dataset generator (426 tweets, 14 locations, all 10 cases) |
| `requirements.txt` | 5 | 0.2 KB | Python dependencies |
| `GEOSPATIAL_FEATURE.md` | ~550 | 42 KB | Detailed geo feature documentation |
| `README.md` | ~500 | — | Project overview and quick start |
| `PROJECT_DOCUMENTATION.md` | this file | — | This comprehensive technical documentation |
