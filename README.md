# 🌊 Disaster Tweet Multi-Class Classification System

> **A comprehensive AI-powered disaster tweet classification system with explainable AI (XAI), actionable intelligence extraction, geospatial temporal analysis, and a Human-in-the-Loop (HITL) verification workflow.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-16+-green.svg)](https://nodejs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen.svg)](https://mongodb.com)

---

## 📑 Table of Contents

1. [Overview](#-overview)
2. [System Architecture](#-system-architecture)
3. [The 5-Class Model — DeLTran15](#-the-5-class-model--deltran15)
4. [Explainable AI (XAI)](#-explainable-ai-xai)
5. [Actionable Information Extraction](#-actionable-information-extraction)
6. [Human-in-the-Loop (HITL)](#-human-in-the-loop-hitl)
7. [Geospatial Temporal Aggregation](#-geospatial-temporal-aggregation)
8. [Desktop Dashboard](#-desktop-dashboard)
9. [Backend API](#-backend-api)
10. [Project Structure](#-project-structure)
11. [Quick Start](#-quick-start)
12. [Results & Evaluation](#-results--evaluation)
13. [Privacy & Security](#-privacy--security)
14. [Dependencies](#-dependencies)
15. [Troubleshooting](#-troubleshooting)
16. [Documentation](#-documentation)

---

## 🌐 Overview

### The Problem

During natural disasters (earthquakes, floods, hurricanes, wildfires), social media platforms like Twitter become a critical source of real-time information. Emergency responders, governments, and NGOs need to rapidly understand:

- **Who** is affected and where
- **What** infrastructure is damaged
- **What** resources are needed
- **What** help is already being offered

However, the volume of tweets during a disaster (thousands per hour) makes manual reading impossible. Automated classification is needed, but a simple "disaster / not disaster" binary classifier is insufficient — responders need to know the **type** of information in each tweet.

### Our Solution

This system classifies disaster tweets into **5 specific humanitarian categories**, provides **explainable AI** so users understand why the model made each decision, extracts **actionable intelligence** (locations, people counts, resource needs), and aggregates this data **geospatially** to give a location-level situation overview.

### Key Features

| Feature | Description |
|---|---|
| **5-Class Classification** | Affected Individuals, Infrastructure Damage, Not Humanitarian, Other Information, Rescue/Donation |
| **Explainable AI (XAI)** | Gradient-based token-level highlighting shows which words influenced the prediction |
| **Actionable Info Extraction** | NLP pipeline extracts locations, needs, damage types, people counts, time mentions |
| **Geospatial Aggregation** | Groups tweets by location, computes 5-class consensus, applies 10-case decision matrix |
| **HITL Verification** | Low-confidence predictions are flagged for human review and correction |
| **Privacy-First** | Model runs 100% locally — no cloud APIs, no data leaves your machine |
| **Desktop Dashboard** | Full-featured Python/CustomTkinter GUI for classification and analysis |

---

## 🏗️ System Architecture

The project follows a **decoupled, privacy-first architecture** where data storage is centralized (MongoDB) but all AI inference runs securely on the user's local machine.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER'S LOCAL MACHINE                              │
│                                                                                │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │  Desktop Dashboard (Python / CustomTkinter)                              │  │
│  │                                                                          │  │
│  │  ┌────────────┐  ┌──────────────┐  ┌───────────────┐  ┌──────────────┐  │  │
│  │  │ dashboard.py│  │model_inference│  │  geospatial_  │  │ api_client.py│  │  │
│  │  │  Main UI    │→ │  .py         │  │  aggregator.py│  │  HTTP Client │  │  │
│  │  │  Tabs/Views │  │  DeLTran15   │  │  6-Step       │  │  REST calls  │  │  │
│  │  │  HITL       │  │  XAI         │  │  Pipeline     │  │  to backend  │  │  │
│  │  └─────┬───────┘  │  Actionable  │  │  10 Cases     │  └───────┬──────┘  │  │
│  │        │          │  Info        │  └───────────────┘          │         │  │
│  │        │          └──────────────┘                             │         │  │
│  └────────│──────────────────────────────────────────────────────│─────────┘  │
│           │                                                      │            │
│  ┌────────│──────────────────────────────────────────────────────│─────────┐  │
│  │  Trained Model Files                                          │         │  │
│  │  ├── deltran15_minilm_fp32.pt (90 MB, model weights)         │         │  │
│  │  ├── Model.py (DeLTran15 architecture)                        │         │  │
│  │  ├── Explainable_AI.py (gradient-based XAI)                   │         │  │
│  │  ├── Actionable_Info.py (spaCy NER + regex)                   │         │  │
│  │  └── Model_Tokenizer/ (tokenizer files)                       │         │  │
│  └───────────────────────────────────────────────────────────────│─────────┘  │
│                                                                  │            │
└──────────────────────────────────────────────────────────────────│────────────┘
                                                                   │
                                                          HTTP REST API
                                                                   │
┌──────────────────────────────────────────────────────────────────│────────────┐
│  Backend Server (Node.js / Express)                              │            │
│  ├── POST /api/tweets/create       ← Store tweets + location ←──┘            │
│  ├── GET  /api/tweets              → Fetch tweets (paginated)                │
│  ├── PUT  /api/tweets/:id/classify ← Store classification results            │
│  ├── GET  /api/tweets/geo-aggregate→ Server-side geo aggregation             │
│  └── MongoDB (Atlas or local)      → Persistent tweet storage                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: User enters a tweet manually in the Dashboard, or tweets are fetched from the MongoDB database
2. **Local Inference**: The Dashboard passes the tweet text to the local PyTorch model (`model_inference.py`)
3. **Classification + XAI**: The model predicts a category (out of 5), calculates token-level importance (for XAI highlighting), and extracts actionable information using NLP
4. **Storage**: Classification results are synced back to the MongoDB database
5. **Geo Analysis**: When Geo Analysis is activated, all tweets are grouped by location and analyzed as clusters

---

## 🧠 The 5-Class Model — DeLTran15

### Classification Categories

| ID | Category | Icon | What It Means | Example Tweet |
|---|---|---|---|---|
| 0 | **Affected Individuals** | 🔴 | People hurt, trapped, displaced, or missing | *"Family trapped on roof in Houston, need rescue NOW"* |
| 1 | **Infrastructure Damage** | 🟠 | Roads, bridges, power, water infrastructure damaged | *"Highway 59 bridge collapsed, avoid the area"* |
| 2 | **Not Humanitarian** | ⚪ | Irrelevant / non-disaster content | *"Beautiful sunset in Paris today"* |
| 3 | **Other Information** | 🔵 | Warnings, official updates, general awareness | *"Earthquake alert issued for Manila region"* |
| 4 | **Rescue/Donation** | 🟢 | Help offers, volunteer coordination, donations | *"Red Cross shelter open at NRG Center"* |

### Model Architecture

```
Input Text: "30 people trapped after bridge collapse in Manila"
                    ↓
            Tokenizer (all-MiniLM-L6-v2)
                    ↓
         Token IDs: [101, 1017, 2111, 7416, ...]
                    ↓
    ┌───────────────────────────────┐
    │ Transformer Encoder           │
    │ (all-MiniLM-L6-v2, 22M params)│
    │ 6 layers, 384 hidden dim     │
    └───────────────┬───────────────┘
                    ↓
            [CLS] Embedding (384-dim vector)
                    ↓
            Dropout (p=0.3)
                    ↓
    ┌───────────────────────────────┐
    │ Linear Classifier             │
    │ (384 → 5 classes)             │
    └───────────────┬───────────────┘
                    ↓
            Softmax → [0.02, 0.85, 0.01, 0.08, 0.04]
                    ↓
            Prediction: Label 1 (Infrastructure Damage), 85% confidence
```

**Model File:** `Trained_Model/deltran15_minilm_fp32.pt` (90 MB)
**Architecture:** `Trained_Model/Model.py` — `DeLTran15` class (nn.Module)
**Base Model:** `sentence-transformers/all-MiniLM-L6-v2` — chosen for strong semantic understanding of short texts while remaining lightweight

### Training Process

1. **Data Collection**: Raw disaster tweets collected in `Data_Set/Unprocessed_Data_Sets/`
2. **Preprocessing**: Custom scripts (`Data_Extraction.py`, `Data_Cleaning.py`) clean text — removing URLs, stripping special characters, expanding contractions, caching emojis
3. **Labeling**: Data categorized into the 5 classes
4. **Fine-Tuning**: `Model_Build/Build.py` fine-tunes the base model locally, outputting `deltran15_minilm_fp32.pt`

### Performance

| Metric | Score |
|---|---|
| **Global Accuracy** | 87.0% |
| **Macro F1-Score** | 87.26% |
| **Weighted F1-Score** | 87.26% |

---

## 🔍 Explainable AI (XAI)

**File:** `Trained_Model/Explainable_AI.py`

### How It Works

The XAI system uses **gradient-based token importance analysis** to show which words most influenced the model's prediction:

1. **Forward pass**: Run the tweet through the model to get predictions
2. **Backward pass**: Compute gradients of the predicted class logit with respect to input embeddings
3. **Token scoring**: Calculate the L2 norm of each token's embedding vector — higher norm = more important
4. **Visual highlighting**: Map importance scores to a white→red color gradient in the UI

### Example

```
Input:  "30 people trapped after bridge collapse in Manila"
         ── ────── ─────── ───── ────── ──────── ── ──────
Score:  0.1  0.6    0.9    0.3   0.8    0.7    0.2  0.5
                    ^^^           ^^^    ^^^          ^^^
                 (trapped)     (bridge)(collapse)  (Manila)
                 
Visual: [white "30"] [pink "people"] [RED "trapped"] [light "after"]
        [RED "bridge"] [RED "collapse"] [white "in"] [orange "Manila"]
```

This builds **trust and transparency** — users can see the model is focusing on the right words, not making random predictions.

---

## 📋 Actionable Information Extraction

**File:** `Trained_Model/Actionable_Info.py`

For tweets classified as disaster-related (labels 0, 1, 4), the system extracts structured actionable intelligence using **spaCy NLP** (Named Entity Recognition) and **regex patterns**:

| Extracted Entity | Method | Example |
|---|---|---|
| 📍 **Locations** | spaCy NER (GPE, LOC, FAC entities) + regex fallback | "Manila", "Highway 59", "Houston, TX" |
| 👥 **People Counts** | Regex patterns for casualty phrases | "30 injured" → `{count: 30, status: "injured"}` |
| 🆘 **Needs** | Keyword matching (food, water, shelter, medicine, etc.) | ["food", "water", "medicine"] |
| 💥 **Damage Types** | Keyword matching (bridge, road, collapsed, flooded, etc.) | ["bridge", "collapsed", "flooded"] |
| ⏰ **Time Mentions** | Regex patterns (now, today, 2 hours ago, etc.) | ["now", "immediately"] |

### Pipeline

```python
Input: "30 people trapped after bridge collapse near Manila Bay, need water and food NOW"

extract_actionable_info(text) →
{
    "locations": ["Manila Bay"],
    "people_count": [{"count": 30, "status": "trapped"}],
    "needs": ["food", "water"],
    "damage_type": ["bridge", "collapsed"],
    "time_mentions": ["NOW"]
}
```

---

## 🔄 Human-in-the-Loop (HITL)

### Why HITL?

No model is 100% accurate. When the model's confidence is below the threshold (default: 0.7), the tweet is flagged as **"unverified"** and shown in the HITL queue for human review.

### Workflow

```
Tweet classified → confidence score
                    ↓
              ≥ 70%?  ──YES──→  "verified" (auto-accepted)
                ↓ NO
           "unverified" → shown in HITL queue
                ↓
          Human reviewer sees:
          ├── Original tweet text
          ├── Model's prediction + confidence
          ├── XAI highlighting (why the model chose this)
          └── 5 category buttons to correct if needed
                ↓
          Human clicks correct category
                ↓
          Status → "human_verified"
          Saved to database with human label
```

### In the Dashboard

- Click the **"To Verify"** button (orange) to filter to unverified tweets
- Click the **"Verified"** button (purple) to see all human-verified tweets  
- Each unverified tweet shows the model's prediction with a label row of 5 buttons for correction

---

## 🌍 Geospatial Temporal Aggregation

> **Full documentation:** See [GEOSPATIAL_FEATURE.md](GEOSPATIAL_FEATURE.md) for the complete technical reference.

### What It Does

Instead of classifying tweets one at a time, this feature **groups tweets by location** and provides a **location-level situation report**:

```
Manila, Philippines (25 tweets from 11 authors):
  🔴 Affected Individuals:    8%    ← People need rescue
  🟠 Infrastructure Damage:  28%    ← Roads, bridges damaged
  ⚪ Not Humanitarian:        0%    ← No noise — all real
  🔵 Other Information:      36%    ← Updates and warnings
  🟢 Rescue/Donation:        28%    ← Relief is underway
  
  Status: ACTIVE EVENT (CRITICAL)
  Reason: "25 tweets within 2.0 hours. Ongoing event detected!"
  → Deploy rescue teams + engineers + coordinate donations
```

### The 6-Step Pipeline

| Step | What It Does | Code |
|---|---|---|
| 1. Cluster | Group tweets by normalized location (3-tier resolution) | `cluster_tweets()` |
| 2. Consensus | Compute weighted 5-class distribution per cluster | `compute_cluster_consensus()` |
| 3. Temporal | Detect temporal bursts (≤6h) vs spreads (>72h) | `_compute_temporal_pattern()` |
| 4. Decision | Apply 10-case decision matrix → status + severity | `determine_cluster_status()` |
| 5. Intelligence | Merge actionable info from all tweets in cluster | `combine_actionable_info()` |
| 6. Actions | Generate specific response recommendations | `generate_recommended_actions()` |

### The 10-Case Decision Matrix

| # | Case | Trigger | Severity | Action |
|---|---|---|---|---|
| 1 | Insufficient Data | 1 tweet only | UNKNOWN | Ignore |
| 2 | Suspicious Source | 1 author, many tweets | UNKNOWN | Flag |
| 3 | Early Signal | <5 tweets, ≥2 authors | LOW | Monitor |
| 10 | Location Uncertain | >60% remote authors | UNCERTAIN | Verify |
| 6 | No Disaster | ≥80% "Not Humanitarian" | NONE | Dismiss |
| 7 | Low Confidence | Avg confidence <50% | UNCERTAIN | Human Review |
| 5 | Ambiguous | Low agreement, >30% humanitarian | MEDIUM | Human Review |
| 8 | Active Event | Temporal burst + >70% humanitarian | **CRITICAL** | **IMMEDIATE** |
| 9 | Recovery Phase | Spread >72h + recovery > rescue | MEDIUM | Shift Response |
| 4 | Confirmed Event | ≥ min tweets + ≥ min authors | Computed | Alert |

---

## 💻 Desktop Dashboard

The primary user interface, built with **Python/CustomTkinter**.

### Main Dashboard View

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Disaster Tweet Classifier    [Refresh Database] [🌍 Geo Analysis]       │
├──────────────────────────┬───────────────────────────────────────────────┤
│ [Manual Tweet Input]     │                                               │
│ ┌──────────────────────┐ │  Tweet Detail Panel                          │
│ │ Enter tweet text...  │ │  ┌─────────────────────────────────────────┐ │
│ │        [Classify]    │ │  │ "30 people trapped after bridge collapse│ │
│ └──────────────────────┘ │  │  in Manila, need rescue NOW"            │ │
│                          │  │                                          │ │
│ [All][🔴][🟠][⚪][🔵][🟢]│  │  Category: 🔴 Affected Individuals     │ │
│ [To Verify] [Verified]   │  │  Confidence: 92%                        │ │
│                          │  │                                          │ │
│ Tweet List:              │  │  Token Highlighting (XAI):              │ │
│ ┌──────────────────────┐ │  │  "30 people [TRAPPED] after [BRIDGE]    │ │
│ │ 🔴 Family trapped... │ │  │   [COLLAPSE] in [Manila], need rescue"  │ │
│ │ 🟠 Highway 59 bridge │ │  │                                          │ │
│ │ 🟢 Red Cross shelter │ │  │  Actionable Information:                │ │
│ │ ⚪ Watching the game  │ │  │  📍 Locations: Manila                   │ │
│ │ 🔵 Alert issued for  │ │  │  👥 People: 30 trapped                  │ │
│ └──────────────────────┘ │  │  🆘 Needs: rescue                       │ │
│                          │  │  💥 Damage: bridge, collapse             │ │
│                          │  │  ⏰ Time: NOW                            │ │
│                          │  └─────────────────────────────────────────┘ │
└──────────────────────────┴───────────────────────────────────────────────┘
```

### Geo Analysis View

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 🌍 Geo Analysis        [24 clusters loaded]  [🔄 Refresh] [← Back]    │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Location Clusters│  📍 Manila, Philippines                              │
│                  │  Severity: CRITICAL  Urgency: IMMEDIATE  Case: ...  │
│ 🚨 Manila   CRIT│  "25 tweets within 2.0 hours. Ongoing event!"       │
│ ✅ Houston   MED│                                                       │
│ ✅ Miami     MED│  5-Class Distribution:                                │
│ ✅ S.F.      MED│  🔴 Affected     ████           8%                   │
│ ⚠️ Istanbul  MED│  🟠 Infrastructure██████████    28%                  │
│ ❓ Houston   UNC│  ⚪ Not Humanitarian             0%                   │
│ 📊 Paris    NONE│  🔵 Other Info    ████████████  36%                  │
│      ...        │  🟢 Rescue        ██████████    28%                  │
│                 │                                                       │
│                 │  Recommended Actions:                                  │
│                 │  🚨 IMMEDIATE: Real-time disaster in progress         │
│                 │  🔴 RESCUE: Deploy teams to Manila, Tondo            │
│                 │  🟠 ENGINEERS: Dispatch for bridge, flooded          │
└──────────────────┴───────────────────────────────────────────────────────┘
```

---

## 🗄️ Backend API

**Stack:** Node.js + Express + MongoDB (Mongoose)

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/tweets/create` | Create a new tweet (with location fields) |
| `GET` | `/api/tweets` | Fetch tweets (paginated, with query/filter) |
| `PUT` | `/api/tweets/:id/classify` | Save classification results for a tweet |
| `DELETE` | `/api/tweets/:id` | Delete a tweet |
| `GET` | `/api/tweets/geo-aggregate` | Server-side geospatial aggregation |
| `GET` | `/api/health` | Health check endpoint |

### Tweet Schema (MongoDB)

```javascript
{
  text: String,                    // Tweet text
  author: String,                  // Twitter handle
  tweetId: String,                 // Unique tweet ID
  source: String,                  // "manual" | "twitter_api" | "database"
  status: String,                  // "pending" | "verified" | "unverified" | "human_verified"
  
  // Classification (filled by model)
  classification: {
    predictedLabelId: Number,      // 0-4
    predictedLabel: String,        // e.g., "Affected Individuals"
    confidenceScores: [Number],    // [0.85, 0.05, 0.03, 0.04, 0.03]
    classifiedAt: Date
  },
  
  // XAI explanation
  explanation: [{ token: String, score: Number }],
  
  // Actionable info (spaCy NER)
  actionableInfo: {
    locations: [String],
    peopleCount: [{ count: Number, status: String }],
    needs: [String],
    damageType: [String],
    timeMentions: [String]
  },

  // Location fields (for Geo Analysis)
  location: String,                // Resolved location (indexed)
  locationSource: String,          // "place_tag" | "user_profile" | "text_extraction"
  userProfileLocation: String,
  placeTag: String,
  placeCountry: String,
  geoCoordinates: { type: String, coordinates: [Number] },
  
  createdAt: Date,
  updatedAt: Date
}
```

---

## 📁 Project Structure

```
Disaster_Tweets_MultiClass_Classifier/
│
├── Dashboard/                          # Python Desktop Dashboard ⭐
│   ├── dashboard.py                   # Main UI (1270 lines) — MainDashboard + GeoAnalysisView
│   ├── model_inference.py             # Model wrapper — predict, XAI, actionable info
│   ├── geospatial_aggregator.py       # 🌍 GeoSpatial aggregation (684 lines)
│   ├── api_client.py                  # HTTP client for backend REST API
│   ├── config.py                      # Configuration — model paths, labels, geo settings
│   ├── token_highlighter.py           # Token-level color highlighting for XAI
│   ├── twitter_api.py                 # Twitter API v2 wrapper (optional)
│   ├── HITL/                          # Human-in-the-Loop components
│   └── __init__.py
│
├── Trained_Model/                      # Pre-trained AI Model
│   ├── deltran15_minilm_fp32.pt       # 🧠 Model weights (90 MB)
│   ├── Model.py                       # DeLTran15 architecture (MiniLM + classifier head)
│   ├── Explainable_AI.py              # Gradient-based XAI implementation
│   ├── Actionable_Info.py             # spaCy NER + regex info extraction
│   ├── Main.py                        # CLI classifier (standalone)
│   └── Model_Tokenizer/              # HuggingFace tokenizer files
│
├── backend/                            # Node.js Backend Server
│   ├── models/
│   │   └── Tweet.js                   # MongoDB schema (with location fields)
│   ├── routes/
│   │   ├── tweets.js                  # Tweet CRUD + geo-aggregate routes
│   │   └── auth.js                    # Authentication routes
│   ├── services/
│   │   └── twitterService.js          # Twitter API wrapper (for web frontend)
│   ├── server.js                      # Express server entry point
│   └── package.json
│
├── Data_Set/                           # Training Data Pipeline
│   ├── Unprocessed_Data_Sets/         # Raw disaster tweet datasets
│   ├── Processed_Data_Set/            # Cleaned CSVs ready for training
│   └── Data_Preprocessing/           # Data cleaning scripts
│       ├── Data_Extraction.py
│       └── Data_Cleaning.py
│
├── Model_Build/                        # Model Training
│   └── Build.py                       # Fine-tuning script
│
├── reports/                            # Evaluation Reports
│   └── test_tweets_report/            # Model performance visualizations
│       ├── fig_confusion_matrix_labeled.png
│       ├── fig_expected_vs_predicted_labeled.png
│       ├── fig_confidence_distribution.png
│       └── fig_actionable_field_coverage.png
│
├── run_dashboard.py                    # 🚀 Entry point for desktop dashboard
├── mock_geo_tweets.py                  # Mock dataset generator (426 tweets, 14 locations)
├── requirements.txt                    # Python dependencies
├── GEOSPATIAL_FEATURE.md              # 🌍 Detailed geo analysis documentation
└── README.md                          # This file
```

---

## 🚀 How to Run This Project

### Prerequisites

Before you begin, make sure you have the following installed:

| Software | Version | Check Command | Download |
|---|---|---|---|
| **Python** | 3.8+ (tested on 3.13) | `python --version` | [python.org](https://python.org) |
| **Node.js** | v16+ | `node --version` | [nodejs.org](https://nodejs.org) |
| **npm** | v8+ | `npm --version` | Comes with Node.js |
| **MongoDB** | Any (local or Atlas) | `mongosh --version` | [mongodb.com](https://mongodb.com) |
| **Git** | Any | `git --version` | [git-scm.com](https://git-scm.com) |

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/Disaster_Tweets_MultiClass_Classifier.git
cd Disaster_Tweets_MultiClass_Classifier
```

### Step 2: Set Up MongoDB

**Option A — MongoDB Atlas (Cloud, Recommended):**
1. Go to [MongoDB Atlas](https://www.mongodb.com/atlas) and create a free account
2. Create a new cluster (free tier is sufficient)
3. Create a database user with a password
4. Whitelist your IP address (or allow access from anywhere: `0.0.0.0/0`)
5. Click **"Connect"** → **"Connect your application"** → Copy the connection string
6. It will look like: `mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/disaster_tweets`

**Option B — Local MongoDB:**
1. Install MongoDB Community Edition from [mongodb.com](https://www.mongodb.com/try/download/community)
2. Start the MongoDB service: `mongod`
3. Your connection string will be: `mongodb://localhost:27017/disaster_tweets`

### Step 3: Set Up the Backend Server

```bash
# Navigate to backend folder
cd backend

# Install Node.js dependencies
npm install
```

Create a `.env` file in the `backend/` folder with your MongoDB connection string:  -- do this only if shows any error 

```bash 
# backend/.env
MONGODB_URI=mongodb+srv://your_username:your_password@cluster0.xxxxx.mongodb.net/disaster_tweets
PORT=5000  
```

Start the backend server:

```bash
npm run dev
```

You should see:
```
Server running on port 5000
Connected to MongoDB
```

> **Keep this terminal open.** The backend must be running for the dashboard to work.

### Step 4: Set Up the Python Dashboard

Open a **new terminal** (keep the backend running in the first one):

```bash
# Navigate back to project root (if you're in backend/)
cd ..

# Install Python dependencies
pip install -r requirements.txt
```

This installs: `torch`, `transformers`, `customtkinter`, `spacy`, `requests`

Download the spaCy English language model (required for actionable info extraction):

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Verify the Model File Exists

The trained model weights file must be present:

```bash
# Check if the model file exists (should be ~90 MB)
# Windows:
dir Trained_Model\deltran15_minilm_fp32.pt

# macOS/Linux:
ls -la Trained_Model/deltran15_minilm_fp32.pt
```

If the file is missing, you need to either:
- Download it from your team's shared storage
- Or retrain the model: `cd Model_Build && python Build.py`

### Step 6: Run the Dashboard

```bash
python run_dashboard.py
```

The dashboard window will open and automatically try to connect to the backend at `http://localhost:5000`.

You should see:
```
Model loaded successfully!
Backend server connected
```

### Step 7: Load Test Data (Optional)

If your MongoDB is empty and you want to test with sample data:

```bash
# Load 426 mock disaster tweets across 14 locations (for testing all 10 geo cases)
python mock_geo_tweets.py
```

Then click **"Refresh Database"** in the dashboard to load the tweets.

### Step 8: Using the Dashboard

#### Basic Tweet Classification
1. **Manual Input**: Type or paste a tweet in the input box → Click **"Classify"**
2. **From Database**: Click **"Refresh Database"** to load tweets from MongoDB
3. Tweets are **automatically classified** by the local model
4. Click any tweet to see:
   - XAI token highlighting (which words mattered)
   - Actionable intelligence (locations, needs, damage)
   - Confidence score

#### Filtering & HITL
5. **Filter by category**: Click the colored buttons (🔴 🟠 ⚪ 🔵 🟢) in the toolbar
6. **HITL Review**: Click **"To Verify"** to see low-confidence tweets that need human review
7. **Correct predictions**: Click the correct category button to override the model's prediction

#### Geospatial Analysis
8. Click **"🌍 Geo Analysis"** in the top bar
9. Wait for auto-classification (~2 minutes for 426 tweets on first load)
10. Browse location clusters on the left panel
11. Click any cluster to see:
    - 5-class distribution bars
    - Severity / Urgency / Case badges
    - Combined actionable intelligence
    - Recommended response actions

### Running Everything Together (Quick Command Summary)

```bash
# Terminal 1 — Backend
cd backend
npm run dev

# Terminal 2 — Dashboard (new terminal)
cd ..
python run_dashboard.py

# Terminal 3 — Load test data (optional, new terminal)
python mock_geo_tweets.py
```

---

## 📊 Results & Evaluation

Our DeLTran15 model (`all-MiniLM-L6-v2` fine-tuned) was evaluated on a held-out test set of 300 high-confidence labeled tweets:

| Metric | Score |
|---|---|
| **Global Accuracy** | 87.0% |
| **Macro F1-Score** | 87.26% |
| **Weighted F1-Score** | 87.26% |

### Confusion Matrix
The confusion matrix shows strong diagonal alignment — the model rarely confuses distinct disaster categories.

![Confusion Matrix](reports/test_tweets_report/fig_confusion_matrix_labeled.png)

### Expected vs. Predicted Distributions
The model's predicted distribution closely matches the ground-truth labels across all 5 classes.

![Expected vs Predicted](reports/test_tweets_report/fig_expected_vs_predicted_labeled.png)

### Confidence Distribution
The model shows high confidence in most predictions, with scores skewing close to 1.0.

![Confidence Distribution](reports/test_tweets_report/fig_confidence_distribution.png)

### Actionable Field Coverage
For actionable tweets (labels 0, 1, 4), the NLP pipeline successfully extracts critical metadata.

![Actionable Field Coverage](reports/test_tweets_report/fig_actionable_field_coverage.png)

---

## 🔒 Privacy & Security

- ✅ **Model runs locally** — All inference happens on your machine, never in the cloud
- ✅ **No external API calls** — No OpenAI, no Google, no cloud classification services
- ✅ **Data stays private** — Your tweets and classifications stay on your machine + your MongoDB
- ✅ **Offline capable** — Once tweets are loaded, classification works without internet
- ✅ **No Twitter API in dashboard** — Dashboard works with database only

---

## 📦 Dependencies

### Python (Desktop Dashboard)

| Package | Version | Purpose |
|---|---|---|
| `torch` | ≥ 2.0 | Deep learning framework |
| `transformers` | ≥ 4.30 | HuggingFace transformers (tokenizer + base model) |
| `customtkinter` | ≥ 5.0 | Modern desktop UI framework |
| `spacy` | ≥ 3.0 | NLP for actionable info extraction (NER) |
| `requests` | ≥ 2.28 | HTTP client for backend API |

### Node.js (Backend)

| Package | Purpose |
|---|---|
| `express` | Web framework |
| `mongoose` | MongoDB ODM |
| `cors` | CORS middleware |
| `dotenv` | Environment variables |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| **Model not loading** | Ensure `Trained_Model/deltran15_minilm_fp32.pt` exists (90 MB file) |
| **Backend errors** | Check MongoDB is running and `.env` has correct `MONGODB_URI` |
| **Dashboard connection errors** | Ensure backend is running on port 5000 before starting dashboard |
| **Import errors** | Run `pip install -r requirements.txt` |
| **spaCy model missing** | Run `python -m spacy download en_core_web_sm` |
| **Threading errors (Python 3.13)** | Dashboard uses polling pattern; restart if you see RuntimeError |
| **Geo Analysis shows "unclassified"** | Wait for auto-classification on first load (~2 min for 426 tweets) |

---

## 📚 Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | This file — project overview |
| [GEOSPATIAL_FEATURE.md](GEOSPATIAL_FEATURE.md) | Complete geospatial analysis documentation (10 cases, pipeline, scoring) |
| [Dashboard/README.md](Dashboard/README.md) | Dashboard-specific documentation |

---

## 🙏 Acknowledgments

- **DeLTran15** — Custom fine-tuned model based on `sentence-transformers/all-MiniLM-L6-v2`
- **HuggingFace** — Transformer model infrastructure
- **spaCy** — Named Entity Recognition for actionable info extraction
- **CustomTkinter** — Modern Python desktop UI framework
- **MongoDB Atlas** — Cloud database hosting
