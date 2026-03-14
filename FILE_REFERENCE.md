# File-by-File Reference Guide

> **Purpose:** This document explains **what every single file** in this project does, organized by folder. Written for teachers, reviewers, and evaluators who want to understand the role of each component.

---

## 📁 Root Directory

These files sit at the top level of the project.

| File | What It Does |
|---|---|
| `run_dashboard.py` | **Entry point** of the application. This is the file you run (`python run_dashboard.py`) to launch the desktop dashboard. It simply imports and calls `main()` from `Dashboard/dashboard.py`. |
| `requirements.txt` | Lists all Python packages needed to run the project: `torch`, `transformers`, `customtkinter`, `spacy`, `requests`. Install them with `pip install -r requirements.txt`. |
| `mock_geo_tweets.py` | **Test data generator.** Contains 426 hardcoded tweet dictionaries across 14 different locations (Houston, Manila, Paris, etc.). When run, it sends each tweet to the backend API via HTTP POST requests, populating the MongoDB database with realistic test data that covers all 10 geospatial decision cases. |
| `mock_geo_tweets.json` | **JSON export** of the same 426 tweets. A static snapshot of the mock data so you can inspect the raw tweet data without running the generator script. |
| `load_test_tweets.py` | A utility script to load a smaller set of test tweets from `test_tweets.txt` into the database. Used for quick testing of the classification pipeline without loading the full 426-tweet geo dataset. |
| `clear_tweets.py` | A utility script that **deletes all tweets** from the MongoDB database. Useful for resetting the database to a clean state before loading fresh test data. |
| `test_aggregator.py` | A standalone test script for the `GeoSpatialAggregator` class. It creates sample tweet data and runs it through the aggregator to verify that the 6-step pipeline and 10-case decision matrix work correctly without needing the full dashboard. |
| `test_tweets.txt` | A text file containing sample tweet texts, one per line. Used by `load_test_tweets.py` to bulk-load tweets for testing. |
| `test_tweets_mixed.txt` | A larger collection of sample tweets with a mix of all 5 categories. Used for more comprehensive testing of the model's classification accuracy across different tweet types. |
| `test_tweets_simple.txt` | A small collection of simple, clear-cut tweet examples. Used for quick sanity checks — each tweet is obviously one category, making it easy to verify the model is working. |
| `README.md` | The main project documentation. Explains the architecture, features, how to run the project, model performance, and all components. |
| `GEOSPATIAL_FEATURE.md` | Detailed technical documentation specifically for the Geospatial Temporal Aggregation feature. Covers the 10-case decision matrix, the 6-step pipeline, severity scoring, and the complete report structure. |
| `PROJECT_DOCUMENTATION.md` | In-depth technical documentation for the entire project. Covers the model architecture, XAI internals, NLP pipeline, threading model, backend schema, and every component in code-level detail. |
| `.gitignore` | Tells Git which files to ignore (e.g., `__pycache__/`, `node_modules/`, `.env`, model weights). Prevents large or sensitive files from being committed to the repository. |

---

## 📁 Dashboard/ — Python Desktop Application

This folder contains the **desktop GUI application** — the primary user interface.

### `dashboard.py` (~1270 lines, 53 KB) — The Main Application

This is the **largest and most important file** in the project. It contains the entire desktop UI built with CustomTkinter. It has 4 major classes:

| Class | Lines | What It Does |
|---|---|---|
| `ConnectionFrame` | ~70 | The first screen users see. It checks if the backend server is running by hitting `GET /api/health`. Shows a "Connecting..." spinner, and if the backend is unreachable, shows a "Retry" button. Uses a thread-safe polling pattern for Python 3.13 compatibility. |
| `TweetInputFrame` | ~50 | The text input area where users can type or paste a tweet. Has a text box and a "Classify" button. When clicked, sends the tweet through the full classification pipeline (model → XAI → actionable info). |
| `GeoAnalysisView` | ~200 | The **Geo Analysis tab**. Has a split layout: left panel shows a scrollable list of location cluster cards (e.g., "Manila — 25 tweets — CRITICAL"), right panel shows the full detail report for the selected cluster (5-class distribution bars, severity badges, recommended actions). Handles auto-classification of all tweets in a background thread. |
| `MainDashboard` | ~900 | The main application window. Contains the top toolbar (filter buttons, Geo Analysis button, Refresh button), the tweet list panel, and the tweet detail panel. Manages loading tweets from the database, running local classification, displaying results with XAI highlighting, and HITL verification buttons. |

**Key methods in `MainDashboard`:**
- `setup_ui()` — Builds the entire UI layout (toolbar, panels, buttons)
- `load_tweets_from_db()` — Fetches tweets from MongoDB via the API client, classifies them locally
- `classify_tweets_locally(tweets)` — Runs each tweet through `ModelInference.classify_tweet()`
- `display_tweets()` — Renders the tweet list with category colors and confidence badges
- `show_tweet_detail(tweet)` — Shows XAI highlighting, actionable info, and HITL buttons for a selected tweet
- `filter_tweets(label_id)` — Filters the tweet list by category (e.g., show only 🔴 Affected Individuals)
- `show_geo_analysis()` — Switches from main view to the Geo Analysis view

---

### `model_inference.py` (291 lines, 10 KB) — AI Model Wrapper

This file is the **bridge between the UI and the AI model**. It wraps all the ML logic into a clean Python class.

| Method | What It Does |
|---|---|
| `__init__()` | Initializes the model wrapper, sets up file paths |
| `_load_model()` | Loads the DeLTran15 model weights from `deltran15_minilm_fp32.pt` and the tokenizer from `Model_Tokenizer/`. This takes ~5 seconds on first load. |
| `predict(text)` | Takes tweet text → tokenizes it → runs through the model → returns `(label_id, confidence_scores, label_name)`. Example: `predict("bridge collapsed") → (1, [0.04, 0.85, ...], "infrastructure_damage")` |
| `get_explanation(text)` | Calls `Explainable_AI.py` → returns a list of `(token, importance_score)` tuples for XAI highlighting |
| `get_actionable_info(text, label_id)` | Calls `Actionable_Info.py` → returns extracted locations, needs, damage types, people counts, time mentions. Only runs for actionable labels (0, 1, 4). |
| `classify_tweet(text)` | **The main method.** Runs the complete pipeline: predict → XAI → actionable info. Returns a single dict with all results. |
| `classify_tweet_cluster(tweets, location)` | Classifies all tweets in a geo cluster and returns a consensus report. Used by the Geo Analysis feature. |

---

### `geospatial_aggregator.py` (684 lines, 29 KB) — Geospatial Analysis Engine

The **core intelligence module** for location-based analysis. Contains two classes:

**`LocationResolver`** — Resolves where a tweet is from
- `resolve(tweet)` — Checks 3 sources in priority order: (1) Twitter place tag → (2) user profile location → (3) text extraction. Returns the best available location and its source.
- `normalize(location)` — Cleans and normalizes location strings so that "Downtown Houston", "North Houston", and "Houston, TX" all map to the same cluster.

**`GeoSpatialAggregator`** — The 6-step analysis pipeline
- `cluster_tweets(tweets)` — Groups 426 tweets into ~24 clusters by normalized location
- `compute_cluster_consensus(tweets)` — Computes the weighted 5-class distribution for a cluster (how many tweets are label 0, 1, 2, 3, 4)
- `_compute_temporal_pattern(tweets)` — Analyzes timestamps: is it a burst (all within 6 hours = active event) or spread (over 3+ days = recovery phase)?
- `determine_cluster_status(consensus, temporal, tweets)` — **The 10-case decision matrix.** Evaluates conditions in order and returns the first matching case (status, severity, urgency, reason).
- `combine_actionable_info(tweets)` — Merges all actionable info from all tweets in the cluster into one combined intelligence report
- `generate_recommended_actions(consensus, info, status)` — Generates concrete response recommendations based on the 5-class distribution
- `analyze_all_clusters(tweets)` — **Main entry point.** Runs all 6 steps for every cluster and returns sorted reports.

---

### `api_client.py` (~302 lines, 12 KB) — Backend API Client

Handles all HTTP communication between the Python dashboard and the Node.js backend.

| Method | What It Does |
|---|---|
| `check_health()` | Calls `GET /api/health` to verify the backend is running |
| `get_tweets(page, limit, query)` | Calls `GET /api/tweets` to fetch paginated tweets from MongoDB |
| `get_all_tweets_for_geo()` | Fetches **all** tweets by paginating through every page (100 per request). Used by Geo Analysis. |
| `create_tweet(tweet_data)` | Calls `POST /api/tweets/create` to store a new tweet (includes location fields) |
| `classify_tweet(tweet_id, classification)` | Calls `PUT /api/tweets/:id/classify` to save model results back to MongoDB |
| `get_geo_clusters(min_size)` | Calls `GET /api/tweets/geo-aggregate` for server-side aggregation |

---

### `config.py` (66 lines, 2 KB) — Configuration

All configurable parameters in one file:
- **Model paths**: Where to find the model weights and tokenizer
- **Label mappings**: ID → name, ID → display name, ID → color
- **HITL threshold**: Confidence level below which tweets are flagged for human review
- **Geo settings**: Minimum cluster size, agreement threshold, burst window, severity scoring thresholds
- **Twitter API credentials**: For optional Twitter integration (loaded from environment variables)

---

### `token_highlighter.py` (~200 lines, 7 KB) — XAI Visualization

Converts the raw XAI importance scores into visual colors for the dashboard:
- Takes a list of `(token, score)` tuples from `Explainable_AI.py`
- Normalizes scores to a 0–1 range
- Maps each score to a color on the **white → red gradient** (low importance = white, high = red)
- Returns colored CTkLabel widgets that the dashboard renders in a row

---

### `twitter_api.py` (~200 lines, 7 KB) — Twitter API Wrapper (Optional)

Provides integration with Twitter's API v2 for fetching live tweets:
- Searches for tweets by keyword/hashtag
- Extracts tweet text, author, geolocation data
- Not required for the main workflow (dashboard works with database-only)

---

### `HITL/feedback_storage.py` (1.4 KB) — Human Feedback Storage

Stores human corrections from the HITL verification process:
- When a human reviewer corrects a model prediction, this module saves the correction
- Writes to `Data_Set/Processed_Data_Set/human_feedback.csv`
- These corrections can later be used to retrain/improve the model

---

### `__init__.py` — Package Marker

A tiny file (67 bytes) that makes the `Dashboard/` folder a Python package, allowing imports like `from Dashboard.dashboard import main`.

---

## 📁 Trained_Model/ — The AI Model

This folder contains the **trained DeLTran15 model** and all its supporting modules.

### `deltran15_minilm_fp32.pt` (90 MB) — Model Weights

The **trained parameters** (weights and biases) of the DeLTran15 neural network. This is the result of fine-tuning `all-MiniLM-L6-v2` on our disaster tweet dataset. Loaded by `model_inference.py` at startup. Without this file, the model cannot make predictions.

---

### `Model.py` (22 lines, 0.7 KB) — Model Architecture

Defines the `DeLTran15` class — the neural network structure:

```
DeLTran15(nn.Module)
├── encoder: AutoModel (all-MiniLM-L6-v2, 22M parameters)
│   └── 6 transformer layers, 384-dim hidden size
├── dropout: nn.Dropout(0.3)
│   └── Regularization during training
└── classifier: nn.Linear(384, 5)
    └── Maps 384-dim embedding to 5 class logits
```

The `forward()` method: tokenized input → transformer encoder → extract [CLS] token embedding → dropout → linear classifier → 5 logit scores.

---

### `Explainable_AI.py` (52 lines, 1.3 KB) — XAI Engine

Implements **gradient-based token importance analysis**:

1. Runs the tweet through the model (forward pass)
2. Computes gradients of the predicted class score (backward pass)
3. Calculates the L2 norm of each token's embedding vector
4. Higher norm = more important token (more influence on the prediction)
5. Returns `[(token, score), ...]` for visual highlighting

This tells users **which words** the model focused on, building trust in the AI's decisions.

---

### `Actionable_Info.py` (114 lines, 3.5 KB) — Information Extraction

Extracts **structured actionable intelligence** from tweet text using NLP:

| What It Extracts | Method | Example |
|---|---|---|
| 📍 Locations | spaCy NER (GPE/LOC/FAC entities) + regex fallback | "Manila Bay", "Houston, TX" |
| 👥 People Counts | Regex patterns for casualty phrases | "30 trapped" → `{count: 30, status: "trapped"}` |
| 🆘 Needs | Keyword matching against a predefined needs vocabulary | "food", "water", "shelter" |
| 💥 Damage Types | Keyword matching against damage vocabulary | "bridge", "collapsed", "flooded" |
| ⏰ Time Mentions | Regex patterns for temporal references | "now", "2 hours ago", "today" |

---

### `Main.py` (~60 lines, 2.4 KB) — Standalone CLI Classifier

A **command-line interface** for classifying tweets without the GUI dashboard. You type a tweet, it prints the prediction, confidence scores, XAI explanation, and actionable info. Useful for quick testing.

```bash
python Trained_Model/Main.py
# Enter tweet: "30 people trapped after bridge collapse in Manila"
# → Label: Infrastructure Damage (85%)
```

---

### `Model_Tokenizer/` — HuggingFace Tokenizer Files

Contains the tokenizer configuration files for `all-MiniLM-L6-v2`:
- `tokenizer_config.json` — Tokenizer settings
- `vocab.txt` — Token vocabulary (30,522 word pieces)
- `special_tokens_map.json` — Special tokens ([CLS], [SEP], [MASK])
- `tokenizer.json` — Fast tokenizer data

These files are loaded by `model_inference.py` to convert tweet text into token IDs that the model can process.

---

## 📁 backend/ — Node.js Server

The backend is a **REST API server** that stores and retrieves tweets from MongoDB. It does NOT run any AI — it's purely data infrastructure.

### `server.js` (1.3 KB) — Express Server

The **entry point** of the backend. It:
- Creates an Express app
- Connects to MongoDB using the connection string from `.env`
- Mounts the tweet routes at `/api/tweets`
- Starts listening on port 5000

---

### `models/Tweet.js` — MongoDB Schema

Defines the **data structure** for tweets stored in MongoDB using Mongoose. Every tweet document has:
- `text`, `author`, `tweetId` — Core tweet data
- `classification` — Model prediction (label ID, confidence scores, timestamp)
- `explanation` — XAI token importance scores
- `actionableInfo` — Extracted locations, needs, damage, people, time
- `status` — HITL status ("pending", "verified", "unverified", "human_verified")
- `location`, `placeTag`, `userProfileLocation` — Geospatial fields for clustering
- `createdAt`, `updatedAt` — Timestamps

---

### `routes/tweets.js` (~220 lines) — API Routes

Defines all the HTTP endpoints:

| Endpoint | What It Does |
|---|---|
| `POST /create` | Stores a new tweet. Resolves location using 3-tier priority (placeTag → userProfileLocation → fallback). |
| `GET /` | Fetches tweets with pagination, optional text search, and filtering. Returns `{tweets, totalPages, currentPage}`. |
| `PUT /:id/classify` | Saves classification results (prediction, XAI explanation, actionable info) for a specific tweet. Called by the dashboard after model inference. |
| `DELETE /:id` | Deletes a single tweet. |
| `GET /geo-aggregate` | Performs server-side MongoDB aggregation: groups tweets by `location` field, counts tweets per location, returns cluster summaries. |
| `GET /health` | Returns `{status: "ok"}` — used by the dashboard to check if the backend is reachable. |

---

### `package.json` — Node.js Dependencies

Lists the backend's npm packages: `express`, `mongoose`, `cors`, `dotenv`.

### `.env.example` — Environment Template

Template for the `.env` file showing required variables (`MONGODB_URI`, `PORT`).

---

## 📁 Data_Set/ — Training Data Pipeline

### `Unprocessed_Data_Sets/` — Raw Disaster Tweet Datasets

Seven `.tsv` files containing raw, unprocessed disaster tweet data from real events:

| File | Disaster Event | Size |
|---|---|---|
| `hurricane_harvey_final_data.tsv` | Hurricane Harvey (2017, Texas) | 1.6 MB |
| `hurricane_irma_final_data.tsv` | Hurricane Irma (2017, Caribbean/Florida) | 1.6 MB |
| `hurricane_maria_final_data.tsv` | Hurricane Maria (2017, Puerto Rico) | 1.6 MB |
| `california_wildfires_final_data.tsv` | California Wildfires (2017) | 600 KB |
| `mexico_earthquake_final_data.tsv` | Mexico Earthquake (2017) | 530 KB |
| `iraq_iran_earthquake_final_data.tsv` | Iraq-Iran Earthquake (2017) | 230 KB |
| `srilanka_floods_final_data.tsv` | Sri Lanka Floods (2017) | 370 KB |

### `Data_Preprocessing/` — Cleaning Scripts

| File | What It Does |
|---|---|
| `Data_Extraction.py` | Reads the raw `.tsv` files, extracts the `text` and `label` columns, and combines them into a single intermediate CSV (`disaster_text_only.csv`). |
| `Data_Cleaning.py` | Takes the extracted CSV and cleans the text: removes URLs, strips special characters, expands contractions, normalizes whitespace. Outputs `disaster_text_clean.csv`. |

### `Processed_Data_Set/` — Clean Training Data

| File | What It Does |
|---|---|
| `disaster_text_only.csv` | Intermediate file: extracted text + labels, before cleaning (2.2 MB) |
| `disaster_text_clean.csv` | **Final training data**: cleaned text + labels, ready for model training (1.4 MB) |
| `human_feedback.csv` | Stores human corrections from HITL verification. Used to improve the model. |

---

## 📁 Model_Build/ — Model Training

### `Build.py` — Fine-Tuning Script

The script that **trains the DeLTran15 model**:
1. Loads `disaster_text_clean.csv`
2. Splits into train/validation/test sets
3. Creates data loaders with the `all-MiniLM-L6-v2` tokenizer
4. Fine-tunes the `DeLTran15` model using cross-entropy loss + AdamW optimizer
5. Tracks validation accuracy, saves best checkpoint
6. Outputs `Trained_Model/deltran15_minilm_fp32.pt`

---

## 📁 reports/ — Model Evaluation

### `generate_test_tweets_report.py` (19 KB) — Report Generator

A comprehensive script that evaluates the trained model on test data and generates:
- Classification metrics (accuracy, F1-score, precision, recall)
- Confusion matrix
- Confidence distribution
- Actionable info coverage statistics
- Saves all results as CSVs and visualizations (PNG charts)

### `test_tweets_report/` — Generated Reports

| File | What It Contains |
|---|---|
| `summary_metrics.json` | Overall accuracy (87%), macro/weighted F1 scores |
| `classification_report_labeled.csv` | Per-class precision, recall, F1-score |
| `confusion_matrix_labeled.csv` | Actual vs predicted counts for all 5×5 class combinations |
| `predictions_per_tweet.csv` | Every test tweet with its predicted vs actual label |
| `confidence_stats.csv` | Confidence score statistics (mean, median, std dev) |
| `fig_confusion_matrix_labeled.png` | Visual confusion matrix (heatmap chart) |
| `fig_expected_vs_predicted_labeled.png` | Bar chart comparing expected vs predicted distributions |
| `fig_confidence_distribution.png` | Histogram of model confidence scores |
| `fig_actionable_field_coverage.png` | Bar chart showing NLP extraction coverage |
| `fig_predicted_class_distribution.png` | Pie/bar chart of predicted class distribution |
| `REPORT_SUMMARY.md` | Human-readable summary of the evaluation results |

---

## Summary — File Count

| Folder | Files | Purpose |
|---|---|---|
| **Root** | 14 | Entry point, test data, documentation |
| **Dashboard/** | 9 | Desktop GUI, model wrapper, geo aggregator, API client |
| **Trained_Model/** | 5 + tokenizer | AI model, XAI, NLP extraction, CLI |
| **backend/** | 6 | REST API server, MongoDB schema, routes |
| **Data_Set/** | 12 | Raw datasets, cleaning scripts, processed training data |
| **Model_Build/** | 1 | Model training script |
| **reports/** | 15 | Evaluation metrics, charts, CSVs |
| **Total** | **~62 files** | |
