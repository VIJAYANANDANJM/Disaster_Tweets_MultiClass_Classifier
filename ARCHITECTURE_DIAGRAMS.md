# Architecture Diagrams

This document contains PlantUML code for the project's architecture diagrams.
Copy the code into [PlantUML Online Editor](https://www.plantuml.com/plantuml/uml) to render.

---

## 0. Quick Overview — Simple Block Diagram

This diagram gives a **quick, at-a-glance** understanding of the entire project flow.

```plantuml
@startuml Quick_Overview
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam rectangleBackgroundColor #16213e
skinparam rectangleBorderColor #533483
skinparam arrowColor #e94560
skinparam titleFontSize 22
skinparam defaultFontSize 13
skinparam rectangleRoundCorner 15

title Disaster Tweet Classification — Quick Overview

rectangle "**📥 INPUT**\nManual Tweet Entry\nor Database Fetch" as INPUT #0f3460

rectangle "**🧠 AI MODEL**\nDeLTran15\n(MiniLM + 5-Class Head)\n87% Accuracy" as MODEL #e94560

rectangle "**🔍 XAI**\nGradient-Based\nToken Highlighting\n(Why this label?)" as XAI #533483

rectangle "**📋 NLP EXTRACTION**\nspaCy + Regex\n📍Locations 👥People\n🆘Needs 💥Damage ⏰Time" as NLP #533483

rectangle "**🔄 HITL**\nLow Confidence?\n→ Human Reviews\n→ Corrects Label" as HITL #E67E22

rectangle "**🌍 GEO ANALYSIS**\nGroup by Location\n10-Case Decision\nSituation Reports" as GEO #27AE60

rectangle "**📊 DASHBOARD**\nTweet List + Filters\nXAI Visualization\nCluster Reports" as DASH #3498DB

rectangle "**🗄️ BACKEND**\nNode.js + Express\nMongoDB Storage\nREST API" as BACKEND #95A5A6

INPUT -right-> MODEL : "tweet text"
MODEL -right-> XAI : "prediction\n+ embeddings"
MODEL -down-> NLP : "if disaster\nlabel"

XAI -down-> DASH : "token\nimportance"
NLP -right-> DASH : "actionable\ninfo"

MODEL -down-> HITL : "if conf < 70%"
HITL -right-> DASH : "corrected\nlabel"

DASH -down-> GEO : "all classified\ntweets"
DASH -right-> BACKEND : "sync results"
BACKEND -up-> DASH : "fetch tweets"

note bottom of MODEL
  **5 Categories:**
  🔴 Affected Individuals
  🟠 Infrastructure Damage
  ⚪ Not Humanitarian
  🔵 Other Information
  🟢 Rescue / Donation
end note

note bottom of GEO
  **10-Case Decision Matrix**
  → Severity & Urgency
  → Recommended Actions
  → Per-location reports
end note

@enduml
```

---

## 1. System Architecture Diagram

```plantuml
@startuml System_Architecture
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam componentBackgroundColor #16213e
skinparam componentBorderColor #0f3460
skinparam packageBackgroundColor #0f3460
skinparam packageBorderColor #533483
skinparam arrowColor #e94560
skinparam noteBorderColor #533483
skinparam noteBackgroundColor #16213e
skinparam titleFontSize 24
skinparam titleFontColor #e94560

title Disaster Tweet Multi-Class Classification System\nComplete Architecture

package "User's Local Machine" as LOCAL #0f3460 {

    package "Desktop Dashboard (Python / CustomTkinter)" as DASHBOARD #16213e {
        component [dashboard.py\n---\nMainDashboard\nConnectionFrame\nTweetInputFrame\nTweetCard\nTweetDetailFrame\nGeoAnalysisView] as UI
        component [model_inference.py\n---\nModelInference\n- predict()\n- get_explanation()\n- get_actionable_info()\n- classify_tweet()] as INFERENCE
        component [geospatial_aggregator.py\n---\nLocationResolver\nGeoSpatialAggregator\n- 6-Step Pipeline\n- 10-Case Decision Matrix] as GEO
        component [api_client.py\n---\nAPIClient\n- get_tweets()\n- create_tweet()\n- classify_tweet()\n- get_all_tweets_for_geo()] as API_CLIENT
        component [config.py\n---\nModel Paths\nLabel Maps\nGeo Thresholds\nHITL Threshold] as CONFIG
        component [token_highlighter.py\n---\nXAI Color Mapping\nWhite→Red Gradient] as HIGHLIGHTER
        component [HITL/feedback_storage.py\n---\nHuman Corrections\nCSV Storage] as HITL
    }

    package "Trained Model (PyTorch)" as MODEL_PKG #16213e {
        component [DeLTran15\n(Model.py)\n---\nall-MiniLM-L6-v2\n+ Linear(384→5)] as MODEL
        component [Explainable_AI.py\n---\nGradient-Based XAI\nToken Importance\nL2 Norm Scoring] as XAI
        component [Actionable_Info.py\n---\nspaCy NER\nRegex Extraction\nLocations | Needs\nDamage | People | Time] as ACTIONABLE
        database "deltran15_minilm_fp32.pt\n(90 MB Weights)" as WEIGHTS
        folder "Model_Tokenizer/\nvocab.txt\ntokenizer.json" as TOKENIZER
    }
}

package "Backend Server (Node.js / Express)" as BACKEND #533483 {
    component [server.js\n---\nExpress App\nPort 5000\nCORS Middleware] as SERVER
    component [routes/tweets.js\n---\nPOST /create\nGET /\nPUT /:id/classify\nDELETE /:id\nGET /geo-aggregate\nGET /health] as ROUTES
    component [models/Tweet.js\n---\nMongoose Schema\nClassification Fields\nLocation Fields\nHITL Status Fields] as SCHEMA
}

database "MongoDB\n(Atlas / Local)\n---\ntweets collection" as DB #e94560

' === Relationships ===

' Dashboard internal
UI --> INFERENCE : "classify tweet"
UI --> GEO : "geo analysis"
UI --> API_CLIENT : "HTTP calls"
UI --> HIGHLIGHTER : "render XAI"
UI --> HITL : "save corrections"
UI ..> CONFIG : "read settings"

' Model inference
INFERENCE --> MODEL : "forward pass"
INFERENCE --> XAI : "explain prediction"
INFERENCE --> ACTIONABLE : "extract info"
MODEL --> WEIGHTS : "load weights"
MODEL --> TOKENIZER : "tokenize text"

' Geo aggregator
GEO --> INFERENCE : "classify unclassified"

' Network
API_CLIENT -down-> ROUTES : "HTTP REST\n(localhost:5000)"

' Backend internal
SERVER --> ROUTES
ROUTES --> SCHEMA
SCHEMA -down-> DB : "Mongoose ODM"

' Notes
note right of MODEL
  **DeLTran15 Architecture**
  Base: all-MiniLM-L6-v2 (22M params)
  + Dropout(0.3)
  + Linear(384 → 5 classes)
  Output: 5-class probabilities
end note

note right of GEO
  **6-Step Pipeline**
  1. Cluster by location
  2. Compute consensus
  3. Temporal analysis
  4. 10-case decision
  5. Merge intelligence
  6. Recommend actions
end note

note bottom of DB
  Stores: tweets, classifications,
  XAI explanations, actionable info,
  HITL status, geo location fields
end note

@enduml
```

---

## 2. Data Flow Diagram

```plantuml
@startuml Data_Flow
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam activityBackgroundColor #16213e
skinparam activityBorderColor #0f3460
skinparam arrowColor #e94560
skinparam titleFontSize 20

title Data Flow — Tweet Classification Pipeline

start

:Tweet enters the system;
note right
  Sources:
  - Manual input (dashboard)
  - Database fetch
  - Mock data (mock_geo_tweets.py)
end note

:Tokenize text\n(all-MiniLM-L6-v2 tokenizer);
:Input IDs + Attention Mask;

:Forward Pass\n(DeLTran15 model);
note right: 6 transformer layers\n384-dim [CLS] embedding

:Softmax → 5 probability scores;

if (max confidence ≥ 70%?) then (yes)
  :Status = **"verified"**;
  #27AE60:Auto-accepted ✅;
else (no)
  :Status = **"unverified"**;
  #E74C3C:Flagged for HITL ⚠️;
  :Human reviews + corrects;
  :Status = **"human_verified"**;
endif

:Predicted Label;
note right
  0 = Affected Individuals 🔴
  1 = Infrastructure Damage 🟠
  2 = Not Humanitarian ⚪
  3 = Other Information 🔵
  4 = Rescue/Donation 🟢
end note

fork
  :Gradient Backward Pass;
  :Token importance scores\n(L2 norm per token);
  :XAI Highlighting\n(white → red heat map);
fork again
  if (Label is 0, 1, or 4?) then (yes)
    :spaCy NER + Regex;
    :Extract:\n📍 Locations\n👥 People counts\n🆘 Needs\n💥 Damage types\n⏰ Time mentions;
  else (no)
    :Skip extraction\n(not actionable);
  endif
end fork

:Complete Result;
note right
  {
    label_id, label_name,
    confidence_scores[5],
    explanation[(token, score)...],
    actionable_info{...},
    status
  }
end note

:Sync to MongoDB\n(PUT /api/tweets/:id/classify);

:Display in Dashboard\n(TweetCard + TweetDetailFrame);

stop

@enduml
```

---

## 3. Geospatial Analysis Pipeline

```plantuml
@startuml Geo_Pipeline
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam activityBackgroundColor #16213e
skinparam activityBorderColor #0f3460
skinparam arrowColor #e94560
skinparam titleFontSize 20

title Geospatial Temporal Aggregation — 6-Step Pipeline

start

:Fetch ALL tweets from MongoDB;
note right: api_client.get_all_tweets_for_geo()

if (All tweets classified?) then (yes)
  :Use cached classifications;
else (no)
  :Classify unclassified tweets\n(ModelInference);
  :Sync to backend\n(PUT /classify);
  :Cache in memory\n(_cached_tweets);
endif

partition "Step 1: Cluster Tweets" {
  :Group by normalized location;
  note right
    LocationResolver:
    Priority: placeTag > userProfile > text
    Normalize: "Downtown Houston" → "houston"
  end note
  :→ N location clusters;
}

partition "Step 2: Compute Consensus" {
  :For each cluster, compute\nweighted 5-class distribution;
  :confidence_scores[label] × weight;
  :→ {label_id: percentage} per cluster;
}

partition "Step 3: Temporal Analysis" {
  :Analyze tweet timestamps;
  if (time span ≤ 6 hours?) then (yes)
    :Pattern = **BURST** ⚡;
  elseif (time span > 72 hours?) then (yes)
    :Pattern = **SPREAD** 📅;
  else (neither)
    :Pattern = **NORMAL**;
  endif
}

partition "Step 4: 10-Case Decision Matrix" {
  :Evaluate in order:;
  :1. Insufficient Data (1 tweet) → UNKNOWN;
  :2. Suspicious Source (1 author) → UNKNOWN;
  :3. Early Signal (<5 tweets) → LOW;
  :10. Location Uncertain (>60% remote) → UNCERTAIN;
  :6. No Disaster (≥80% non-hum.) → NONE;
  :7. Low Confidence (avg <50%) → UNCERTAIN;
  :5. Ambiguous (low agreement) → MEDIUM;
  :8. Active Event (burst+hum.) → **CRITICAL** 🚨;
  :9. Recovery Phase (spread+recov.) → MEDIUM;
  :4. Confirmed Event → **COMPUTED**;
  note right
    First matching case wins.
    Severity scoring for Case 4
    uses point-based algorithm.
  end note
}

partition "Step 5: Merge Intelligence" {
  :Combine actionable info\nfrom ALL tweets in cluster;
  :Deduplicate locations, needs,\ndamage types, people counts;
  :→ unified intelligence report;
}

partition "Step 6: Recommend Actions" {
  :Map dominant labels → actions;
  note right
    Label 0 → "Deploy rescue teams"
    Label 1 → "Send engineers"
    Label 4 → "Coordinate donations"
  end note
}

:Generate Cluster Report;
note right
  {location, tweet_count,
  unique_authors, consensus,
  status, severity, urgency,
  reason, actionable_info,
  recommended_actions}
end note

:Sort by severity (CRITICAL first);
:Render in GeoAnalysisView;

stop

@enduml
```

---

## 4. Component Block Diagram

```plantuml
@startuml Block_Diagram
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam rectangleBackgroundColor #16213e
skinparam rectangleBorderColor #0f3460
skinparam arrowColor #e94560
skinparam titleFontSize 22
skinparam defaultFontSize 12

title Component Block Diagram

rectangle "**DATA LAYER**" as DATA #0f3460 {
    rectangle "MongoDB Atlas\n(tweets collection)" as MONGO #533483
    rectangle "Raw Datasets\n(.tsv files × 7)" as RAW #533483
    rectangle "Processed CSV\n(disaster_text_clean.csv)" as CLEAN #533483
    rectangle "Human Feedback\n(human_feedback.csv)" as FEEDBACK #533483
}

rectangle "**DATA PREPROCESSING**" as PREPROCESS #0f3460 {
    rectangle "Data_Extraction.py\nExtract text + labels" as EXTRACT #533483
    rectangle "Data_Cleaning.py\nURL/emoji/whitespace cleanup" as CLEANING #533483
}

rectangle "**MODEL TRAINING**" as TRAIN_BLOCK #0f3460 {
    rectangle "Build.py\nFine-tune MiniLM\n→ deltran15_minilm_fp32.pt" as BUILD #533483
}

rectangle "**AI INFERENCE ENGINE**" as AI #0f3460 {
    rectangle "DeLTran15\n(Model.py)\nTransformer + Linear(384→5)" as DL_MODEL #e94560
    rectangle "XAI Engine\n(Explainable_AI.py)\nGradient Token Importance" as XAI_BLOCK #e94560
    rectangle "NLP Extractor\n(Actionable_Info.py)\nspaCy NER + Regex" as NLP #e94560
    rectangle "Model Inference\n(model_inference.py)\nOrchestrator" as ORCH #e94560
}

rectangle "**GEOSPATIAL ENGINE**" as GEO_BLOCK #0f3460 {
    rectangle "LocationResolver\nNormalize + Resolve" as LOC_RES #533483
    rectangle "GeoSpatialAggregator\n6-Step Pipeline\n10-Case Decision Matrix" as AGG #533483
}

rectangle "**BACKEND API**" as API #0f3460 {
    rectangle "Express Server\n(server.js)\nPort 5000" as EXPRESS #533483
    rectangle "Tweet Routes\n(tweets.js)\nCRUD + Geo Aggregate" as ROUTE_BLOCK #533483
    rectangle "Tweet Schema\n(Tweet.js)\nMongoose ODM" as SCHEMA_BLOCK #533483
}

rectangle "**DESKTOP UI**" as UI_BLOCK #0f3460 {
    rectangle "Main Dashboard\nTweet List + Filters\nTweet Detail + XAI" as MAIN_UI #27AE60
    rectangle "Tweet Input\nText + Author\nLocation + Profile Location" as INPUT_UI #27AE60
    rectangle "Geo Analysis View\nCluster Cards\nDetail Reports" as GEO_UI #27AE60
    rectangle "HITL Verification\nTo Verify Queue\nCorrection Popup" as HITL_UI #27AE60
}

rectangle "**EVALUATION**" as EVAL #0f3460 {
    rectangle "Report Generator\n(generate_test_tweets_report.py)\nConfusion Matrix, F1, Charts" as REPORT #533483
}

' === Data flow ===
RAW -down-> EXTRACT
EXTRACT -down-> CLEANING
CLEANING -down-> CLEAN
CLEAN -down-> BUILD
BUILD -down-> DL_MODEL

ORCH --> DL_MODEL : predict
ORCH --> XAI_BLOCK : explain
ORCH --> NLP : extract

MAIN_UI --> ORCH : classify_tweet()
GEO_UI --> AGG : analyze_all_clusters()
AGG --> LOC_RES : resolve location
AGG --> ORCH : classify unclassified

MAIN_UI --> ROUTE_BLOCK : HTTP REST
GEO_UI --> ROUTE_BLOCK : HTTP REST
INPUT_UI --> ROUTE_BLOCK : POST /create

ROUTE_BLOCK --> SCHEMA_BLOCK
SCHEMA_BLOCK --> MONGO

HITL_UI --> FEEDBACK : save correction
HITL_UI --> ROUTE_BLOCK : PUT /classify

REPORT --> DL_MODEL : evaluate

@enduml
```

---

## 5. Class Diagram — Dashboard

```plantuml
@startuml Class_Diagram
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam classBorderColor #0f3460
skinparam classBackgroundColor #16213e
skinparam classHeaderBackgroundColor #533483
skinparam arrowColor #e94560
skinparam titleFontSize 20

title Dashboard Class Diagram

class MainDashboard {
    - api_client: APIClient
    - model_inference: ModelInference
    - tweets: list
    - current_filter: int
    - hitl_mode: bool
    --
    + setup_ui()
    + setup_dashboard()
    + on_connection_success()
    + show_geo_analysis()
    + show_main_dashboard()
    + handle_manual_tweet(tweet_data)
    + classify_single_tweet(tweet_data)
    + save_tweet_to_db(tweet_data)
    + load_tweets_from_db()
    + classify_tweets_locally(tweets)
    + display_tweets()
    + filter_tweets(label_id)
    + filter_hitl()
    + filter_verified()
    + select_tweet(tweet)
}

class ConnectionFrame {
    - api_client: APIClient
    - on_success: callback
    --
    + setup_ui()
    + check_backend_status()
    - _poll_result()
}

class TweetInputFrame {
    - on_submit: callback
    - tweet_text: CTkTextbox
    - author_entry: CTkEntry
    - location_entry: CTkEntry
    - profile_location_entry: CTkEntry
    --
    + setup_ui()
    + submit_tweet()
}

class TweetCard {
    - tweet_data: dict
    - classification: dict
    - on_select: callback
    --
    + setup_ui()
}

class TweetDetailFrame {
    - token_highlighter
    - api_client: APIClient
    - dashboard_ref: MainDashboard
    --
    + setup_ui()
    + display_tweet(tweet_data)
    + sync_to_db(tweet_data)
    + open_correction_popup(tweet, label_id)
}

class GeoAnalysisView {
    - api_client: APIClient
    - back_callback: callback
    - reports: list
    - _cached_tweets: list
    --
    + load_clusters(force_refresh)
    - _poll_geo()
    - _render_clusters(reports)
    - _make_cluster_card(report)
    - _show_detail(report)
}

class ModelInference {
    - model: DeLTran15
    - tokenizer
    + loaded: bool
    --
    + predict(text): (id, scores, name)
    + get_explanation(text): [(token,score)]
    + get_actionable_info(text, label): dict
    + classify_tweet(text): dict
}

class APIClient {
    - base_url: str
    --
    + check_health(): bool
    + get_tweets(page, limit): tuple
    + get_all_tweets_for_geo(): tuple
    + create_tweet(data): dict
    + classify_tweet(id, data): dict
}

class GeoSpatialAggregator {
    - min_cluster_size: int
    --
    + cluster_tweets(tweets): dict
    + compute_cluster_consensus(tweets): dict
    + determine_cluster_status(...): dict
    + combine_actionable_info(tweets): dict
    + generate_recommended_actions(...): list
    + analyze_all_clusters(tweets): list
}

class LocationResolver {
    --
    + resolve(tweet): (location, source)
    + normalize(location): str
}

class DeLTran15 {
    - encoder: AutoModel
    - dropout: nn.Dropout
    - classifier: nn.Linear
    --
    + forward(input_ids, mask): logits
}

' Relationships
MainDashboard *-- ConnectionFrame
MainDashboard *-- TweetInputFrame
MainDashboard *-- TweetDetailFrame
MainDashboard *-- GeoAnalysisView
MainDashboard o-- ModelInference
MainDashboard o-- APIClient
MainDashboard --> TweetCard : creates

GeoAnalysisView o-- APIClient
GeoAnalysisView --> GeoSpatialAggregator : uses
GeoSpatialAggregator --> LocationResolver : uses

ModelInference --> DeLTran15 : loads
TweetDetailFrame o-- APIClient

@enduml
```

---

## 6. Sequence Diagram — Manual Tweet Classification

```plantuml
@startuml Sequence_Manual_Tweet
!theme cerulean-outline
skinparam backgroundColor #1a1a2e
skinparam defaultFontColor #e0e0e0
skinparam sequenceArrowColor #e94560
skinparam sequenceLifeLineBorderColor #0f3460
skinparam sequenceParticipantBackgroundColor #16213e
skinparam sequenceParticipantBorderColor #533483
skinparam titleFontSize 18

title Manual Tweet Classification Flow

actor User
participant "TweetInputFrame" as INPUT
participant "MainDashboard" as DASH
participant "ModelInference" as MODEL
participant "DeLTran15" as DL
participant "Explainable_AI" as XAI
participant "Actionable_Info" as ACT
participant "APIClient" as API
participant "Backend\n(Express)" as SERVER
database "MongoDB" as DB

User -> INPUT : type tweet + location\nclick "Classify"
INPUT -> INPUT : validate text
INPUT -> DASH : handle_manual_tweet(tweet_data)

DASH -> DASH : start background thread

group Background Thread
    DASH -> MODEL : classify_tweet(text)

    MODEL -> DL : forward(input_ids, mask)
    DL --> MODEL : logits [5 scores]
    MODEL -> MODEL : softmax → probabilities
    MODEL -> MODEL : check confidence vs 0.7

    MODEL -> XAI : explain_prediction(text)
    XAI -> DL : forward + backward pass
    XAI --> MODEL : [(token, importance), ...]

    MODEL -> ACT : extract_actionable_info(text)
    ACT -> ACT : spaCy NER + regex
    ACT --> MODEL : {locations, needs, ...}

    MODEL --> DASH : {label, scores,\nexplanation, actionable_info}

    DASH -> API : create_tweet(tweet_data)
    API -> SERVER : POST /api/tweets/create
    SERVER -> DB : insert document
    DB --> SERVER : saved
    SERVER --> API : {_id, ...}
end

DASH -> DASH : _poll_manual_tweet()
DASH -> DASH : tweets.insert(0, tweet_data)
DASH -> DASH : display_tweets()
DASH -> DASH : select_tweet(tweet_data)

DASH --> User : Tweet appears at TOP\nwith XAI + actionable info

@enduml
```

---

## How to Render

1. **Online**: Go to [plantuml.com/plantuml/uml](https://www.plantuml.com/plantuml/uml), paste any code block above
2. **VS Code**: Install "PlantUML" extension, open this file, `Alt+D` to preview
3. **CLI**: `java -jar plantuml.jar ARCHITECTURE_DIAGRAMS.md` (renders all diagrams)
4. **IntelliJ**: Built-in PlantUML support with the PlantUML Integration plugin
