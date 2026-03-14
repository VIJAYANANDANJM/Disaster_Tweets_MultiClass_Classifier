# Geospatial Temporal Aggregation — Complete Technical Documentation

> **Feature**: Location-aware, multi-tweet consensus classification using the 5-class multi-class disaster tweet model.
> **Status**: ✅ Complete — All components implemented and tested.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Data Flow — End to End](#3-data-flow--end-to-end)
4. [Backend — Schema & API](#4-backend--schema--api)
5. [Location Resolution — 3-Tier Priority](#5-location-resolution--3-tier-priority)
6. [GeoSpatialAggregator — The 6-Step Pipeline](#6-geospatialaggregator--the-6-step-pipeline)
7. [The 10-Case Decision Matrix (In Detail)](#7-the-10-case-decision-matrix-in-detail)
8. [Severity Scoring Algorithm](#8-severity-scoring-algorithm)
9. [Actionable Intelligence Merging](#9-actionable-intelligence-merging)
10. [Recommended Actions Generation](#10-recommended-actions-generation)
11. [The Final Cluster Report — Complete Structure](#11-the-final-cluster-report--complete-structure)
12. [Dashboard UI — GeoAnalysisView](#12-dashboard-ui--geoanalysisview)
13. [Configuration Parameters](#13-configuration-parameters)
14. [Mock Dataset — All 10 Cases](#14-mock-dataset--all-10-cases)
15. [Files & Code Map](#15-files--code-map)

---

## 1. Problem Statement

Our system classifies each tweet into one of **5 humanitarian categories**:

| ID | Category | Emoji | Meaning | Example Tweet |
|---|---|---|---|---|
| 0 | Affected Individuals | 🔴 | People hurt, trapped, displaced, missing | "Family trapped on roof in Houston, need rescue NOW" |
| 1 | Infrastructure & Utility Damage | 🟠 | Roads, bridges, power, water damaged | "Highway 59 bridge collapsed, avoid the area" |
| 2 | Not Humanitarian | ⚪ | Irrelevant (food, sports, casual chat) | "Beautiful sunset in Paris today" |
| 3 | Other Relevant Information | 🔵 | Warnings, updates, awareness | "Earthquake alert issued for Manila region" |
| 4 | Rescue/Volunteering/Donation | 🟢 | Help offers, donations, relief | "Red Cross shelter open at NRG Center, need volunteers" |

### The Problem

Classifying a **single tweet** tells us what category *that tweet* is. But for a government or NGO to act, they need a **location-level picture**:

> *"What is the overall situation in Houston right now?"*

The answer lies in the **distribution** across all 5 classes from many tweets **at the same location**, not from one isolated prediction.

### What Geo-Aggregation Provides

For each location, it groups 20-60+ tweets and shows the **5-class distribution**:

```
Houston Cluster (54 tweets, 14 authors):
  🔴 Affected Individuals:     12 tweets (22%) → People need rescue
  🟠 Infrastructure Damage:    11 tweets (20%) → Roads, power damaged
  ⚪ Not Humanitarian:          8 tweets (15%) → Noise filtered out
  🔵 Other Information:        12 tweets (22%) → Official updates
  🟢 Rescue/Donation:          11 tweets (20%) → Relief is underway

  Status: CONFIRMED EVENT
  Severity: MEDIUM
  → Actions: Deploy rescue teams + engineers + coordinate donations
```

This gives **actionable decision-making intelligence**, not just a label.

---

## 2. Architecture Overview

```
Tweet Sources (Twitter API / Manual Input / Database)
        ↓
┌─────────────────────────────────────────────────────┐
│  Backend (Node.js + Express + MongoDB)              │
│  ├── Tweet.js schema: text, author, location fields │
│  ├── POST /api/tweets/create: store tweet + location│
│  ├── PUT /api/tweets/:id/classify: store model result│
│  └── GET /api/tweets: paginated tweet retrieval     │
└─────────────────────────┬───────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────┐
│  Python Dashboard (CustomTkinter)                   │
│  ├── api_client.py: GET all tweets via pagination   │
│  ├── model_inference.py: classify unclassified tweets│
│  ├── geospatial_aggregator.py: THE CORE LOGIC       │
│  │   ├── Step 1: Cluster by location                │
│  │   ├── Step 2: Compute 5-class consensus          │
│  │   ├── Step 3: Temporal pattern analysis           │
│  │   ├── Step 4: Apply 10-case decision matrix      │
│  │   ├── Step 5: Combine actionable intelligence    │
│  │   └── Step 6: Generate recommended actions       │
│  └── dashboard.py: GeoAnalysisView renders results  │
└─────────────────────────────────────────────────────┘
```

---

## 3. Data Flow — End to End

Here is exactly what happens when a user clicks the **"🌍 Geo Analysis"** button:

### Step 1: User clicks the button
- `dashboard.py` → `MainDashboard.show_geo_analysis()` is called
- The main dashboard frame is hidden, `GeoAnalysisView` is shown
- `GeoAnalysisView.load_clusters()` starts automatically

### Step 2: Fetch all tweets from MongoDB
- A background worker thread starts
- Calls `api_client.get_all_tweets_for_geo()` which paginates through:
  ```
  GET /api/tweets?page=1&limit=100  → 100 tweets
  GET /api/tweets?page=2&limit=100  → 100 tweets
  GET /api/tweets?page=3&limit=100  → 100 tweets
  GET /api/tweets?page=4&limit=100  → 100 tweets
  GET /api/tweets?page=5&limit=100  →  26 tweets
  Total: 426 tweets
  ```
- All tweets are collected into a single Python list

### Step 3: Classify unclassified tweets
- The worker checks each tweet: does `classification.predictedLabelId` exist?
- If `None` → tweet hasn't been classified yet
- Creates `ModelInference()` → loads the XtremeDistil model
- For each unclassified tweet, calls `model.classify_tweet(text)`:
  - Tokenizes the text
  - Runs through the transformer model
  - Gets probability distribution across 5 classes
  - Picks highest probability as prediction
  - Runs XAI (attention-based explanation)
  - Runs Actionable Info extraction (spaCy NER for locations, needs, damage, people)
- Updates the tweet dict in-place with classification results
- Shows live progress: "Classifying 42/426 tweets..."

### Step 4: Run GeoSpatialAggregator
- Creates `GeoSpatialAggregator(min_cluster_size=1)`
- Calls `aggregator.analyze_all_clusters(all_426_tweets)`
- This runs the **6-step pipeline** (explained in detail in Section 6)
- Returns a list of cluster report dicts, sorted by severity

### Step 5: Render in the UI
- Main thread receives the reports via polling
- Left panel: renders cluster cards (`_make_cluster_card()`)
- Right panel: renders full detail for selected cluster (`_show_detail()`)
- First cluster is auto-selected

---

## 4. Backend — Schema & API

### 4.1 Tweet Schema (`backend/models/Tweet.js`)

Six new location-related fields were added to the existing Tweet model:

```javascript
// Existing fields
text: String,          // The tweet text
author: String,        // Twitter handle
tweetId: String,       // Unique tweet ID
classification: {      // Model prediction results
  predictedLabelId: Number,    // 0-4
  predictedLabel: String,      // e.g., "Affected Individuals"
  confidenceScores: [Number],  // [0.85, 0.05, 0.03, 0.04, 0.03]
},
actionableInfo: Object, // Extracted actionable data

// NEW location fields (added for Geo Analysis)
location: { type: String, index: true },  // Resolved location (indexed for queries)
locationSource: String,     // "place_tag" | "user_profile" | "text_extraction"
userProfileLocation: String, // User's self-declared location
placeTag: String,            // Twitter geo-tagged place name
placeCountry: String,        // Country from Twitter place tag
geoCoordinates: {            // Optional GPS coordinates
  type: { type: String },
  coordinates: [Number]
}
```

### 4.2 API Endpoints

**POST `/api/tweets/create`**
- Accepts all location fields in the request body
- Uses 3-tier priority to set `location` and `locationSource`:
  1. `placeTag` → location = placeTag, source = "place_tag"
  2. `userProfileLocation` → location = userProfileLocation, source = "user_profile"
  3. Falls back to empty

**PUT `/api/tweets/:id/classify`**
- Saves classification results + updated location data

**GET `/api/tweets/geo-aggregate`** (new endpoint)
- Aggregates tweets in MongoDB using `$group` by `location`
- Returns pre-computed cluster summary (used as alternative to client-side aggregation)
- Parameters: `?minClusterSize=5`

---

## 5. Location Resolution — 3-Tier Priority

**File:** `Dashboard/geospatial_aggregator.py` → `class LocationResolver`

Every tweet can have location information from multiple sources. The resolver picks the most reliable one:

| Priority | Source | Field | Reliability | Example |
|---|---|---|---|---|
| 🥇 1st | Twitter Place Tag | `placeTag` | High — GPS-based | "Manila, Philippines" |
| 🥈 2nd | User Profile Location | `userProfileLocation` | Medium — self-declared | "Houston, TX" |
| 🥉 3rd | NER Text Extraction | `actionableInfo.locations[0]` | Variable — NLP-based | "flooding in Tokyo" → "Tokyo" |

### How Resolution Works

```python
def resolve(tweet_data):
    # Tier 1: Place tag (from Twitter geo)
    place_tag = tweet_data.get("placeTag", "")
    if place_tag and place_tag.strip():
        return place_tag.strip(), "place_tag"

    # Tier 2: User profile location
    profile_loc = tweet_data.get("userProfileLocation", "")
    if profile_loc and profile_loc.strip():
        return profile_loc.strip(), "user_profile"

    # Tier 3: Text extraction
    location_field = tweet_data.get("location", "")
    if location_field and location_field.strip():
        return location_field.strip(), "text_extraction"
    
    # Check actionableInfo.locations
    actionable = tweet_data.get("actionableInfo", {})
    locations = actionable.get("locations", [])
    if locations:
        return locations[0], "text_extraction"

    return "", ""  # No location found
```

### Normalization

After resolving, the location is normalized for consistent grouping:

```python
def normalize(location_str):
    # "Downtown Houston" → "houston"
    # "Houston, TX"      → "houston, tx"
    # "North Manila"     → "manila"
    
    loc = location_str.strip().lower()
    
    # Strip directional prefixes
    for prefix in ["downtown", "near", "north", "south", "east", "west",
                   "central", "greater", "upper", "lower", "outer", "inner"]:
        if loc.startswith(prefix + " "):
            loc = loc[len(prefix) + 1:]
    
    loc = re.sub(r"\s+", " ", loc).strip()
    return loc
```

This ensures that "Downtown Houston", "North Houston", and "Houston" all cluster together as `"houston"`.

---

## 6. GeoSpatialAggregator — The 6-Step Pipeline

**File:** `Dashboard/geospatial_aggregator.py` → `class GeoSpatialAggregator`

The main entry point is `analyze_all_clusters(tweets)` which runs these 6 steps for every location cluster:

### Step 1: `cluster_tweets(tweets)` — Group by Location

**What it does:** Takes all 426 tweets, resolves each tweet's location, normalizes it, and groups tweets with the same normalized location together.

**Input:** List of 426 tweet dicts
**Output:** Dictionary like:
```python
{
    "manila, philippines": [tweet1, tweet2, ..., tweet25],   # 25 tweets
    "houston, tx":         [tweet3, tweet4, ..., tweet56],   # 54 tweets
    "berlin, germany":     [tweet100],                        # 1 tweet
    ...                                                       # 24 clusters total
}
```

**Important:** Tweets with no resolvable location are silently dropped (they don't appear in any cluster).

---

### Step 2: `compute_cluster_consensus(cluster_tweets)` — 5-Class Distribution

**What it does:** For each cluster, counts how many tweets fall into each of the 5 classes, weighted by the model's confidence.

**Algorithm:**
1. Separate classified tweets (have `predictedLabelId`) from unclassified ones
2. For each classified tweet:
   - Get its `predictedLabelId` (0-4)
   - Get its confidence score for that label
   - Add 1 to that label's count
   - Add the confidence score to that label's weighted total
3. Compute:
   - `percentage` = (count / total_classified) × 100
   - `weighted_percentage` = (weighted_sum / total_weighted_sum) × 100
   - `agreement_score` = count_of_primary_label / total_classified
   - `humanitarian_percentage` = (tweets NOT label 2) / total × 100
   - `avg_confidence` = sum of all confidences / total_classified

**Example Output for Manila (25 tweets):**
```python
{
    "total_tweets": 25,
    "classified_count": 25,
    "unique_authors": 11,
    "label_distribution": {
        0: {"count": 2,  "percentage": 8.0,  "weighted": 1.7},   # Affected
        1: {"count": 7,  "percentage": 28.0, "weighted": 5.9},   # Infrastructure
        2: {"count": 0,  "percentage": 0.0,  "weighted": 0.0},   # Not Humanitarian
        3: {"count": 9,  "percentage": 36.0, "weighted": 7.5},   # Other Info
        4: {"count": 7,  "percentage": 28.0, "weighted": 5.8},   # Rescue/Donation
    },
    "primary_label_id": 3,       # Other Info has highest weighted
    "agreement_score": 0.36,     # Only 36% agree on label 3
    "avg_confidence": 0.88,
    "humanitarian_percentage": 100.0,  # 0% is "Not Humanitarian"
}
```

---

### Step 3: `_compute_temporal_pattern(cluster_tweets)` — Time Analysis

**What it does:** Parses all tweet timestamps, calculates the time span from earliest to latest, and determines if there's a temporal burst or spread.

**Definitions:**
- **Burst** ⚡: All tweets within ≤6 hours → suggests an **active, ongoing event**
- **Spread** 📆: Tweets span >72 hours (3+ days) → suggests a **recovery/evolving situation**
- **Normal**: Everything in between

**Output:**
```python
{
    "time_span_hours": 2.0,      # Earliest tweet to latest tweet
    "is_burst": True,            # 2.0 hours ≤ 6 hours
    "is_spread": False,          # 2.0 hours ≤ 72 hours
    "earliest": "2025-12-15T08:00:00+00:00",
    "latest":   "2025-12-15T10:00:00+00:00",
}
```

---

### Step 4: `determine_cluster_status(consensus, temporal, tweets)` — 10-Case Decision Matrix

**This is the core intelligence.** See Section 7 for the full 10-case breakdown.

**Input:** The consensus, temporal, and raw tweets from Steps 1-3
**Output:**
```python
{
    "status": "active_event",                                    # Case name
    "reason": "25 tweets within 2.0 hours. Ongoing event!",     # Human-readable why
    "severity": "CRITICAL",                                      # CRITICAL/HIGH/MEDIUM/LOW
    "urgency": "IMMEDIATE",                                      # Action urgency
}
```

---

### Step 5: `combine_actionable_info(cluster_tweets)` — Merge Intelligence

**What it does:** Takes the `actionableInfo` from every tweet in the cluster and merges them into a single intelligence summary.

See Section 9 for details.

---

### Step 6: `generate_recommended_actions(consensus, combined_info, status)` — Actions

**What it does:** Maps the 5-class distribution to concrete response recommendations.

See Section 10 for details.

---

### Final Assembly: `generate_cluster_report()`

All 6 steps are combined into a single report dict. See Section 11 for the complete structure.

---

## 7. The 10-Case Decision Matrix (In Detail)

**File:** `Dashboard/geospatial_aggregator.py` → `determine_cluster_status()` (Lines 293–446)

Cases are evaluated **in order** — the **first matching case wins**. This order is intentional: edge cases (insufficient data, spam) are checked first, confirmed events last.

---

### Case 1: Insufficient Data
```
Condition: total_tweets == 1
```
| Field | Value |
|---|---|
| Status | `insufficient_data` |
| Severity | UNKNOWN |
| Urgency | NONE |
| Reason | "Only 1 tweet from this location. Cannot verify from a single source." |

**Why:** A single tweet could be anything — a joke, a misclassification, or genuine. You cannot make a location-level assessment from one data point.

**Example:** 1 tweet from Berlin says "Earthquake destroyed buildings" → Could be real, could be metaphorical. Need more tweets to verify.

---

### Case 2: Suspicious Single Source
```
Condition: unique_authors == 1 AND total_tweets > 1
```
| Field | Value |
|---|---|
| Status | `suspicious_single_source` |
| Severity | UNKNOWN |
| Urgency | FLAG |
| Reason | "All {N} tweets are from the same author. No independent corroboration." |

**Why:** If one person posts 5 tweets about a disaster in Sydney, it could be:
- A bot spamming
- Someone retweeting themselves
- A genuine witness — but we have zero independent verification

**Example:** 5 tweets from Sydney, all authored by @disaster_bot → Flagged, not trusted.

---

### Case 3: Early Signal
```
Condition: total_tweets < min_cluster_size (default 5) AND unique_authors >= 2
```
| Field | Value |
|---|---|
| Status | `early_signal` |
| Severity | LOW |
| Urgency | MONITOR |
| Reason | "Only {N} tweets from {M} authors. May be emerging event." |

**Why:** 3-4 tweets from different people is promising but not conclusive. It could be the very beginning of an event that will generate more tweets over time.

**Example:** 4 tweets from Mumbai, 4 different authors, 3 say "flooding" → Worth monitoring, not enough to deploy resources.

---

### Case 10: Location Uncertain
```
Condition: total_tweets >= 5 AND >60% of authors have profile locations
           that don't match the cluster location
```
| Field | Value |
|---|---|
| Status | `location_uncertain` |
| Severity | UNCERTAIN |
| Urgency | VERIFY |
| Reason | "{X}% of authors have profiles from other locations. Possible remote reporting." |

**Why:** If tweets about "Seoul earthquake" are mostly from people whose profiles say "New York", "London", "Tokyo" — they're reporting about Seoul from afar, not on the ground. Their information is less reliable.

**Example:** 15 tweets about Seoul, but 10 authors' profiles say they're based in other cities → 67% remote reporting → Location uncertain.

> **Note:** This case is checked early (before Cases 4–9) because it can disqualify what would otherwise look like a confirmed event.

---

### Unclassified (not numbered)
```
Condition: classified_count == 0 (no tweets have been run through the model)
```
| Field | Value |
|---|---|
| Status | `unclassified` |
| Severity | PENDING |
| Urgency | CLASSIFY |
| Reason | "{N} tweets found but none classified yet. Run classification first." |

**Why:** The aggregator needs model predictions to make decisions. Without them, it can't compute any distribution.

---

### Case 6: No Disaster (False Alarm)
```
Condition: "Not Humanitarian" (label 2) percentage >= 80%
```
| Field | Value |
|---|---|
| Status | `no_disaster` |
| Severity | NONE |
| Urgency | DISMISS |
| Reason | "{X}% of tweets classified as 'Not Humanitarian'. Normal social chatter." |

**Why:** If 80%+ of tweets from a location are "Not Humanitarian" (label 2), people are just chatting — there's no disaster.

**Example:** 25 tweets from Paris → 88% are label 2 ("Beautiful day in Paris", "Having coffee at Eiffel Tower") → No disaster, dismiss.

---

### Case 7: Low Model Confidence
```
Condition: average confidence < 50%
```
| Field | Value |
|---|---|
| Status | `low_confidence_cluster` |
| Severity | UNCERTAIN |
| Urgency | HUMAN_REVIEW |
| Reason | "Average model confidence is {X}%. Tweets may be ambiguous." |

**Why:** If the model is only 40-50% sure about its predictions, the 5-class distribution is unreliable. A human analyst needs to review the tweets manually.

**Example:** 20 tweets from Jakarta, model gives 45% confidence on average → Classifications are unreliable → Human review needed.

---

### Case 5: Ambiguous Distribution
```
Condition: agreement_score < agreement_threshold (default 0.5)
           AND humanitarian_percentage > 30%
```
| Field | Value |
|---|---|
| Status | `ambiguous_needs_review` |
| Severity | MEDIUM |
| Urgency | HUMAN_REVIEW |
| Reason | "No class has clear majority ({X}% agreement). Mixed signals from this location." |

**Why:** When no single class has >50% of the tweets, the model's predictions are evenly split — it can't determine what type of disaster (or non-disaster) this is. Needs a human to decide.

**Example:** 30 tweets from Istanbul → 20% each across all 5 classes → Every class is equally likely → Ambiguous.

---

### Case 8: Active Event (Temporal Burst) ⚡
```
Condition: is_burst == True (all tweets within 6 hours)
           AND humanitarian_percentage > 70%
           AND total_tweets >= 10
```
| Field | Value |
|---|---|
| Status | `active_event` |
| Severity | **CRITICAL** |
| Urgency | **IMMEDIATE** |
| Reason | "{N} tweets within {X} hours. Ongoing event detected!" |

**Why:** This is the most urgent case. A sudden burst of humanitarian tweets from one location means a disaster is **actively happening right now**. Immediate response is needed.

**Example:** 25 tweets from Manila within 2 hours → 100% humanitarian content → Real-time disaster → **CRITICAL / IMMEDIATE**

---

### Case 9: Recovery Phase
```
Condition: is_spread == True (tweets span > 72 hours)
           AND classified >= 10
           AND recovery labels (3,4) percentage > rescue labels (0,1) percentage
```
| Field | Value |
|---|---|
| Status | `recovery_phase` |
| Severity | MEDIUM |
| Urgency | SHIFT_RESPONSE |
| Reason | "Tweets span {X}+ hours. Recovery tweets ({Y}%) exceed active disaster ({Z}%)." |

**Why:** When tweets about a location shift from "people trapped" and "bridge collapsed" (labels 0,1) to "donations needed" and "updates" (labels 3,4) over several days, the disaster has moved from **rescue phase to recovery phase**. Response strategy should shift accordingly.

**Example:** 30 tweets from Mexico City over 5 days:
- Day 1: Mostly label 0 (Affected) + label 1 (Infrastructure)
- Day 5: Mostly label 4 (Rescue/Donation) + label 3 (Info)
→ Recovery phase → Shift from rescue to rebuilding

---

### Case 4: Confirmed Event
```
Condition: total >= min_cluster_size AND unique_authors >= min_unique_authors
           (AND none of the above cases matched)
```
| Field | Value |
|---|---|
| Status | `confirmed_event` |
| Severity | Computed (see Section 8) |
| Urgency | ALERT |
| Reason | "{N} tweets from {M} independent authors. {X}% humanitarian content." |

**Why:** This is the "default positive" case — enough tweets from enough different people to trust the classification distribution. The specific severity depends on the scoring algorithm.

**Example:** 54 tweets from Houston, 14 authors, 82% humanitarian → Confirmed event with MEDIUM severity.

---

### Evaluation Order Summary

```
Case 1: Insufficient Data      (total == 1)
Case 2: Suspicious Source       (1 author, many tweets)
Case 3: Early Signal            (too few tweets)
Case 10: Location Uncertain     (>60% remote authors)
─── Unclassified ───            (no classifications)
Case 6: No Disaster             (≥80% "Not Humanitarian")  
Case 7: Low Confidence          (avg confidence < 50%)
Case 5: Ambiguous               (low agreement, humanitarian)
Case 8: Active Event            (temporal burst + humanitarian)
Case 9: Recovery Phase          (spread + recovery > rescue)
Case 4: Confirmed Event         (enough tweets + authors)
─── Fallback ───                (none matched)
```

---

## 8. Severity Scoring Algorithm

**File:** `geospatial_aggregator.py` → `_compute_severity()` (Lines 448–500)

Only applies to **Case 4 (confirmed_event)**. Other cases have fixed severities.

### Point System

| Factor | Low (1 pt) | Medium (2 pts) | High (3 pts) |
|---|---|---|---|
| Tweet count | < 20 | 20–49 | ≥ 50 |
| People affected | < 100 | 100–499 | ≥ 500 |
| Humanitarian % | < 50% | 50–79% (+1) | ≥ 80% (+2) |
| Affected Individuals % | < 30% | — | ≥ 30% (+2) |
| Temporal burst | No | — | Yes (+2) |

### Score → Severity Mapping

| Total Score | Severity |
|---|---|
| ≥ 8 | **CRITICAL** 🔴 |
| 5–7 | **HIGH** 🟠 |
| 3–4 | **MEDIUM** 🟡 |
| 0–2 | **LOW** 🟢 |

### Example Calculation (Manila, 25 tweets)

```
Tweet count: 25 → ≥20 but <50 → +2 points
People affected: 150 → ≥100 but <500 → +2 points
Humanitarian %: 100% → ≥80% → +2 points
Affected Individuals: 8% → <30% → +0 points
Temporal burst: Yes (2.0 hours) → +2 points
─────────────────────────────────
Total: 8 points → CRITICAL
```

---

## 9. Actionable Intelligence Merging

**File:** `geospatial_aggregator.py` → `combine_actionable_info()` (Lines 505–545)

For each tweet in a cluster, the model extracts `actionableInfo` containing:
- `locations`: place names mentioned in the text (via spaCy NER)
- `needs`: what people need (food, water, shelter, medicine)
- `damageType`: what was damaged (bridge, road, building)
- `peopleCount`: estimated number of people affected
- `timeMentions`: time references (now, immediately, today)

The aggregator takes **all** these from all tweets in the cluster and **unions** them:

```python
# From tweet 1: {"locations": ["Manila"], "needs": ["food", "water"]}
# From tweet 2: {"locations": ["Manila Bay", "Tondo"], "needs": ["shelter", "rescue"]}
# From tweet 3: {"damageType": ["bridge", "flooded"], "peopleCount": [{"count": 50}]}
# ...

# Combined result:
{
    "locations": ["Manila", "Manila Bay", "Manila City Hall", "Tondo", "Tondo district"],
    "total_people_affected": 150,        # sum of all people counts
    "needs": ["food", "rescue", "shelter", "water"],
    "damage_types": ["bridge", "destroyed", "flooded", "power"],
    "time_mentions": ["NOW", "immediately", "now"],
}
```

This gives the government/NGO a **complete intelligence picture** from all tweets combined — not just one tweet's perspective.

---

## 10. Recommended Actions Generation

**File:** `geospatial_aggregator.py` → `generate_recommended_actions()` (Lines 550–605)

Actions are generated based on the **5-class distribution percentages** and the **cluster status**:

### Label-Based Actions (triggered if label ≥ 15%)

| Label | Threshold | Action Generated |
|---|---|---|
| 🔴 Label 0 (Affected) | ≥ 15% | "🔴 RESCUE: Deploy search & rescue teams to {locations}" |
| 🟠 Label 1 (Infrastructure) | ≥ 15% | "🟠 ENGINEERS: Dispatch repair teams for {damage_types}" |
| 🟢 Label 4 (Rescue/Donation) | ≥ 15% | "🟢 LOGISTICS: Coordinate {needs} distribution" |
| 🔵 Label 3 (Other Info) | ≥ 15% | "🔵 COMMS: Issue public advisories and situation updates" |

### Status-Based Actions

| Status | Action |
|---|---|
| `active_event` | "🚨 IMMEDIATE: Real-time disaster in progress — activate all response protocols" (inserted at top) |
| `recovery_phase` | "🔄 TRANSITION: Shift from rescue operations to recovery and rebuilding" |
| CRITICAL or HIGH severity | "⚡ PRIORITY: Escalate to national emergency response coordination" |

### Special Cases

| Status | Actions |
|---|---|
| `no_disaster` / `insufficient_data` / `suspicious_single_source` | ["No action required at this time."] |
| `early_signal` | ["Monitor this location.", "Set up automated alerts."] |
| `low_confidence_cluster` | ["Assign human analyst.", "Request ground reports."] |

---

## 11. The Final Cluster Report — Complete Structure

**File:** `geospatial_aggregator.py` → `generate_cluster_report()` (Lines 610–660)

This is the exact dict returned for each cluster:

```python
{
    # ─── IDENTITY ───
    "location": "Manila, Philippines",          # Display name (from first tweet's resolved location)
    "normalized_location": "manila, philippines", # Normalized key used for clustering

    # ─── DECISION (from 10-case matrix) ───
    "status": "active_event",                    # Case name (string)
    "status_reason": "25 tweets within 2.0 hours. Ongoing event detected!",
    "severity": "CRITICAL",                      # CRITICAL / HIGH / MEDIUM / LOW / UNCERTAIN / UNKNOWN / NONE / PENDING
    "urgency": "IMMEDIATE",                      # IMMEDIATE / ALERT / HUMAN_REVIEW / MONITOR / VERIFY / FLAG / NONE / etc.

    # ─── CONSENSUS (from 5-class analysis) ───
    "total_tweets": 25,                          # Total tweets in this cluster
    "classified_count": 25,                      # How many have model predictions
    "unique_authors": 11,                        # How many different people tweeted
    "primary_label_id": 3,                       # Highest weighted class (0-4)
    "primary_label": "Other Information",        # Human-readable name
    "agreement_score": 0.36,                     # % of tweets matching primary class (0.0-1.0)
    "avg_confidence": 0.88,                      # Average model confidence (0.0-1.0)
    "humanitarian_percentage": 100.0,            # % of tweets NOT "Not Humanitarian"
    "label_distribution": {                      # Full 5-class breakdown
        0: {"count": 2,  "percentage": 8.0,  "weighted": 1.7,  "weighted_percentage": 8.1,
            "label_name": "Affected Individuals", "color": "#E74C3C"},
        1: {"count": 7,  "percentage": 28.0, "weighted": 5.9,  "weighted_percentage": 28.2,
            "label_name": "Infrastructure Damage", "color": "#E67E22"},
        2: {"count": 0,  "percentage": 0.0,  "weighted": 0.0,  "weighted_percentage": 0.0,
            "label_name": "Not Humanitarian",      "color": "#95A5A6"},
        3: {"count": 9,  "percentage": 36.0, "weighted": 7.5,  "weighted_percentage": 35.9,
            "label_name": "Other Information",     "color": "#3498DB"},
        4: {"count": 7,  "percentage": 28.0, "weighted": 5.8,  "weighted_percentage": 27.8,
            "label_name": "Rescue/Donation",       "color": "#27AE60"},
    },

    # ─── TEMPORAL ───
    "temporal": {
        "time_span_hours": 2.0,
        "is_burst": True,
        "is_spread": False,
        "earliest": "2025-12-15T08:00:00+00:00",
        "latest":   "2025-12-15T10:00:00+00:00",
    },

    # ─── COMBINED INTELLIGENCE (merged from all tweets) ───
    "combined_actionable_info": {
        "locations": ["Manila", "Manila Bay", "Manila City Hall", "Tondo", "Tondo district"],
        "total_people_affected": 150,
        "people_details": [{"count": 50, "status": "trapped"}, {"count": 100, "status": "displaced"}],
        "needs": ["food", "rescue", "shelter", "water"],
        "damage_types": ["bridge", "destroyed", "flooded", "power"],
        "time_mentions": ["NOW", "immediately", "now"],
    },

    # ─── RECOMMENDED ACTIONS ───
    "recommended_actions": [
        "🚨 IMMEDIATE: Real-time disaster in progress — activate all response protocols",
        "🔴 RESCUE: Deploy search & rescue teams to Manila, Manila Bay, Manila City Hall, Tondo, Tondo district",
        "🟠 ENGINEERS: Dispatch repair teams for bridge, destroyed, flooded, power",
        "🟢 LOGISTICS: Coordinate food, rescue, shelter, water distribution",
        "🔵 COMMS: Issue public advisories and situation updates",
        "⚡ PRIORITY: Escalate to national emergency response coordination"
    ],
}
```

---

## 12. Dashboard UI — GeoAnalysisView

**File:** `Dashboard/dashboard.py` → `class GeoAnalysisView`

### Layout

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 🌍 Geo Analysis        [24 clusters loaded]  [🔄 Refresh] [← Back]    │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Location Clusters│ 🚨 Manila, Philippines                              │
│                  │ Severity: CRITICAL  Urgency: IMMEDIATE  Case: Active│
│ ┌──────────────┐ │ 25 tweets within 2.0 hours. Ongoing event detected! │
│ │🚨 Manila     │ │                                                     │
│ │25t · 11a  CRI│ │ ┌─────┬────┬────┬─────┬────┐                      │
│ └──────────────┘ │ │  25 │ 25 │ 11 │100% │88% │                      │
│ ┌──────────────┐ │ │Tweet│Clas│Auth│Hum. │Conf│                      │
│ │✅ Houston    │ │ └─────┴────┴────┴─────┴────┘                      │
│ │54t · 14a  MED│ │                                                     │
│ └──────────────┘ │ 5-Class Distribution                                │
│ ┌──────────────┐ │ 🔴 Affected     ████                   2 (8.0%)   │
│ │✅ Miami      │ │ 🟠 Infrastructure███████████           7 (28.0%)   │
│ │54t · 14a  MED│ │ ⚪ Not Humanitarian                    0 (0.0%)    │
│ └──────────────┘ │ 🔵 Other Info    ██████████████        9 (36.0%)   │
│        ...       │ 🟢 Rescue/Donation ██████████          7 (28.0%)   │
│                  │                                                     │
│                  │ Combined Actionable Intelligence                     │
│                  │ 📍 Locations: Manila, Manila Bay, Tondo            │
│                  │ 🆘 Needs: food, rescue, shelter, water             │
│                  │ 💥 Damage: bridge, destroyed, flooded              │
│                  │                                                     │
│                  │ Recommended Actions                                  │
│                  │ ┌─────────────────────────────────────┐             │
│                  │ │ 🚨 IMMEDIATE: Real-time disaster... │             │
│                  │ │ 🔴 RESCUE: Deploy search & rescue...│             │
│                  │ │ 🟠 ENGINEERS: Dispatch repair...    │             │
│                  │ └─────────────────────────────────────┘             │
└──────────────────┴───────────────────────────────────────────────────────┘
```

### UI Components

**Left Panel — Cluster Cards (`_make_cluster_card()`)**
- Each card shows: status icon, location name, severity badge, tweet count, author count
- Color-coded border based on severity
- Clickable — clicking a card loads its full report on the right

**Right Panel — Detail View (`_show_detail()`)**
- **Header**: Location name with status icon
- **Badges**: Severity (color-coded), Urgency, Case name (purple badge)
- **Status reason**: Gray text explaining why this case was triggered
- **Stats row**: Total tweets, classified, authors, humanitarian %, avg confidence
- **5-Class Distribution**: Horizontal bars with percentages for each label
- **Combined Actionable Intelligence**: Merged locations, needs, damage, people, time
- **Recommended Actions**: Dark cards with action strings
- **Temporal**: Time span and burst/spread indicator

### Thread Safety

The UI uses a **polling pattern** instead of `self.after()` from background threads (required for Python 3.13 compatibility):

1. Worker thread sets shared variables: `_geo_status_text`, `_geo_reports`, `_geo_done`
2. Main thread polls every 300ms via `_poll_geo()`
3. When `_geo_done == True`, the main thread renders the results

---

## 13. Configuration Parameters

**File:** `Dashboard/config.py`

```python
# Minimum tweets needed to form a "real" cluster (Case 3 threshold)
GEO_MIN_CLUSTER_SIZE = 5

# Agreement threshold — if primary class has less than 50% of tweets,
# the distribution is considered "ambiguous" (Case 5)
GEO_AGREEMENT_THRESHOLD = 0.5

# Minimum unique authors for a confirmed event (Case 4)
GEO_MIN_UNIQUE_AUTHORS = 3

# Temporal burst window — all tweets within this many hours = "burst" (Case 8)
GEO_TEMPORAL_BURST_HOURS = 6

# Severity scoring thresholds (Case 4 only)
GEO_SEVERITY_THRESHOLDS = {
    "tweet_count_high": 50,      # ≥50 tweets → +3 severity points
    "tweet_count_medium": 20,    # ≥20 tweets → +2 severity points
    "people_high": 500,          # ≥500 people affected → +3 severity points
    "people_medium": 100,        # ≥100 people affected → +2 severity points
}
```

---

## 14. Mock Dataset — All 10 Cases

**File:** `mock_geo_tweets.py` / `mock_geo_tweets.json`

The mock dataset contains **~426 tweets** across **14 locations**, specifically designed to trigger all 10 decision cases:

| Location | Tweets | Authors | Target Case | Expected Status |
|---|---|---|---|---|
| Houston, TX | 54 | 15 | Case 4 | `confirmed_event` |
| San Francisco, CA | 55 | 15 | Case 4 | `confirmed_event` |
| Miami, FL | 54 | 15 | Case 4 | `confirmed_event` |
| London, England | 54 | 15 | Case 4 | `confirmed_event` |
| Tokyo, Japan | 54 | 15 | Case 4 | `confirmed_event` |
| Berlin, Germany | 1 | 1 | Case 1 | `insufficient_data` |
| Sydney, NSW | 5 | 1 | Case 2 | `suspicious_single_source` |
| Mumbai, India | 4 | 4 | Case 3 | `early_signal` |
| Istanbul, Turkey | 30 | 12 | Case 5 | `ambiguous_needs_review` |
| Paris, France | 25 | 12 | Case 6 | `no_disaster` |
| Jakarta, Indonesia | 20 | 10 | Case 7 | `low_confidence_cluster` |
| Manila, Philippines | 25 | 15 | Case 8 | `active_event` |
| Mexico City, Mexico | 30 | 12 | Case 9 | `recovery_phase` |
| Seoul, South Korea | 15 | 15 | Case 10 | `location_uncertain` |
| **TOTAL** | **~426** | | | |

> **Note:** The actual case triggered at runtime may differ slightly from the "target" because the model's real-time classifications affect the 5-class distribution, which in turn affects which decision case matches. The mock data is designed to push each cluster toward its target case.

---

## 15. Files & Code Map

| File | Lines | Role |
|---|---|---|
| `Dashboard/geospatial_aggregator.py` | 684 | **Core logic** — `LocationResolver` + `GeoSpatialAggregator` |
| `Dashboard/dashboard.py` | ~1270 | **UI** — `GeoAnalysisView` class (cluster cards + detail panel) |
| `Dashboard/api_client.py` | ~302 | **API** — `get_all_tweets_for_geo()`, `get_geo_clusters()` |
| `Dashboard/model_inference.py` | ~260 | **ML** — `classify_tweet()`, `classify_tweet_cluster()` |
| `Dashboard/config.py` | ~58 | **Config** — geo thresholds and label mappings |
| `backend/models/Tweet.js` | ~70 | **Schema** — 6 location fields added |
| `backend/routes/tweets.js` | ~220 | **API** — `/geo-aggregate` endpoint, location in create/classify |
| `mock_geo_tweets.py` | ~varies | **Test data** — generates 426 tweets for all 10 cases |

---

## Progress

- [x] Design 10-case decision framework (multi-class aware)
- [x] Create mock Twitter dataset (14 locations, ~426 tweets, all 10 cases)
- [x] Backend: Add location fields to Tweet schema
- [x] Backend: Add geo-aggregate API endpoint
- [x] Python: Create `geospatial_aggregator.py` module (LocationResolver + GeoSpatialAggregator)
- [x] Python: Update `model_inference.py` with `classify_tweet_cluster()` method
- [x] Python: Update `config.py` with geospatial settings
- [x] Python: Update `api_client.py` with geo API methods + location fields
- [x] Dashboard: Add `GeoAnalysisView` class with cluster visualization
- [x] Dashboard: Fix thread safety for Python 3.13 (polling pattern)
- [x] Dashboard: Add auto-classification in Geo Analysis view
- [x] Verification: Syntax checks + import verification + live testing
