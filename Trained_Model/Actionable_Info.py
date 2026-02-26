import re

_SPACY_ERROR = None
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception as exc:
    _NLP = None
    _SPACY_ERROR = str(exc)

NEEDS_KEYWORDS = {
    "food", "water", "medicine", "medical", "shelter", "blankets", "clothes",
    "rescue", "ambulance", "evacuation", "supplies", "aid", "donation"
}

DAMAGE_KEYWORDS = {
    "bridge", "road", "highway", "power", "electricity", "collapsed", "flooded",
    "fire", "wildfire", "damaged", "destroyed", "outage", "blocked", "landslide", "roads", "floods"
}

TIME_PATTERNS = [
    r"\bnow\b",
    r"\btoday\b",
    r"\byesterday\b",
    r"\btonight\b",
    r"\bthis\s+morning\b",
    r"\bthis\s+afternoon\b",
    r"\bthis\s+evening\b",
    r"\b\d+\s*(mins?|minutes?|hrs?|hours?)\s*ago\b",
]

PEOPLE_COUNT_PATTERN = r"\b(\d+)\s+(injured|dead|killed|missing|trapped|wounded)\b"


def _extract_locations(text):
    locations = []

    if _NLP is not None:
        doc = _NLP(text)
        for ent in doc.ents:
            if ent.label_ in {"GPE", "LOC", "FAC"}:
                locations.append(ent.text)

    # Fallback heuristic if spaCy isn't available or returns nothing.
    if not locations:
        preposition_pattern = (
            r"\b(?:in|near|at|around|from|on|outside|inside|within|across|to)\s+"
            r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)"
        )
        city_state_pattern = r"\b([A-Z][a-zA-Z]+),\s*([A-Z]{2})\b"
        highway_pattern = r"\b(Highway\s+\d+|Route\s+\d+|I-\d+)\b"

        for match in re.findall(preposition_pattern, text):
            locations.append(match)
        for city, state in re.findall(city_state_pattern, text):
            locations.append(f"{city}, {state}")
        for match in re.findall(highway_pattern, text):
            locations.append(match)

    return sorted(set(locations))


def _extract_people_counts(text):
    matches = re.findall(PEOPLE_COUNT_PATTERN, text, flags=re.IGNORECASE)
    results = []
    for count, status in matches:
        results.append({"count": int(count), "status": status.lower()})
    return results


def _extract_time_mentions(text):
    hits = []
    for pattern in TIME_PATTERNS:
        for match in re.findall(pattern, text, flags=re.IGNORECASE):
            hits.append(match)
    return sorted(set(hits))


def _extract_keywords(text, keywords):
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return sorted(set(t for t in tokens if t in keywords))


def extract_actionable_info(text):
    info = {
        "locations": _extract_locations(text),
        "people_count": _extract_people_counts(text),
        "needs": _extract_keywords(text, NEEDS_KEYWORDS),
        "damage_type": _extract_keywords(text, DAMAGE_KEYWORDS),
        "time_mentions": _extract_time_mentions(text),
    }
    if _NLP is None and _SPACY_ERROR:
        info["location_note"] = "spaCy model not available; using rule-based fallback"
    return info
