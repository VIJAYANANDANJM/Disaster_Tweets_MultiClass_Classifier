"""
Geospatial Temporal Aggregator Module
Clusters tweets by location and computes multi-class consensus for disaster assessment.

Classes:
    LocationResolver  — resolves and normalizes tweet locations (3-tier priority)
    GeoSpatialAggregator — clusters, computes consensus, applies decision rules, generates reports
"""
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

from Dashboard.config import (
    LABEL_MAP, LABEL_DISPLAY_NAMES, LABEL_COLORS,
    GEO_MIN_CLUSTER_SIZE, GEO_AGREEMENT_THRESHOLD,
    GEO_MIN_UNIQUE_AUTHORS, GEO_TEMPORAL_BURST_HOURS,
    GEO_SEVERITY_THRESHOLDS
)


# =============================================================================
# LOCATION RESOLVER
# =============================================================================
class LocationResolver:
    """Resolves and normalizes tweet locations using the 3-tier priority system.
    
    Priority:
        1. placeTag     — Twitter geo-tagged place (most reliable)
        2. userProfileLocation — user's self-declared profile location
        3. text extraction     — NER/regex extracted from tweet text
    """

    # Common prefixes/suffixes to strip for normalization
    STRIP_PREFIXES = [
        "downtown", "near", "north", "south", "east", "west",
        "central", "greater", "upper", "lower", "outer", "inner",
    ]

    @staticmethod
    def resolve(tweet_data):
        """Resolve location from a tweet dict using 3-tier priority.
        
        Args:
            tweet_data: dict with keys placeTag, userProfileLocation, 
                        actionableInfo (with .locations list), text
        
        Returns:
            tuple: (resolved_location_str, source_tier)
                   source_tier is one of: 'place_tag', 'user_profile', 'text_extraction', ''
        """
        # Tier 1: Place tag (from Twitter geo)
        place_tag = tweet_data.get("placeTag", "")
        if place_tag and place_tag.strip():
            return place_tag.strip(), "place_tag"

        # Tier 2: User profile location
        profile_loc = tweet_data.get("userProfileLocation", "")
        if profile_loc and profile_loc.strip():
            return profile_loc.strip(), "user_profile"

        # Tier 3: Text extraction (from actionableInfo or location field)
        location_field = tweet_data.get("location", "")
        if location_field and location_field.strip():
            return location_field.strip(), "text_extraction"
        
        actionable = tweet_data.get("actionableInfo", {})
        if isinstance(actionable, dict):
            locations = actionable.get("locations", [])
            if locations:
                return locations[0], "text_extraction"

        return "", ""

    @classmethod
    def normalize(cls, location_str):
        """Normalize a location string for clustering.
        
        Examples:
            'Houston, TX'      → 'houston, tx'
            'Downtown Houston' → 'houston'
            'London, England'  → 'london, england'
        """
        if not location_str:
            return ""

        loc = location_str.strip().lower()

        # Strip common directional/qualifier prefixes
        for prefix in cls.STRIP_PREFIXES:
            if loc.startswith(prefix + " "):
                loc = loc[len(prefix) + 1:]

        # Remove extra whitespace
        loc = re.sub(r"\s+", " ", loc).strip()
        return loc


# =============================================================================
# GEOSPATIAL AGGREGATOR
# =============================================================================
class GeoSpatialAggregator:
    """Clusters tweets by location, computes 5-class consensus, applies decision rules,
    and generates actionable cluster reports for government/NGO use.
    """

    def __init__(self, min_cluster_size=None, agreement_threshold=None,
                 min_unique_authors=None, temporal_burst_hours=None):
        self.min_cluster_size = min_cluster_size or GEO_MIN_CLUSTER_SIZE
        self.agreement_threshold = agreement_threshold or GEO_AGREEMENT_THRESHOLD
        self.min_unique_authors = min_unique_authors or GEO_MIN_UNIQUE_AUTHORS
        self.temporal_burst_hours = temporal_burst_hours or GEO_TEMPORAL_BURST_HOURS
        self.resolver = LocationResolver()

    # -------------------------------------------------------------------------
    # Step 1: Cluster tweets by location
    # -------------------------------------------------------------------------
    def cluster_tweets(self, tweets):
        """Group tweets by normalized resolved location.
        
        Args:
            tweets: list of tweet dicts (from backend API or local data)
            
        Returns:
            dict: {normalized_location: [tweet_dicts]}
        """
        clusters = defaultdict(list)

        for tweet in tweets:
            location, source = self.resolver.resolve(tweet)
            if not location:
                continue
            
            normalized = self.resolver.normalize(location)
            if not normalized:
                continue

            # Attach resolved location metadata to tweet
            tweet["_resolved_location"] = location
            tweet["_location_source"] = source
            tweet["_normalized_location"] = normalized
            clusters[normalized].append(tweet)

        return dict(clusters)

    # -------------------------------------------------------------------------
    # Step 2: Compute 5-class weighted consensus for a single cluster
    # -------------------------------------------------------------------------
    def compute_cluster_consensus(self, cluster_tweets):
        """Compute weighted 5-class distribution and consensus for a location cluster.
        
        Args:
            cluster_tweets: list of classified tweet dicts
            
        Returns:
            dict with keys: label_distribution, primary_label, agreement_score,
                           avg_confidence, classified_count, unique_authors, etc.
        """
        # Separate classified vs unclassified tweets
        classified = []
        for t in cluster_tweets:
            classification = t.get("classification", {})
            if classification and classification.get("predictedLabelId") is not None:
                classified.append(t)

        total_tweets = len(cluster_tweets)
        classified_count = len(classified)

        # Unique authors
        authors = set()
        for t in cluster_tweets:
            author = t.get("author", "")
            if author:
                authors.add(author)
        unique_authors = len(authors)

        # If no tweets are classified yet, return early with metadata only
        if classified_count == 0:
            return {
                "total_tweets": total_tweets,
                "classified_count": 0,
                "unique_authors": unique_authors,
                "label_distribution": {i: {"count": 0, "percentage": 0.0, "weighted": 0.0} for i in range(5)},
                "primary_label_id": None,
                "primary_label": "Unclassified",
                "agreement_score": 0.0,
                "avg_confidence": 0.0,
                "humanitarian_percentage": 0.0,
            }

        # Compute weighted votes per label
        label_weights = defaultdict(float)     # label_id -> sum of confidence weights
        label_counts = Counter()               # label_id -> raw count
        total_confidence = 0.0

        for t in classified:
            cls = t["classification"]
            label_id = cls["predictedLabelId"]
            scores = cls.get("confidenceScores", [])
            confidence = scores[label_id] if scores and label_id < len(scores) else 0.5

            label_counts[label_id] += 1
            label_weights[label_id] += confidence
            total_confidence += confidence

        # Build 5-class distribution
        total_weighted = sum(label_weights.values()) or 1.0
        label_distribution = {}
        for label_id in range(5):
            count = label_counts.get(label_id, 0)
            weighted = label_weights.get(label_id, 0.0)
            label_distribution[label_id] = {
                "count": count,
                "percentage": round((count / classified_count) * 100, 1) if classified_count else 0.0,
                "weighted": round(weighted, 2),
                "weighted_percentage": round((weighted / total_weighted) * 100, 1),
                "label_name": LABEL_DISPLAY_NAMES.get(label_id, f"Label {label_id}"),
                "color": LABEL_COLORS.get(label_id, "#666"),
            }

        # Primary class = highest weighted votes
        primary_label_id = max(label_weights, key=label_weights.get) if label_weights else 2
        
        # Agreement score = % of tweets agreeing with primary class
        primary_count = label_counts.get(primary_label_id, 0)
        agreement_score = round(primary_count / classified_count, 3) if classified_count else 0.0

        # Humanitarian percentage = % of tweets NOT classified as "Not Humanitarian" (label 2)
        non_humanitarian = classified_count - label_counts.get(2, 0)
        humanitarian_pct = round((non_humanitarian / classified_count) * 100, 1) if classified_count else 0.0

        # Average confidence
        avg_confidence = round(total_confidence / classified_count, 3) if classified_count else 0.0

        return {
            "total_tweets": total_tweets,
            "classified_count": classified_count,
            "unique_authors": unique_authors,
            "label_distribution": label_distribution,
            "primary_label_id": primary_label_id,
            "primary_label": LABEL_DISPLAY_NAMES.get(primary_label_id, f"Label {primary_label_id}"),
            "agreement_score": agreement_score,
            "avg_confidence": avg_confidence,
            "humanitarian_percentage": humanitarian_pct,
        }

    # -------------------------------------------------------------------------
    # Step 3: Compute temporal pattern
    # -------------------------------------------------------------------------
    def _compute_temporal_pattern(self, cluster_tweets):
        """Analyze the temporal distribution of tweets in a cluster.
        
        Returns:
            dict with time_span_hours, is_burst, is_spread, earliest, latest
        """
        timestamps = []
        for t in cluster_tweets:
            created = t.get("createdAt", "")
            if isinstance(created, str) and created:
                try:
                    # Handle ISO format with or without timezone
                    ts = created.replace("Z", "+00:00")
                    timestamps.append(datetime.fromisoformat(ts))
                except (ValueError, TypeError):
                    pass
            elif isinstance(created, datetime):
                timestamps.append(created)

        if len(timestamps) < 2:
            return {
                "time_span_hours": 0,
                "is_burst": False,
                "is_spread": False,
                "earliest": None,
                "latest": None,
            }

        timestamps.sort()
        earliest = timestamps[0]
        latest = timestamps[-1]
        time_span = (latest - earliest).total_seconds() / 3600

        return {
            "time_span_hours": round(time_span, 1),
            "is_burst": time_span <= self.temporal_burst_hours,
            "is_spread": time_span > 72,  # More than 3 days
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Step 4: Apply 10-case decision rules
    # -------------------------------------------------------------------------
    def determine_cluster_status(self, consensus, temporal, cluster_tweets):
        """Apply the 10-case decision matrix to determine cluster status.
        
        Args:
            consensus: dict from compute_cluster_consensus()
            temporal: dict from _compute_temporal_pattern()
            cluster_tweets: list of tweet dicts
            
        Returns:
            dict with status, reason, severity, urgency
        """
        total = consensus["total_tweets"]
        unique_authors = consensus["unique_authors"]
        classified = consensus["classified_count"]
        agreement = consensus["agreement_score"]
        avg_conf = consensus["avg_confidence"]
        humanitarian_pct = consensus["humanitarian_percentage"]
        primary_label = consensus["primary_label_id"]
        label_dist = consensus["label_distribution"]
        is_burst = temporal.get("is_burst", False)
        is_spread = temporal.get("is_spread", False)

        # ── Case 1: Insufficient Data ──
        if total == 1:
            return {
                "status": "insufficient_data",
                "reason": "Only 1 tweet from this location. Cannot verify from a single source.",
                "severity": "UNKNOWN",
                "urgency": "NONE",
            }

        # ── Case 2: Suspicious Single Source ──
        if unique_authors == 1 and total > 1:
            return {
                "status": "suspicious_single_source",
                "reason": f"All {total} tweets are from the same author. No independent corroboration.",
                "severity": "UNKNOWN",
                "urgency": "FLAG",
            }

        # ── Case 3: Early Signal ──
        if total < self.min_cluster_size and unique_authors >= 2:
            return {
                "status": "early_signal",
                "reason": f"Only {total} tweets from {unique_authors} authors. May be emerging event.",
                "severity": "LOW",
                "urgency": "MONITOR",
            }

        # ── Case 10: Location Uncertain ──
        # Check if many authors have conflicting profile locations
        if total >= 5:
            profile_locations = []
            for t in cluster_tweets:
                pl = t.get("userProfileLocation", "")
                if pl:
                    profile_locations.append(self.resolver.normalize(pl))

            cluster_loc = self.resolver.normalize(
                cluster_tweets[0].get("_resolved_location", ""))
            if profile_locations:
                local_count = sum(
                    1 for pl in profile_locations
                    if cluster_loc and cluster_loc in pl or pl in cluster_loc
                )
                remote_pct = 1 - (local_count / len(profile_locations)) if profile_locations else 0
                if remote_pct > 0.6:
                    return {
                        "status": "location_uncertain",
                        "reason": f"{int(remote_pct*100)}% of authors have profiles from other locations. Possible remote reporting.",
                        "severity": "UNCERTAIN",
                        "urgency": "VERIFY",
                    }

        # From here on, we have enough tweets (≥ min_cluster_size)
        if classified == 0:
            return {
                "status": "unclassified",
                "reason": f"{total} tweets found but none classified yet. Run classification first.",
                "severity": "PENDING",
                "urgency": "CLASSIFY",
            }

        # ── Case 6: No Disaster (False Alarm) ──
        not_humanitarian_pct = label_dist.get(2, {}).get("percentage", 0)
        if not_humanitarian_pct >= 80:
            return {
                "status": "no_disaster",
                "reason": f"{not_humanitarian_pct}% of tweets classified as 'Not Humanitarian'. Normal social chatter.",
                "severity": "NONE",
                "urgency": "DISMISS",
            }

        # ── Case 7: Low Model Confidence ──
        if avg_conf < 0.50:
            return {
                "status": "low_confidence_cluster",
                "reason": f"Average model confidence is {avg_conf:.0%}. Tweets may be ambiguous.",
                "severity": "UNCERTAIN",
                "urgency": "HUMAN_REVIEW",
            }

        # ── Case 5: Ambiguous Distribution ──
        if agreement < self.agreement_threshold and humanitarian_pct > 30:
            return {
                "status": "ambiguous_needs_review",
                "reason": f"No class has clear majority ({agreement:.0%} agreement). Mixed signals from this location.",
                "severity": "MEDIUM",
                "urgency": "HUMAN_REVIEW",
            }

        # ── Case 8: Active Event (Temporal Burst) ──
        if is_burst and humanitarian_pct > 70 and total >= 10:
            return {
                "status": "active_event",
                "reason": f"{total} tweets within {temporal['time_span_hours']:.1f} hours. Ongoing event detected!",
                "severity": "CRITICAL",
                "urgency": "IMMEDIATE",
            }

        # ── Case 9: Recovery Phase ──
        if is_spread and classified >= 10:
            # Check if label distribution shifts over time (earlier tweets = 0/1, later = 3/4)
            rescue_labels = {0, 1}
            recovery_labels = {3, 4}
            rescue_pct = sum(label_dist.get(l, {}).get("percentage", 0) for l in rescue_labels)
            recovery_pct = sum(label_dist.get(l, {}).get("percentage", 0) for l in recovery_labels)

            if recovery_pct > rescue_pct:
                return {
                    "status": "recovery_phase",
                    "reason": f"Tweets span {temporal['time_span_hours']:.0f}+ hours. Recovery tweets ({recovery_pct:.0f}%) exceed active disaster ({rescue_pct:.0f}%).",
                    "severity": "MEDIUM",
                    "urgency": "SHIFT_RESPONSE",
                }

        # ── Case 4: Confirmed Event ──
        if total >= self.min_cluster_size and unique_authors >= self.min_unique_authors:
            # Determine severity from humanitarian content
            severity = self._compute_severity(consensus, temporal, cluster_tweets)
            return {
                "status": "confirmed_event",
                "reason": f"{total} tweets from {unique_authors} independent authors. {humanitarian_pct:.0f}% humanitarian content.",
                "severity": severity,
                "urgency": "ALERT",
            }

        # Fallback
        return {
            "status": "under_review",
            "reason": "Cluster does not clearly match any decision case. Manual review recommended.",
            "severity": "LOW",
            "urgency": "MONITOR",
        }

    def _compute_severity(self, consensus, temporal, cluster_tweets):
        """Compute severity level: CRITICAL, HIGH, MEDIUM, LOW."""
        total = consensus["total_tweets"]
        humanitarian_pct = consensus["humanitarian_percentage"]
        label_dist = consensus["label_distribution"]
        is_burst = temporal.get("is_burst", False)

        # Count total people affected from actionable info
        total_people = 0
        for t in cluster_tweets:
            ai = t.get("actionableInfo", {})
            if isinstance(ai, dict):
                for pc in ai.get("peopleCount", ai.get("people_count", [])):
                    if isinstance(pc, dict):
                        total_people += pc.get("count", 0)

        # Severity scoring
        score = 0
        thresholds = GEO_SEVERITY_THRESHOLDS

        if total >= thresholds["tweet_count_high"]:
            score += 3
        elif total >= thresholds["tweet_count_medium"]:
            score += 2
        else:
            score += 1

        if total_people >= thresholds["people_high"]:
            score += 3
        elif total_people >= thresholds["people_medium"]:
            score += 2

        if humanitarian_pct >= 80:
            score += 2
        elif humanitarian_pct >= 50:
            score += 1

        # Affected individuals percentage boosts severity
        affected_pct = label_dist.get(0, {}).get("percentage", 0)
        if affected_pct >= 30:
            score += 2

        if is_burst:
            score += 2

        if score >= 8:
            return "CRITICAL"
        elif score >= 5:
            return "HIGH"
        elif score >= 3:
            return "MEDIUM"
        else:
            return "LOW"

    # -------------------------------------------------------------------------
    # Step 5: Combine actionable intelligence from all tweets
    # -------------------------------------------------------------------------
    def combine_actionable_info(self, cluster_tweets):
        """Merge actionable info from all tweets in a cluster.
        
        Returns:
            dict with combined locations, needs, damage_types, people_counts, time_mentions
        """
        all_locations = set()
        all_needs = set()
        all_damage = set()
        all_times = set()
        total_people = 0
        people_details = []

        for t in cluster_tweets:
            ai = t.get("actionableInfo", {})
            if not isinstance(ai, dict):
                continue

            for loc in ai.get("locations", []):
                all_locations.add(loc)
            for need in ai.get("needs", []):
                all_needs.add(need)
            for dmg in ai.get("damageType", ai.get("damage_type", [])):
                all_damage.add(dmg)
            for tm in ai.get("timeMentions", ai.get("time_mentions", [])):
                all_times.add(tm)
            for pc in ai.get("peopleCount", ai.get("people_count", [])):
                if isinstance(pc, dict):
                    count = pc.get("count", 0)
                    status = pc.get("status", "affected")
                    total_people += count
                    people_details.append(pc)

        return {
            "locations": sorted(all_locations),
            "total_people_affected": total_people,
            "people_details": people_details,
            "needs": sorted(all_needs),
            "damage_types": sorted(all_damage),
            "time_mentions": sorted(all_times),
        }

    # -------------------------------------------------------------------------
    # Step 6: Generate recommended actions
    # -------------------------------------------------------------------------
    def generate_recommended_actions(self, consensus, combined_info, status_result):
        """Generate specific response recommendations based on the 5-class distribution.
        
        Returns:
            list of action strings
        """
        actions = []
        label_dist = consensus["label_distribution"]
        severity = status_result.get("severity", "LOW")
        status = status_result.get("status", "")

        if status in ("no_disaster", "insufficient_data", "suspicious_single_source"):
            return ["No action required at this time."]

        if status == "early_signal":
            return ["Monitor this location for additional tweets.",
                    "Set up automated alerts for this area."]

        if status == "low_confidence_cluster":
            return ["Assign human analyst to review these tweets.",
                    "Request ground reports from local contacts."]

        # Label 0: Affected Individuals → Rescue
        if label_dist.get(0, {}).get("percentage", 0) >= 15:
            locations_str = ", ".join(combined_info["locations"][:5]) if combined_info["locations"] else "affected areas"
            people = combined_info["total_people_affected"]
            action = f"🔴 RESCUE: Deploy search & rescue teams to {locations_str}"
            if people:
                action += f" ({people}+ people affected)"
            actions.append(action)

        # Label 1: Infrastructure Damage → Engineers
        if label_dist.get(1, {}).get("percentage", 0) >= 15:
            damage_str = ", ".join(combined_info["damage_types"][:5]) if combined_info["damage_types"] else "infrastructure"
            actions.append(f"🟠 ENGINEERS: Dispatch repair teams for {damage_str}")

        # Label 4: Rescue/Donation → Coordinate
        if label_dist.get(4, {}).get("percentage", 0) >= 15:
            needs_str = ", ".join(combined_info["needs"][:5]) if combined_info["needs"] else "supplies"
            actions.append(f"🟢 LOGISTICS: Coordinate {needs_str} distribution")

        # Label 3: Other Info → Communicate
        if label_dist.get(3, {}).get("percentage", 0) >= 15:
            actions.append("🔵 COMMS: Issue public advisories and situation updates")

        # Severity-based additions
        if severity in ("CRITICAL", "HIGH"):
            actions.append("⚡ PRIORITY: Escalate to national emergency response coordination")

        if status == "active_event":
            actions.insert(0, "🚨 IMMEDIATE: Real-time disaster in progress — activate all response protocols")

        if status == "recovery_phase":
            actions.append("🔄 TRANSITION: Shift from rescue operations to recovery and rebuilding")

        return actions if actions else ["Continue monitoring this location."]

    # -------------------------------------------------------------------------
    # MAIN: Generate full cluster report
    # -------------------------------------------------------------------------
    def generate_cluster_report(self, location_name, cluster_tweets):
        """Generate a complete cluster assessment report.
        
        Args:
            location_name: normalized location string
            cluster_tweets: list of tweet dicts in this cluster
            
        Returns:
            dict: complete cluster report with all assessment data
        """
        consensus = self.compute_cluster_consensus(cluster_tweets)
        temporal = self._compute_temporal_pattern(cluster_tweets)
        status_result = self.determine_cluster_status(consensus, temporal, cluster_tweets)
        combined_info = self.combine_actionable_info(cluster_tweets)
        actions = self.generate_recommended_actions(consensus, combined_info, status_result)

        # Get the display name (use first tweet's resolved location or the cluster key)
        display_name = location_name
        if cluster_tweets:
            resolved = cluster_tweets[0].get("_resolved_location", "")
            if resolved:
                display_name = resolved

        return {
            "location": display_name,
            "normalized_location": location_name,
            "status": status_result["status"],
            "status_reason": status_result["reason"],
            "severity": status_result["severity"],
            "urgency": status_result["urgency"],

            # Consensus data
            "total_tweets": consensus["total_tweets"],
            "classified_count": consensus["classified_count"],
            "unique_authors": consensus["unique_authors"],
            "primary_label_id": consensus["primary_label_id"],
            "primary_label": consensus["primary_label"],
            "agreement_score": consensus["agreement_score"],
            "avg_confidence": consensus["avg_confidence"],
            "humanitarian_percentage": consensus["humanitarian_percentage"],
            "label_distribution": consensus["label_distribution"],

            # Temporal
            "temporal": temporal,

            # Combined actionable intelligence
            "combined_actionable_info": combined_info,

            # Recommended actions
            "recommended_actions": actions,
        }

    def analyze_all_clusters(self, tweets):
        """Full pipeline: cluster tweets → generate reports for each cluster.
        
        Args:
            tweets: list of all tweet dicts
            
        Returns:
            list of cluster report dicts, sorted by severity
        """
        clusters = self.cluster_tweets(tweets)
        reports = []

        for location_name, cluster_tweets in clusters.items():
            report = self.generate_cluster_report(location_name, cluster_tweets)
            reports.append(report)

        # Sort: CRITICAL > HIGH > MEDIUM > LOW > others
        severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3,
                         "UNCERTAIN": 4, "UNKNOWN": 5, "NONE": 6, "PENDING": 7}
        reports.sort(key=lambda r: severity_order.get(r["severity"], 9))

        return reports
