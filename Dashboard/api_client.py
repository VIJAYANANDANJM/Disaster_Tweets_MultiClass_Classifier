"""
API Client for Backend Server
Handles communication with the Node.js backend API.
"""
import requests
from typing import List, Dict, Optional, Any


def _make_json_serializable(obj: Any) -> Any:
    """Convert numpy/torch float32 and similar types to JSON-serializable Python types."""
    if hasattr(obj, 'item'):  # numpy scalar or torch tensor
        return obj.item()
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, (list, tuple)):
        return [_make_json_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    return obj


def _as_list(value: Any) -> List[Any]:
    """Return value as a list for consistent actionable-info processing."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _normalize_actionable_info(actionable_info: Any) -> Dict[str, Any]:
    """
    Normalize actionable info keys to backend schema keys.
    Accepts snake_case and camelCase keys.
    """
    if not isinstance(actionable_info, dict):
        return {}

    normalized: Dict[str, Any] = {}

    locations = [str(x).strip() for x in _as_list(actionable_info.get("locations")) if str(x).strip()]
    needs = [str(x).strip() for x in _as_list(actionable_info.get("needs")) if str(x).strip()]
    damage_type = [
        str(x).strip()
        for x in _as_list(actionable_info.get("damageType") or actionable_info.get("damage_type"))
        if str(x).strip()
    ]
    time_mentions = [
        str(x).strip()
        for x in _as_list(actionable_info.get("timeMentions") or actionable_info.get("time_mentions"))
        if str(x).strip()
    ]

    people_raw = _as_list(actionable_info.get("peopleCount") or actionable_info.get("people_count"))
    people_count = []
    for item in people_raw:
        if isinstance(item, dict):
            count = item.get("count")
            status = item.get("status")
            entry = {}
            if count is not None:
                entry["count"] = _make_json_serializable(count)
            if status is not None and str(status).strip():
                entry["status"] = str(status).strip()
            if entry:
                people_count.append(entry)
        elif str(item).strip():
            people_count.append({"status": str(item).strip()})

    location_note = actionable_info.get("locationNote") or actionable_info.get("location_note")
    if location_note is not None and str(location_note).strip():
        normalized["locationNote"] = str(location_note).strip()

    if locations:
        normalized["locations"] = locations
    if people_count:
        normalized["peopleCount"] = people_count
    if needs:
        normalized["needs"] = needs
    if damage_type:
        normalized["damageType"] = damage_type
    if time_mentions:
        normalized["timeMentions"] = time_mentions

    return normalized


class APIClient:
    """Client for interacting with the backend API."""
    
    def __init__(self, base_url: str = "http://localhost:5000/api"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the backend API
        """
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> bool:
        """Check if the API server is running."""
        try:
            response = self.session.get(f"{self.base_url.replace('/api', '')}/api/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def create_tweet(self, tweet_data: Dict) -> tuple[bool, str, Optional[Dict]]:
        """
        Create a new tweet in the database.
        Only sends basic fields; classification is sent separately via update_tweet_classification.
        """
        try:
            # Send only fields the backend expects (avoid float32 in classification/explanation)
            payload = {
                "text": tweet_data.get("text", ""),
                "author": tweet_data.get("author", "manual_input"),
                "authorName": tweet_data.get("authorName", tweet_data.get("author", "manual_input")),
                "tweetId": tweet_data.get("tweetId"),
                "createdAt": tweet_data.get("createdAt"),
                "retweetCount": tweet_data.get("retweetCount", 0),
                "favoriteCount": tweet_data.get("favoriteCount", 0),
            }
            payload = _make_json_serializable(payload)

            response = self.session.post(
                f"{self.base_url}/tweets/create",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return True, "Tweet created successfully", data.get("tweet")
                else:
                    return False, data.get("error", "Failed to create tweet"), None
            else:
                error_data = response.json() if response.content else {}
                return False, error_data.get("error", "Failed to create tweet"), None
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}", None
    
    def get_tweets(self, page: int = 1, limit: int = 50, 
                   label_id: Optional[int] = None,
                   author: Optional[str] = None) -> tuple[bool, str, List[Dict], Dict]:
        """
        Get tweets from database.
        
        Returns:
            tuple: (success, message, tweets, pagination_info)
        """
        try:
            params = {
                "page": page,
                "limit": limit
            }
            if label_id is not None:
                params["labelId"] = label_id
            if author:
                params["author"] = author
            
            response = self.session.get(
                f"{self.base_url}/tweets",
                params=params,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return True, "Success", data.get("tweets", []), data.get("pagination", {})
                else:
                    return False, data.get("error", "Failed to get tweets"), [], {}
            else:
                error_data = response.json() if response.content else {}
                return False, error_data.get("error", "Failed to get tweets"), [], {}
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}", [], {}
    
    def get_tweet(self, tweet_id: str) -> tuple[bool, str, Optional[Dict]]:
        """Get a single tweet by ID."""
        try:
            response = self.session.get(f"{self.base_url}/tweets/{tweet_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return True, "Success", data.get("tweet")
                else:
                    return False, data.get("error", "Tweet not found"), None
            else:
                error_data = response.json() if response.content else {}
                return False, error_data.get("error", "Tweet not found"), None
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}", None
    
    def update_tweet_classification(self, tweet_id: str, classification: Dict,
                                   explanation: List, actionable_info: Dict,
                                   status: Optional[str] = None) -> tuple[bool, str]:
        """
        Update tweet with classification results from local model.
        Converts float32/numpy types to native Python for JSON serialization.
        
        Returns:
            tuple: (success, message)
        """
        try:
            # Convert explanation: list of (token, score) or list of dicts
            explanation_serializable = []
            for item in (explanation or []):
                if isinstance(item, (list, tuple)):
                    explanation_serializable.append({
                        "token": item[0],
                        "score": _make_json_serializable(item[1])
                    })
                elif isinstance(item, dict):
                    explanation_serializable.append({
                        "token": item.get("token", ""),
                        "score": _make_json_serializable(item.get("score", 0))
                    })
                else:
                    explanation_serializable.append(item)

            payload = {
                "classification": _make_json_serializable(classification or {}),
                "explanation": explanation_serializable,
                "actionableInfo": _make_json_serializable(_normalize_actionable_info(actionable_info or {}))
            }
            if status:
                payload["status"] = status

            response = self.session.put(
                f"{self.base_url}/tweets/{tweet_id}/classify",
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return True, "Classification saved"
                else:
                    return False, data.get("error", "Failed to save classification")
            else:
                error_data = response.json() if response.content else {}
                return False, error_data.get("error", "Failed to save classification")
                
        except requests.exceptions.RequestException as e:
            return False, f"Connection error: {str(e)}"

