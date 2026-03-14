"""Quick script to clear old tweets and reload fresh mock dataset with location fields."""
import requests

BACKEND_URL = "http://localhost:5000/api"

# Step 1: Get all tweet IDs
print("Fetching existing tweets...")
resp = requests.get(f"{BACKEND_URL}/tweets?limit=500")
tweets = resp.json().get("tweets", [])
print(f"Found {len(tweets)} tweets to delete")

# Step 2: Delete each one
for i, t in enumerate(tweets, 1):
    tid = t["tweetId"]
    requests.delete(f"{BACKEND_URL}/tweets/{tid}")
    if i % 50 == 0:
        print(f"  Deleted {i}/{len(tweets)}")

print(f"✓ Deleted all {len(tweets)} tweets")

# Step 3: Verify empty
resp = requests.get(f"{BACKEND_URL}/tweets/stats/summary")
print(f"Remaining tweets: {resp.json()['stats']['totalTweets']}")
