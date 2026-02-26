"""
Script to load test tweets from test_tweets.txt into the database via backend API.
Aligned with the updated HITL Status Schema.
"""
import requests
import time
import uuid

API_URL = "http://localhost:5000/api"

def load_tweets_from_file(filename="test_tweets_simple.txt"):
    """Load tweets from file, skipping comments and empty lines."""
    tweets = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                tweets.append(line)
    except FileNotFoundError:
        print(f"✗ Error: {filename} not found.")
    return tweets

def create_tweet(text, author="test_user"):
    """
    Create a tweet via API.
    Updated to include authorId and authorName as per new schema.
    """
    try:
        # Construct payload matching your Mongoose Schema
        payload = {
            "tweetId": str(uuid.uuid4()),
            "text": text,
            "author": author,
            "authorName": author,
            "authorId": author, # Required by your backend schema
            "createdAt": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "source": "manual_upload",
            "status": "unverified" # Initial state before dashboard classification
        }

        response = requests.post(
            f"{API_URL}/tweets/create",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("tweet", {}).get("tweetId", "unknown")
            else:
                return False, data.get("error", "Unknown error")
        else:
            return False, f"HTTP {response.status_code}: {response.text}"
            
    except requests.exceptions.ConnectionError:
        return False, "Connection error - make sure backend is running"
    except Exception as e:
        return False, str(e)

def main():
    """Main function to load test tweets."""
    print("Loading test tweets into HITL-enabled Database...")
    print("=" * 60)
    
    # Check if backend is running
    try:
        health_check = f"{API_URL.replace('/api', '')}/api/health"
        response = requests.get(health_check, timeout=2)
        if response.status_code != 200:
            print(f"✗ Backend server at {health_check} is not responding properly")
            return
    except:
        print("✗ Cannot connect to backend server")
        print("  Make sure backend is running: cd backend && npm start")
        return
    
    print("✓ Backend server is running")
    print()
    
    # Load tweets from file
    tweets = load_tweets_from_file()
    if not tweets:
        print("No tweets found to upload.")
        return

    print(f"Found {len(tweets)} tweets to load")
    print()
    
    # Load each tweet
    success_count = 0
    error_count = 0
    
    for i, tweet_text in enumerate(tweets, 1):
        # Truncate text for cleaner console output
        display_text = (tweet_text[:50] + '...') if len(tweet_text) > 50 else tweet_text
        print(f"[{i}/{len(tweets)}] Uploading: {display_text}")
        
        success, result = create_tweet(tweet_text)
        
        if success:
            print(f"  ✓ Success - ID: {result}")
            success_count += 1
        else:
            print(f"  ✗ Failed - {result}")
            error_count += 1
        
        # Small delay to prevent rate-limiting or socket issues
        time.sleep(0.05)
    
    print()
    print("=" * 60)
    print("Upload Summary:")
    print(f"  ✓ Successfully loaded: {success_count}")
    print(f"  ✗ Failed: {error_count}")
    print(f"  Total processed: {len(tweets)}")
    print()
    print("Next Steps:")
    print("  1. Launch your Dashboard (dashboard.py)")
    print("  2. Navigate to 'To Verify' to see these tweets (Status: UNVERIFIED)")
    print("  3. Run manual classification/correction to move them to 'HUMAN_VERIFIED'")

if __name__ == "__main__":
    main()