"""
Script to load test tweets from test_tweets.txt into the database via backend API.
Run this after starting the backend server.
"""
import requests
import time
import re

API_URL = "http://localhost:5000/api"

def load_tweets_from_file(filename="test_tweets_simple.txt"):
    """Load tweets from file, skipping comments and empty lines."""
    tweets = []
    
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            tweets.append(line)
    
    return tweets

def create_tweet(text, author="test_user"):
    """Create a tweet via API."""
    try:
        response = requests.post(
            f"{API_URL}/tweets/create",
            json={
                "text": text,
                "author": author,
                "authorName": author
            },
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return True, data.get("tweet", {}).get("tweetId", "unknown")
            else:
                return False, data.get("error", "Unknown error")
        else:
            return False, f"HTTP {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        return False, "Connection error - make sure backend is running"
    except Exception as e:
        return False, str(e)

def main():
    """Main function to load test tweets."""
    print("Loading test tweets from test_tweets_simple.txt...")
    print("=" * 60)
    
    # Check if backend is running
    try:
        response = requests.get(f"{API_URL.replace('/api', '')}/api/health", timeout=2)
        if response.status_code != 200:
            print("✗ Backend server is not responding properly")
            return
    except:
        print("✗ Cannot connect to backend server")
        print("  Make sure backend is running: cd backend && npm start")
        return
    
    print("✓ Backend server is running")
    print()
    
    # Load tweets
    tweets = load_tweets_from_file()
    print(f"Found {len(tweets)} tweets to load")
    print()
    
    # Load each tweet
    success_count = 0
    error_count = 0
    
    for i, tweet_text in enumerate(tweets, 1):
        print(f"[{i}/{len(tweets)}] Loading tweet...")
        success, result = create_tweet(tweet_text)
        
        if success:
            print(f"  ✓ Success - Tweet ID: {result}")
            success_count += 1
        else:
            print(f"  ✗ Failed - {result}")
            error_count += 1
        
        # Small delay to avoid overwhelming the server
        time.sleep(0.1)
    
    print()
    print("=" * 60)
    print(f"Summary:")
    print(f"  ✓ Successfully loaded: {success_count}")
    print(f"  ✗ Failed: {error_count}")
    print(f"  Total: {len(tweets)}")
    print()
    print("Now you can:")
    print("  1. Run the dashboard: python run_dashboard.py")
    print("  2. Click 'Refresh' to load tweets from database")
    print("  3. Tweets will be automatically classified")

if __name__ == "__main__":
    main()
