"""
Twitter/X API Integration Module
Handles authentication and tweet fetching from Twitter API v2.
"""
import tweepy
from Dashboard.config import (
    TWITTER_API_KEY, TWITTER_API_SECRET,
    TWITTER_ACCESS_TOKEN, TWITTER_ACCESS_TOKEN_SECRET,
    TWITTER_BEARER_TOKEN, MAX_TWEETS_PER_FETCH
)


class TwitterAPI:
    """Wrapper for Twitter API v2 operations."""
    
    def __init__(self):
        """Initialize Twitter API client."""
        self.client = None
        self.api = None
        self.authenticated = False
    
    def authenticate(self, api_key=None, api_secret=None, 
                     access_token=None, access_token_secret=None,
                     bearer_token=None):
        """
        Authenticate with Twitter API.
        
        Args:
            api_key: Twitter API Key
            api_secret: Twitter API Secret
            access_token: Access Token
            access_token_secret: Access Token Secret
            bearer_token: Bearer Token (optional, for read-only operations)
        """
        try:
            # Use provided credentials or fall back to config
            api_key = api_key or TWITTER_API_KEY
            api_secret = api_secret or TWITTER_API_SECRET
            access_token = access_token or TWITTER_ACCESS_TOKEN
            access_token_secret = access_token_secret or TWITTER_ACCESS_TOKEN_SECRET
            bearer_token = bearer_token or TWITTER_BEARER_TOKEN
            
            if not all([api_key, api_secret, access_token, access_token_secret]):
                return False, "Missing required API credentials"
            
            # Authenticate with OAuth 1.0a (for user context)
            auth = tweepy.OAuth1UserHandler(
                api_key,
                api_secret,
                access_token,
                access_token_secret
            )
            self.api = tweepy.API(auth, wait_on_rate_limit=True)
            
            # Also create v2 client if bearer token is available
            if bearer_token:
                self.client = tweepy.Client(
                    bearer_token=bearer_token,
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    wait_on_rate_limit=True
                )
            else:
                # Create client without bearer token (uses OAuth 1.0a)
                self.client = tweepy.Client(
                    consumer_key=api_key,
                    consumer_secret=api_secret,
                    access_token=access_token,
                    access_token_secret=access_token_secret,
                    wait_on_rate_limit=True
                )
            
            # Verify credentials
            try:
                user = self.api.verify_credentials()
                self.authenticated = True
                return True, f"Authenticated as @{user.screen_name}"
            except Exception as e:
                return False, f"Authentication failed: {str(e)}"
                
        except Exception as e:
            return False, f"Error during authentication: {str(e)}"
    
    def get_user_timeline(self, username=None, count=MAX_TWEETS_PER_FETCH):
        """
        Get tweets from user's timeline or a specific user's timeline.
        
        Args:
            username: Twitter username (without @). If None, gets authenticated user's timeline.
            count: Number of tweets to fetch (max 200)
            
        Returns:
            list: List of tweet dictionaries
        """
        if not self.authenticated:
            return None, "Not authenticated"
        
        try:
            if username:
                # Get specific user's tweets
                user = self.api.get_user(screen_name=username)
                tweets = self.api.user_timeline(
                    user_id=user.id,
                    count=min(count, 200),
                    tweet_mode='extended',
                    include_rts=False
                )
            else:
                # Get authenticated user's home timeline
                tweets = self.api.home_timeline(
                    count=min(count, 200),
                    tweet_mode='extended',
                    include_rts=False
                )
            
            # Format tweets
            formatted_tweets = []
            for tweet in tweets:
                formatted_tweets.append({
                    "id": tweet.id_str,
                    "text": tweet.full_text,
                    "author": tweet.user.screen_name,
                    "author_name": tweet.user.name,
                    "created_at": tweet.created_at,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count,
                    "url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"
                })
            
            return formatted_tweets, None
            
        except Exception as e:
            return None, f"Error fetching tweets: {str(e)}"
    
    def search_tweets(self, query, count=MAX_TWEETS_PER_FETCH):
        """
        Search for tweets matching a query.
        
        Args:
            query: Search query string
            count: Number of tweets to fetch
            
        Returns:
            list: List of tweet dictionaries
        """
        if not self.authenticated:
            return None, "Not authenticated"
        
        try:
            tweets = self.api.search_tweets(
                q=query,
                count=min(count, 100),  # API v1.1 limit
                tweet_mode='extended',
                lang='en'
            )
            
            formatted_tweets = []
            for tweet in tweets:
                formatted_tweets.append({
                    "id": tweet.id_str,
                    "text": tweet.full_text,
                    "author": tweet.user.screen_name,
                    "author_name": tweet.user.name,
                    "created_at": tweet.created_at,
                    "retweet_count": tweet.retweet_count,
                    "favorite_count": tweet.favorite_count,
                    "url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id_str}"
                })
            
            return formatted_tweets, None
            
        except Exception as e:
            return None, f"Error searching tweets: {str(e)}"
    
    def get_user_info(self):
        """
        Get authenticated user's information.
        
        Returns:
            dict: User information
        """
        if not self.authenticated:
            return None
        
        try:
            user = self.api.verify_credentials()
            return {
                "id": user.id_str,
                "username": user.screen_name,
                "name": user.name,
                "description": user.description,
                "followers_count": user.followers_count,
                "friends_count": user.friends_count,
                "profile_image_url": user.profile_image_url_https
            }
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None

