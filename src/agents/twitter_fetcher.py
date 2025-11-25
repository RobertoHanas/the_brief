import os
import tweepy
from rich import print

class TwitterFetcher:
    def __init__(self, timeout=10):
        self.timeout = timeout
        self.api = None
        
        # Try to initialize Twitter API if credentials are available
        bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        api_key = os.getenv("TWITTER_API_KEY")
        api_secret = os.getenv("TWITTER_API_SECRET")
        access_token = os.getenv("TWITTER_ACCESS_TOKEN")
        access_token_secret = os.getenv("TWITTER_ACCESS_TOKEN_SECRET")
        
        if bearer_token:
            try:
                self.api = tweepy.Client(bearer_token=bearer_token)
            except Exception as e:
                print(f"[TwitterFetcher] Failed to initialize with bearer token: {e}")
        elif api_key and api_secret and access_token and access_token_secret:
            try:
                auth = tweepy.OAuthHandler(api_key, api_secret)
                auth.set_access_token(access_token, access_token_secret)
                self.api = tweepy.API(auth)
            except Exception as e:
                print(f"[TwitterFetcher] Failed to initialize with OAuth: {e}")

    def run(self, keywords: list[str]):
        print("[TwitterFetcher] Fetching Twitter…")
        tweets = []
        
        if not self.api:
            print("[TwitterFetcher] No Twitter API credentials → returning empty results")
            return tweets
        
        for keyword in keywords:
            try:
                # Remove # if present for search
                search_term = keyword.replace("#", "")
                
                # Use tweepy to search
                if hasattr(self.api, 'search_recent_tweets'):
                    # v2 API
                    response = self.api.search_recent_tweets(
                        query=search_term,
                        max_results=10,
                        tweet_fields=['text', 'created_at', 'author_id']
                    )
                    
                    if response.data:
                        for tweet in response.data:
                            tweets.append({
                                "source": f"twitter:{keyword}",
                                "title": f"Tweet about {keyword}",
                                "summary": tweet.text[:200],
                                "link": f"https://twitter.com/i/web/status/{tweet.id}",
                                "content": tweet.text
                            })
                else:
                    # v1.1 API
                    results = self.api.search(q=search_term, count=10, lang="en")
                    for tweet in results:
                        tweets.append({
                            "source": f"twitter:{keyword}",
                            "title": f"Tweet about {keyword}",
                            "summary": tweet.text[:200],
                            "link": f"https://twitter.com/i/web/status/{tweet.id}",
                            "content": tweet.text
                        })
            except Exception as e:
                print(f"[TwitterFetcher] Failed on {keyword}: {e}")
        
        print(f"[TwitterFetcher] Fetched {len(tweets)} tweets")
        return tweets

