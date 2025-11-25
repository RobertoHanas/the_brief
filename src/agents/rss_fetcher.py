import feedparser
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from rich import print
from typing import List, Dict, Any


class RSSFetcher:
    def __init__(self, timeout=10):
        self.timeout = timeout

    def discover_feeds(self, url: str) -> List[str]:
        """
        Discover RSS/Atom feeds from a website URL.
        """
        feeds = []
        try:
            response = requests.get(url, timeout=self.timeout, headers={
                'User-Agent': 'Mozilla/5.0 (compatible; FeedDiscovery/1.0)'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for RSS/Atom link tags
            for link in soup.find_all('link', type=['application/rss+xml', 
                                                     'application/atom+xml',
                                                     'application/xml']):
                feed_url = link.get('href')
                if feed_url:
                    feeds.append(urljoin(url, feed_url))
            
            # Common RSS feed locations
            common_paths = ['/feed', '/rss', '/feed.xml', '/rss.xml', '/atom.xml']
            base_url = f"{urlparse(url).scheme}://{urlparse(url).netloc}"
            
            for path in common_paths:
                potential_feed = base_url + path
                if self._is_valid_feed(potential_feed):
                    feeds.append(potential_feed)
                    
        except Exception as e:
            print(f"[RSSFetcher] Feed discovery failed for {url}: {e}")
        
        return list(set(feeds))

    def _is_valid_feed(self, url: str) -> bool:
        """Check if a URL is a valid RSS/Atom feed."""
        try:
            response = requests.head(url, timeout=5, allow_redirects=True)
            content_type = response.headers.get('content-type', '')
            return any(t in content_type for t in ['xml', 'rss', 'atom'])
        except:
            return False

    def search_feeds_by_topic(self, topic: str, max_results: int = 5) -> List[str]:
        """
        Search for RSS feeds related to a topic using common feed directories.
        """
        feeds = []
        
        # Google News RSS (most reliable)
        feeds.append(f"https://news.google.com/rss/search?q={topic.replace(' ', '+')}")
        
        # Common blog/news sites - you can expand this list
        common_sites = [
            f"https://www.reddit.com/r/{topic.replace(' ', '')}.rss",
            f"https://techcrunch.com/tag/{topic.replace(' ', '-')}/feed/",
            f"https://medium.com/feed/tag/{topic.replace(' ', '-')}",
        ]
        
        for site in common_sites[:max_results]:
            if self._is_valid_feed(site):
                feeds.append(site)
        
        return feeds

    def run(self, feeds: List[str], discover_more: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch articles from RSS feeds.
        
        Args:
            feeds: List of feed URLs
            discover_more: If True, attempt to discover additional feeds from provided URLs
        """
        print("[RSSFetcher] Fetching RSS…")
        articles = []
        all_feeds = feeds.copy()

        # Optionally discover more feeds from provided URLs
        if discover_more:
            print("[RSSFetcher] Discovering additional feeds…")
            for url in feeds:
                if not any(ext in url for ext in ['.xml', '.rss', '/feed']):
                    discovered = self.discover_feeds(url)
                    all_feeds.extend(discovered)
                    print(f"[RSSFetcher] Found {len(discovered)} feeds from {url}")

        all_feeds = list(set(all_feeds))  # Remove duplicates
        print(f"[RSSFetcher] Processing {len(all_feeds)} feeds…")

        for url in all_feeds:
            try:
                parsed = feedparser.parse(url)
                
                if parsed.bozo and parsed.bozo_exception:
                    print(f"[RSSFetcher] Warning on {url}: {parsed.bozo_exception}")
                
                for entry in parsed.entries:
                    # Extract content with fallback chain
                    content = ""
                    if hasattr(entry, 'content') and entry.content:
                        content = entry.content[0].get('value', '')
                    elif hasattr(entry, 'summary'):
                        content = entry.summary
                    elif hasattr(entry, 'description'):
                        content = entry.description
                    
                    articles.append({
                        "source": url,
                        "title": entry.get("title", "No title"),
                        "summary": entry.get("summary", "")[:500],  # Limit summary length
                        "link": entry.get("link", ""),
                        "content": content,
                        "published": entry.get("published", ""),
                        "author": entry.get("author", "")
                    })
                    
                print(f"[RSSFetcher] ✓ {url}: {len(parsed.entries)} articles")
                
            except Exception as e:
                print(f"[RSSFetcher] ✗ Failed on {url}: {e}")

        print(f"[RSSFetcher] Total articles collected: {len(articles)}")
        return articles