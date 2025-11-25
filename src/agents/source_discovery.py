import json
import re
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List
from rich import print


class DiscoveredSources(BaseModel):
    """Pydantic model for discovered sources"""
    rss_feeds: List[str]
    websites: List[str]
    twitter_accounts: List[str]
    subreddits: List[str]


class SourceDiscoveryAgent:
    def __init__(self, client: OpenAI = None, base_rss: List[str] = None, 
                 base_twitter: List[str] = None):
        self.client = client
        # Handle both list and None/empty cases
        self.base_rss = base_rss if isinstance(base_rss, list) else []
        self.base_twitter = base_twitter if isinstance(base_twitter, list) else []

    def extract_json(self, text: str) -> str:
        """Extract JSON from LLM output."""
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip()
        
        stack = []
        start = None
        
        for i, char in enumerate(text):
            if char == '{':
                if start is None:
                    start = i
                stack.append('{')
            elif char == '}' and stack:
                stack.pop()
                if not stack:
                    return text[start:i+1]
        
        raise ValueError("Could not extract JSON from model output.")

    def discover_sources_with_llm(self, topic_info, ctx) -> DiscoveredSources:
        """
        Use LLM to discover high-quality sources for the given topic.
        """
        if not self.client:
            print("[SourceDiscovery] No OpenAI client, skipping LLM discovery")
            return DiscoveredSources(
                rss_feeds=[],
                websites=[],
                twitter_accounts=[],
                subreddits=[]
            )

        prompt = f"""
        You are a research assistant helping discover high-quality information sources.
        
        Topic: {topic_info.original}
        Subtopics: {', '.join(topic_info.subtopics[:5])}
        Persona: {ctx.persona}
        
        Find the BEST and most AUTHORITATIVE sources for this topic. Focus on:
        - Major news outlets and industry publications with RSS feeds
        - Expert blogs and specialized websites
        - Active Twitter accounts of experts, organizations, and news sources
        - Relevant subreddits with active communities
        
        Return ONLY a JSON object with these fields:
        {{
            "rss_feeds": ["url1", "url2"],  // Direct RSS feed URLs
            "websites": ["domain1.com", "domain2.com"],  // Websites to check for RSS
            "twitter_accounts": ["@expert1", "@org2"],  // Twitter handles
            "subreddits": ["r/topic1", "r/topic2"]  // Reddit communities
        }}
        
        Provide 3-5 high-quality sources for each category.
        """

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            raw = response.choices[0].message.content.strip()
            json_str = self.extract_json(raw)
            
            return DiscoveredSources.model_validate_json(json_str)
            
        except Exception as e:
            print(f"[SourceDiscovery] LLM discovery failed: {e}")
            return DiscoveredSources(
                rss_feeds=[],
                websites=[],
                twitter_accounts=[],
                subreddits=[]
            )

    def run(self, topic_info, ctx):
        print("[SourceDiscovery] Discovering sources…")
        
        # Auto-generate Google News RSS feeds
        auto_rss = [
            f"https://news.google.com/rss/search?q={kw.replace(' ', '+')}"
            for kw in topic_info.rss_keywords
        ]

        # Auto-generate Twitter hashtags
        auto_twitter = [
            f"#{kw}" if not kw.startswith("#") else kw
            for kw in topic_info.twitter_keywords
        ]

        # Use LLM to discover additional high-quality sources
        discovered = self.discover_sources_with_llm(topic_info, ctx)
        
        # Combine all sources
        all_rss = list(set(
            self.base_rss + 
            auto_rss + 
            discovered.rss_feeds
        ))
        
        all_twitter = list(set(
            self.base_twitter + 
            auto_twitter + 
            discovered.twitter_accounts
        ))
        
        # Add subreddit RSS feeds
        for subreddit in discovered.subreddits:
            clean_name = subreddit.replace('r/', '')
            all_rss.append(f"https://www.reddit.com/r/{clean_name}/.rss")
        
        result = {
            "rss": all_rss,
            "twitter": all_twitter,
            "websites": discovered.websites,  # For potential feed discovery
            "subreddits": discovered.subreddits
        }
        
        print(f"[SourceDiscovery] Found {len(result['rss'])} RSS feeds")
        print(f"[SourceDiscovery] Found {len(result['twitter'])} Twitter sources")
        print(f"[SourceDiscovery] Found {len(result['websites'])} websites to explore")
        
        return result