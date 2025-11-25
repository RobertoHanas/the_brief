import os
from dotenv import load_dotenv
from rich import print
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Memory + context
from src.agents.memory_agent import MemoryAgent

# Agents
from src.agents.topic_manager import TopicManager
from src.agents.source_discovery import SourceDiscoveryAgent
from src.agents.rss_fetcher import RSSFetcher
from src.agents.twitter_fetcher import TwitterFetcher
from src.agents.relevance_agent import RelevanceAgent
from src.agents.synthesizer import SynthesizerAgent
from src.agents.brief_agent import BriefAgent
from src.agents.publisher import PublisherAgent

# Cost tracking
from src.agents.cost_tracker import CostTracker


# ---------------------------------------------------------
# USER ID RESOLUTION (GitHub username or fallback)
# ---------------------------------------------------------

def resolve_user_id_from_github():
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        return "local_user"

    import requests
    try:
        resp = requests.get("https://api.github.com/user", headers={
            "Authorization": f"token {token}"
        })

        if resp.status_code == 200:
            return resp.json().get("login", "local_user")

        return "local_user"
    except:
        return "local_user"


# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------

def run_pipeline(topic: str, ui_preferences: dict, api_key: str = None):
    """
    End-to-end run cycle called by Gradio UI.

    Steps:
    1) Determine user_id
    2) Initialize MemoryAgent
    3) Update memory using UI + topic
    4) Get PersonalizationContext
    5) Run all research/summarization agents with enhanced discovery
    6) Save + optionally publish
    7) Return final brief and metadata
    
    Args:
        topic: The research topic
        ui_preferences: User preferences from UI
        api_key: OpenAI API key (from UI or env var)
    """

    print("\n[Orchestrator] Starting run…")

    # Initialize cost tracker
    cost_tracker = CostTracker()
    cost_tracker.log_agent_activity("🚀 Pipeline started")

    # ----------- SETUP -----------
    user_id = resolve_user_id_from_github()
    github_repo = os.getenv("GITHUB_REPO")
    github_token = os.getenv("GITHUB_TOKEN")

    # Get API key from parameter or environment variable
    openai_key = api_key if api_key and api_key.strip() else os.getenv("OPENAI_API_KEY")
    if not openai_key or not openai_key.strip():
        raise ValueError("OpenAI API key is required. Please provide it in the UI or set OPENAI_API_KEY environment variable.")
    
    # Clean the API key (remove whitespace)
    openai_key = openai_key.strip()
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_key)

    # Initialize memory
    mem = MemoryAgent(user_id=user_id, openai_client=client)

    # Update memory with new data
    mem.update(ui_preferences=ui_preferences, topic=topic)

    # Get personalized context
    ctx = mem.get_context()
    
    # Get max_items limit from preferences (default to unlimited)
    max_items = ui_preferences.get("max_items", 1000)
    if max_items >= 1000:
        max_items = None  # Unlimited
        print(f"[Orchestrator] 📊 Fetching unlimited items")
    else:
        print(f"[Orchestrator] 📊 Limiting to {max_items} items")

    # Initialize research agents with context and OpenAI client
    tm = TopicManager(client=client, ctx=ctx)
    
    # Enhanced source discovery with LLM capabilities
    sd = SourceDiscoveryAgent(
        client=client,  # Pass client for LLM-powered discovery
        base_rss=ui_preferences.get("custom_rss_feeds", []),
        base_twitter=ui_preferences.get("custom_twitter_accounts", [])
    )
    
    # Enhanced RSS fetcher with discovery capabilities
    rss_fetcher = RSSFetcher(timeout=15)
    
    twitter_fetcher = TwitterFetcher()
    relevance = RelevanceAgent(client=client, ctx=ctx)
    synthesizer = SynthesizerAgent(client=client, ctx=ctx)
    brief_agent = BriefAgent(client=client, ctx=ctx)
    publisher = PublisherAgent(
        github_repo=github_repo,
        github_token=github_token
    )

    # ----------- EXECUTION -----------
    print("[Orchestrator] 🎯 Expanding topic…")
    topic_info = tm.run(topic)

    print("[Orchestrator] 🔍 Discovering sources with AI…")
    sources = sd.run(topic_info, ctx)
    
    print(f"[Orchestrator] Found {len(sources['rss'])} RSS feeds")
    print(f"[Orchestrator] Found {len(sources['twitter'])} Twitter sources")
    print(f"[Orchestrator] Found {len(sources.get('websites', []))} websites to explore")

    # ----------- ENHANCED RSS FETCHING -----------
    print("[Orchestrator] 📰 Fetching RSS feeds…")
    rss_items = rss_fetcher.run(sources["rss"])
    
    # Apply limit if set
    if max_items and len(rss_items) >= max_items:
        print(f"[Orchestrator] ⚠️  Reached RSS item limit ({max_items}), skipping additional discovery")
        rss_items = rss_items[:max_items]
    else:
        # Discover additional feeds from websites if any were found
        if sources.get("websites"):
            print("[Orchestrator] 🌐 Discovering additional feeds from websites…")
            for website in sources["websites"][:3]:  # Limit to top 3 to avoid slowdown
                # Check if we've hit the limit
                if max_items and len(rss_items) >= max_items:
                    print(f"[Orchestrator] ⚠️  Reached item limit ({max_items})")
                    break
                    
                try:
                    discovered_feeds = rss_fetcher.discover_feeds(f"https://{website}")
                    if discovered_feeds:
                        print(f"[Orchestrator] ✓ Found {len(discovered_feeds)} feeds from {website}")
                        more_articles = rss_fetcher.run(discovered_feeds)
                        rss_items.extend(more_articles)
                        
                        # Trim if over limit
                        if max_items and len(rss_items) > max_items:
                            rss_items = rss_items[:max_items]
                            break
                except Exception as e:
                    print(f"[Orchestrator] ✗ Could not discover feeds from {website}: {e}")
        
        # Search for topic-specific feeds
        if not max_items or len(rss_items) < max_items:
            print("[Orchestrator] 🔎 Searching for topic-specific feeds…")
            try:
                topic_feeds = rss_fetcher.search_feeds_by_topic(topic_info.original, max_results=3)
                if topic_feeds:
                    topic_articles = rss_fetcher.run(topic_feeds)
                    rss_items.extend(topic_articles)
                    
                    # Trim if over limit
                    if max_items and len(rss_items) > max_items:
                        rss_items = rss_items[:max_items]
                    
                    print(f"[Orchestrator] ✓ Found {len(topic_articles)} articles from topic search")
            except Exception as e:
                print(f"[Orchestrator] ✗ Topic search failed: {e}")

    print(f"[Orchestrator] 📊 Total RSS articles collected: {len(rss_items)}")

    # ----------- TWITTER FETCHING -----------
    print("[Orchestrator] 🐦 Fetching Twitter…")
    
    # Calculate remaining budget for Twitter
    if max_items:
        remaining_budget = max_items - len(rss_items)
        if remaining_budget <= 0:
            print(f"[Orchestrator] ⚠️  Item limit reached, skipping Twitter")
            tweets = []
        else:
            print(f"[Orchestrator] Twitter budget: {remaining_budget} items")
            tweets = twitter_fetcher.run(sources["twitter"])
            # Trim to budget
            tweets = tweets[:remaining_budget]
    else:
        tweets = twitter_fetcher.run(sources["twitter"])
    
    print(f"[Orchestrator] 📊 Total tweets collected: {len(tweets)}")

    # ----------- CONTENT PROCESSING -----------
    # Combine all content
    content_pool = rss_items + tweets
    print(f"[Orchestrator] 📚 Total content items: {len(content_pool)}")

    print("[Orchestrator] 🎯 Filtering for relevance…")
    relevant = relevance.run(
        topic=topic,
        chunks=content_pool,
        method="embedding",  # "embedding" (fast) or "llm" (slow but more nuanced)
        threshold=0.5,  # Adjust based on needs (0.4-0.6 typical)
        save_results=True  # Save scoring details to files
    )
    print(f"[Orchestrator] ✓ {len(relevant)}/{len(content_pool)} items deemed relevant")

    print("[Orchestrator] 🧠 Synthesizing insights…")
    synthesis = synthesizer.run(topic, relevant)

    print("[Orchestrator] ✍️ Creating personalized brief…")
    final_brief = brief_agent.run(topic, synthesis)

    print("[Orchestrator] 💾 Publishing…")
    saved_path = publisher.run(
        topic=topic,
        brief=final_brief,
        persona=ctx.persona,
        metadata={
            "total_items_collected": len(content_pool),
            "relevant_items_count": len(relevant),
            "sources_discovered": {
                "rss_feeds": len(sources["rss"]),
                "twitter_accounts": len(sources["twitter"]),
                "websites_explored": len(sources.get("websites", [])),
                "subreddits": len(sources.get("subreddits", []))
            },
            "topic_expansion": {
                "original": topic_info.original,
                "subtopics": topic_info.subtopics,
                "keywords": topic_info.rss_keywords
            }
        },
        formats=['md', 'json', 'html']  # Save in all formats
    )

    # ----------- OUTPUT -----------
    print("\n[Orchestrator] ✅ COMPLETE.")
    print(f"[Orchestrator] Brief saved to: {saved_path}")
    
    cost_tracker.log_agent_activity("✅ Pipeline complete")
    
    # Print cost summary
    stats = cost_tracker.get_stats()
    print(f"\n💰 Total Cost: ${stats['total_cost']:.4f}")
    print(f"📊 Total Tokens: {stats['total_tokens']:,}")
    print(f"⏱️  Duration: {stats['duration']:.1f}s")
    
    return {
        "brief": final_brief,
        "context": ctx,
        "saved_path": saved_path,
        "relevant_items_count": len(relevant),
        "total_items_collected": len(content_pool),
        "sources_discovered": {
            "rss_feeds": len(sources["rss"]),
            "twitter_accounts": len(sources["twitter"]),
            "websites_explored": len(sources.get("websites", [])),
            "subreddits": len(sources.get("subreddits", []))
        },
        "topic_expansion": {
            "original": topic_info.original,
            "subtopics": topic_info.subtopics,
            "keywords": topic_info.rss_keywords
        },
        "cost_tracker": cost_tracker  # Add cost tracker to results
    }