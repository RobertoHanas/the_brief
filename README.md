# **BRIEF — Behavior-Driven Research Intelligence Enhancement Framework**
### Concierge Capstone – Multi-Agent Personalized Research System

**BRIEF** is a local, one-command, multi-agent concierge that:
- Takes a user-defined topic
- Expands it into subtopics using a Topic Manager
- Discovers relevant external sources (RSS + Twitter/X)
- Fetches articles/posts via dedicated tools
- Filters them through a persona-aware Relevance Agent
- Synthesizes findings into a coherent research brief
- Personalizes style and depth using long-term Memory
- Saves each daily brief locally (and optionally to GitHub)
- Runs fully locally with a clean UI


## **Why I Created BRIEF**
Modern information streams are overwhelming. AI news moves too fast, newsletters are too broad, RSS feeds are too fragmented, and social media is too noisy. Most people don't have the time or cognitive bandwidth to sift through dozens of sources every day — and even when they do, the information they find rarely matches their personal needs.

Existing solutions fall short:

- News aggregators are generic  
- LLM summarizers produce uniform, non-personalized outputs  
- Bookmarking tools don’t solve the relevance problem  
- AI copilots lack memory and don’t adapt over time  
- Research workflows are manual, repetitive, and time-intensive  

I created **BRIEF** to fix this.

BRIEF transforms scattered information into a **daily personalized research brief** that understands:

- who the user is  
- how they think  
- what they care about  
- what level of depth they prefer  
- which topics matter to their role or context  

At its core, BRIEF is a **behavior-driven intelligence pipeline**.  
The system observes user patterns through a MemoryAgent, learns continuously, and then shapes the flow of information through a suite of specialized agents — ensuring that every summary becomes more relevant with each run.

---

## 🚀 **Run with One Command**
```bash
pip install -r requirements.txt
python app.py
```

This launches a Gradio 4.x interface with:
- Topic input  
- Persona selection  
- Tone + length preferences  
- Technicality level  
- Optional API key override  
- Summary output  
- Debug context info  
- File-save path  

---

## 🧠 **Agent Architecture**

### **Core Agents**
- **MemoryAgent**  
  Long-term preference learning (persona, tone, depth, patterns).
  
- **TopicManager**  
  Topic expansion (subtopics + RSS/Twitter keywords).

- **SourceDiscoveryAgent**  
  Maps subtopics to RSS feeds and search terms.

- **RSSFetcher**  
  Retrieves articles from external RSS feeds.

- **TwitterFetcher**  
  Retrieves tweets/threads via search (mock or API-ready).

- **RelevanceAgent**  
  Filters items using persona-aware scoring & context.

- **SynthesizerAgent**  
  Multi-source synthesis into structured text.

- **BriefAgent**  
  Final user-facing daily brief formulation.

- **PublisherAgent**  
  Saves files locally and optionally pushes to a GitHub repo.

---

## 🔧 **Tools**
BRIEF uses a modular tool layer (expandable with MCP or OpenAPI tools):

- **RSS Fetch Tool**  
- **Twitter/X Fetch Tool**  
- **GitHub Publishing Tool**  
- **Local File Writer Tool**  
- **Search Tool** (planned: Google Search / web fetch)  
- **Memory Storage Tool** (local JSON-based store)

Tools allow agents to stay simple, specialized, and interchangeable.

---

## 💾 **Memory System**

User preferences persist across sessions, including:
- persona  
- tone  
- summary length  
- technicality  
- inferred traits  
- behavioral patterns  
- recent topics  
- implicit preferences (LLM-inferred)

Memory is updated with every run → BRIEF becomes more personalized over time.

---

## 📊 **Observability**

Each pipeline run logs:
- agent activity  
- source counts  
- relevance scoring per item  
- synthesis decisions  
- memory updates  
- final summary stats  
- file publishing operations  

This allows debugging and improving agent behavior.



## 📝 **Notes**
- No API keys are included.  
- `.env.example` is provided for environment variable setup.  
- Compatible with Python 3.13 and Gradio 4.x.  
