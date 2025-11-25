# Personal Research Butler (PRB)
### Concierge Capstone – Multi-Agent Personalized Research System

PRB is a local, one-command, agent-based concierge that:
- Accepts a user topic
- Automatically discovers relevant sources (RSS + Twitter)
- Filters them using a Relevance Agent
- Generates a personalized research brief
- Stores preferences via a Memory Agent
- Runs locally with a clean UI

### 🚀 One Command to Run
pip install -r requirements.txt
python app.py


### 🧠 Agents
- Personalization Agent (long-term memory)
- Topic Manager
- Source Discovery Agent
- RSS Fetcher
- Twitter Fetcher
- Relevance Agent
- Knowledge Synthesis Agent
- Research Brief Agent
- Publisher Agent

### 🔧 Tools
- Custom RSS tool
- Custom Twitter tool
- GitHub push tool
- File writer tool
- Search tool
- Memory storage tool

### 💾 Memory
User preferences persist across sessions:
- persona
- tone
- summary length
- technicality
- source bias
- recent topics

### 📊 Observability
Each step logs:
- agent activity
- relevance scoring
- metrics (#sources, #items filtered)
- trace information

### 📁 Project Structure
(see repository tree above)

### 📽️ Video Demo
(Insert YouTube link if available)

### 📄 Kaggle Notebook
Located in `/notebooks/CapstoneNotebook.ipynb`

### © Notes
No API keys included. `.env.example` provided.
