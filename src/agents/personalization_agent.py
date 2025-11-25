import json
import os
from rich import print

class UserProfileContext:
    """Shared context passed to all agents after memory is loaded."""
    def __init__(self, persona, preferences, recent_topics):
        self.persona = persona
        self.preferences = preferences
        self.recent_topics = recent_topics

class PersonalizationAgent:
    """
    Advanced memory agent that:
        - stores persona + preferences
        - learns patterns over time
        - updates memory after each digest
        - provides UserProfileContext to all other agents
    """

    def __init__(self, user_id, base_path="src/memory"):
        self.user_id = user_id
        self.memory_path = os.path.join(base_path, f"{user_id}.json")
        os.makedirs(base_path, exist_ok=True)
        self.memory = self._load_or_default()

    def _default(self):
        return {
            "persona": "default",
            "preferences": {
                "tone": "neutral",
                "length": "medium",
                "technicality": "medium",
                "threshold": 0.4,
                "summary_style": "insightful"
            },
            "recent_topics": [],
            "usage_metrics": {
                "total_runs": 0,
                "persona_usage": {},
                "preferred_topic_clusters": {}
            }
        }

    def _load_or_default(self):
        if not os.path.exists(self.memory_path):
            print(f"[Memory] Creating new memory for {self.user_id}")
            return self._default()

        try:
            with open(self.memory_path, "r") as f:
                data = json.load(f)
            print(f"[Memory] Loaded memory for {self.user_id}")
            return data
        except Exception as e:
            print(f"[Memory] ERROR loading, resetting memory → {e}")
            return self._default()

    # --------------------------------------------------------
    # PUBLIC METHODS TO GET CONTEXT
    # --------------------------------------------------------
    def get_context(self) -> UserProfileContext:
        return UserProfileContext(
            persona=self.memory["persona"],
            preferences=self.memory["preferences"],
            recent_topics=self.memory["recent_topics"]
        )

    # --------------------------------------------------------
    # MEMORY UPDATES
    # --------------------------------------------------------
    def record_topic(self, topic):
        self.memory["recent_topics"].append(topic)
        self.memory["recent_topics"] = self.memory["recent_topics"][-50:]
        self._save()

    def update_preferences(self, **kwargs):
        # Update selective preferences
        for key, value in kwargs.items():
            self.memory["preferences"][key] = value
        self._save()

    def learn_from_usage(self):
        """Optional ML-like memory learning."""
        self.memory["usage_metrics"]["total_runs"] += 1

        # Learn if user tends to prefer short summaries
        if len(self.memory["recent_topics"]) >= 5:
            # Example heuristic: if user clears UI quickly, prefers short
            pass

        self._save()

    def set_persona(self, persona):
        self.memory["persona"] = persona
        self._save()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------
    def _save(self):
        with open(self.memory_path, "w") as f:
            json.dump(self.memory, f, indent=2)
        print(f"[Memory] Memory saved for {self.user_id}")
