import json
import os
from datetime import datetime
from rich import print
from openai import OpenAI


class PersonalizationContext:
    """
    Lightweight data carrier passed into all agents.
    Contains ALL memory-derived personalization information.
    """

    def __init__(self, persona, preferences, implicit_traits, patterns, recent_topics):
        self.persona = persona
        self.preferences = preferences
        self.implicit_traits = implicit_traits
        self.patterns = patterns
        self.recent_topics = recent_topics


class MemoryAgent:
    """
    Unified memory + personalization system.

    Responsibilities:
        - Load or create memory.json
        - Merge user-set UI preferences into memory
        - Use LLM to infer patterns and implicit traits
        - Persist memory updates
        - Output a PersonalizationContext for all other agents
    """

    def __init__(self, user_id: str, openai_client: OpenAI, base_path="src/memory"):
        self.user_id = user_id
        self.openai = openai_client
        self.base_path = base_path
        self.path = os.path.join(base_path, f"{user_id}.json")

        os.makedirs(base_path, exist_ok=True)
        self.memory = self._load_or_default()

    # -------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------

    def _default(self):
        return {
            "persona": "default",
            "preferences": {
                "tone": "neutral",
                "length": "medium",
                "technicality": "medium",
                "style": "balanced"
            },
            "implicit_traits": {},
            "patterns": {},
            "topic_history": [],
            "last_updated": str(datetime.now())
        }

    def _load_or_default(self):
        if not os.path.exists(self.path):
            print(f"[Memory] Creating new memory for {self.user_id}")
            return self._default()

        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            print(f"[Memory] Loaded memory for {self.user_id}")
            return data
        except:
            print("[Memory] Failed to load memory, resetting.")
            return self._default()

    # -------------------------------------------------------
    # Main Logic
    # -------------------------------------------------------

    def update(self, ui_preferences: dict, topic: str):
        """
        Updates memory.json using both:
         - Explicit UI preferences
         - Implicit inference via LLM
        """

        # Prepare LLM update prompt
        prompt = f"""
        You are the memory system for a personalized research agent.
        Your job is to update the user's long-term memory.

        CURRENT MEMORY:
        {json.dumps(self.memory, indent=2)}

        NEW USER INPUT FROM UI:
        {json.dumps(ui_preferences, indent=2)}

        NEW TOPIC SELECTED:
        "{topic}"

        Update rules:
          - Merge UI preferences into preferences.
          - Append topic to topic_history (no duplicates, keep latest 50).
          - Infer implicit traits from patterns in UI inputs + topic history.
          - Infer patterns (domains, technicality tendency, length trends).
          - Maintain consistent JSON schema.
          - Do not hallucinate random fields.

        Output ONLY valid JSON.
        """

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )

        updated_content = response.choices[0].message.content.strip()

        try:
            updated_json = json.loads(updated_content)
            self.memory = updated_json
            self.memory["last_updated"] = str(datetime.now())
            self._save()
            print("[Memory] Memory updated by LLM.")
        except Exception as e:
            print(f"[Memory] Could not parse LLM output: {e}")
            print("[Memory] Keeping previous memory state.")

    # -------------------------------------------------------
    # Save + Context Access
    # -------------------------------------------------------

    def _save(self):
        with open(self.path, "w") as f:
            json.dump(self.memory, f, indent=2)
        print(f"[Memory] Saved memory.json for {self.user_id}")

    def get_context(self) -> PersonalizationContext:
        """Returns a standardized personalization object consumed by all agents."""
        return PersonalizationContext(
            persona=self.memory["persona"],
            preferences=self.memory["preferences"],
            implicit_traits=self.memory.get("implicit_traits", {}),
            patterns=self.memory.get("patterns", {}),
            recent_topics=self.memory.get("topic_history", [])[-5:]
        )
