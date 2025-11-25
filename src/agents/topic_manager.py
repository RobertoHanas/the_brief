import json
import re
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List


class TopicExpansion(BaseModel):
    """Pydantic model for topic expansion output"""
    original: str
    subtopics: List[str]
    rss_keywords: List[str]
    twitter_keywords: List[str]


class TopicManager:
    def __init__(self, client: OpenAI, ctx):
        self.client = client
        self.ctx = ctx

    def extract_json(self, text: str) -> str:
        """
        Extracts the first valid JSON object from ANY text.
        """
        # Remove markdown code fence markers but keep the content
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip()

        # Find first { ... } block including nested braces
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
                    end = i + 1
                    return text[start:end]

        raise ValueError("Could not extract a JSON object from model output.")

    def run(self, topic: str) -> TopicExpansion:
        print("[TopicManager] Expanding topic…")

        prompt = f"""
        Expand the topic '{topic}' for persona '{self.ctx.persona}'.

        Return ONLY a JSON object with fields:
        original, subtopics, rss_keywords, twitter_keywords.
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        

        raw = response.choices[0].message.content.strip()
        print("============ RAW MODEL OUTPUT ============")
        print(repr(raw))
        print("==========================================")     
        
        # Extract JSON cleanly
        try:
            json_str = self.extract_json(raw)
        except Exception as e:
            print("[TopicManager] RAW MODEL OUTPUT:")
            print(raw)
            raise e

        # Validate with Pydantic
        try:
            return TopicExpansion.model_validate_json(json_str)
        except ValidationError as e:
            print("[TopicManager] JSON EXTRACTED BUT INVALID:")
            print(json_str)
            raise e