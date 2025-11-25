from openai import OpenAI
from rich import print
from typing import List, Dict, Any
from collections import defaultdict
import json


class SynthesizerAgent:
    def __init__(self, client: OpenAI, ctx):
        self.client = client
        self.ctx = ctx

    def _cluster_by_theme(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Use LLM to identify themes and cluster content.
        """
        print("[Synthesizer] 🔍 Identifying themes…")
        
        # Get titles/summaries for theme identification
        items_preview = []
        for i, chunk in enumerate(chunks[:50]):  # Limit to first 50 for theme detection
            title = chunk.get("title", "")
            summary = chunk.get("summary", "")[:200]
            items_preview.append(f"{i+1}. {title}: {summary}")
        
        preview_text = "\n".join(items_preview)
        
        prompt = f"""Analyze these content items and identify 4-6 major themes/topics.
        
Items:
{preview_text}

Return ONLY a JSON array of theme names, like:
["Theme 1", "Theme 2", "Theme 3"]

Keep themes specific and actionable."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for cost savings
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                themes = json.loads(json_match.group())
                print(f"[Synthesizer] ✓ Identified {len(themes)} themes: {', '.join(themes[:3])}...")
                return themes
        except Exception as e:
            print(f"[Synthesizer] ⚠️  Theme detection failed: {e}")
        
        # Fallback: generic themes
        return ["Key Findings", "Recent Developments", "Expert Insights", "Practical Applications"]

    def _assign_to_themes(self, chunks: List[Dict[str, Any]], themes: List[str]) -> Dict[str, List[Dict]]:
        """
        Assign each chunk to the most relevant theme.
        """
        print("[Synthesizer] 📊 Organizing content by themes…")
        
        clustered = defaultdict(list)
        
        # Simple keyword-based assignment for speed
        # In production, could use embeddings for better accuracy
        for chunk in chunks:
            content = (chunk.get("title", "") + " " + chunk.get("content", "")[:500]).lower()
            
            # Assign to first matching theme or default to first theme
            assigned = False
            for theme in themes:
                theme_keywords = theme.lower().split()
                if any(keyword in content for keyword in theme_keywords):
                    clustered[theme].append(chunk)
                    assigned = True
                    break
            
            if not assigned:
                clustered[themes[0]].append(chunk)
        
        # Print distribution
        for theme in themes:
            count = len(clustered.get(theme, []))
            print(f"[Synthesizer]   • {theme}: {count} items")
        
        return dict(clustered)

    def _extract_key_facts(self, chunks: List[Dict[str, Any]], max_facts: int = 15) -> List[str]:
        """
        Extract key facts, statistics, and quotes from content.
        """
        print("[Synthesizer] 📝 Extracting key facts…")
        
        # Combine content from top chunks
        combined_content = []
        for chunk in chunks[:20]:  # Top 20 chunks
            content = chunk.get("content", "") or chunk.get("summary", "")
            if content:
                combined_content.append(content[:1000])
        
        combined = "\n\n".join(combined_content)
        
        prompt = f"""Extract the {max_facts} most important facts, statistics, quotes, or insights from this content.

Focus on:
- Specific numbers, percentages, or data points
- Notable quotes from experts or sources
- Surprising or counter-intuitive findings
- Actionable insights
- Recent developments or trends

Content:
{combined[:8000]}

Return as a JSON array of strings. Each fact should be concise (1-2 sentences) and specific.
Format: ["fact 1", "fact 2", ...]"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            # Extract JSON array
            import re
            json_match = re.search(r'\[.*?\]', content, re.DOTALL)
            if json_match:
                facts = json.loads(json_match.group())
                print(f"[Synthesizer] ✓ Extracted {len(facts)} key facts")
                return facts[:max_facts]
        except Exception as e:
            print(f"[Synthesizer] ⚠️  Fact extraction failed: {e}")
        
        return []

    def run(self, topic: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Synthesize relevant content into a structured analysis.
        """
        print(f"[Synthesizer] 🧠 Synthesizing {len(relevant_chunks)} items…")
        
        if not relevant_chunks:
            return "No relevant content found for synthesis."
        
        # Get preferences safely
        if hasattr(self.ctx, 'preferences') and self.ctx.preferences:
            prefs = self.ctx.preferences
            if isinstance(prefs, dict):
                length = prefs.get('length', 'medium')
                tone = prefs.get('tone', 'neutral')
                technicality = prefs.get('technicality', 'medium')
            else:
                length = tone = technicality = 'medium'
        else:
            length = tone = technicality = 'medium'
        
        # Step 1: Identify themes
        themes = self._cluster_by_theme(relevant_chunks)
        
        # Step 2: Cluster content by theme
        clustered = self._assign_to_themes(relevant_chunks, themes)
        
        # Step 3: Extract key facts
        key_facts = self._extract_key_facts(relevant_chunks)
        
        # Step 4: Build synthesis prompt with structured data
        print("[Synthesizer] ✍️  Generating synthesis…")
        
        # Prepare theme summaries
        theme_summaries = []
        for theme in themes:
            theme_chunks = clustered.get(theme, [])[:5]  # Top 5 per theme
            if theme_chunks:
                theme_content = "\n".join([
                    f"- {c.get('title', 'Untitled')}: {(c.get('content', '') or c.get('summary', ''))[:300]}"
                    for c in theme_chunks
                ])
                theme_summaries.append(f"**{theme}**:\n{theme_content}")
        
        themes_text = "\n\n".join(theme_summaries)
        facts_text = "\n".join([f"- {fact}" for fact in key_facts])
        
        # Get source diversity
        sources = set()
        for chunk in relevant_chunks[:30]:
            source = chunk.get("source", "")
            if source:
                sources.add(source)
        
        synthesis_prompt = f"""You are synthesizing research on: "{topic}"

Persona: {self.ctx.persona}
Preferences: {tone} tone, {technicality} technical level, {length} length

Your task: Create a comprehensive synthesis that will be used to generate a final brief.

KEY FACTS & STATISTICS:
{facts_text}

CONTENT BY THEME:
{themes_text}

SOURCES ANALYZED: {len(relevant_chunks)} items from {len(sources)} sources

Create a synthesis that includes:
1. **Overview** - What's the current state/landscape?
2. **Key Insights** - What are the most important findings? (Use specific facts/numbers)
3. **Themes & Patterns** - What themes emerged? What connects different pieces?
4. **Notable Details** - Interesting examples, quotes, or surprising findings
5. **Implications** - What does this mean? Why does it matter?
6. **Knowledge Gaps** - What's missing or uncertain?

Guidelines:
- Be specific and concrete - use the facts and numbers provided
- Cite interesting insights with [Source: X] when relevant
- Aim for ~600-800 words
- Structure clearly with headers
- Connect insights meaningfully
- Highlight what's most actionable or surprising

Write the synthesis now:"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.6,
                max_tokens=2000
            )
            
            synthesis = response.choices[0].message.content.strip()
            
            # Add metadata footer
            synthesis += f"\n\n---\n**Synthesis Metadata:**\n"
            synthesis += f"- Analyzed {len(relevant_chunks)} relevant items\n"
            synthesis += f"- Identified {len(themes)} major themes\n"
            synthesis += f"- Extracted {len(key_facts)} key facts\n"
            synthesis += f"- Sources: {len(sources)} unique sources\n"
            
            print("[Synthesizer] ✅ Synthesis complete")
            return synthesis
            
        except Exception as e:
            print(f"[Synthesizer] ✗ Error during synthesis: {e}")
            
            # Fallback to simpler synthesis
            combined = "\n\n".join([
                f"**{c.get('title', 'Untitled')}**\n{(c.get('content', '') or c.get('summary', ''))[:500]}"
                for c in relevant_chunks[:10]
            ])
            
            return f"## Research Findings on {topic}\n\n{combined}\n\n*Note: Full synthesis failed, showing raw content.*"