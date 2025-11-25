from openai import OpenAI
from rich import print


class BriefAgent:
    def __init__(self, client: OpenAI, ctx):
        self.client = client
        self.ctx = ctx

    def run(self, topic: str, synthesis: str) -> str:
        """
        Generate a comprehensive, well-structured brief from synthesized content.
        """
        print("[BriefAgent] Creating personalized brief…")

        # Build context from available attributes
        context_parts = [f"Persona: {self.ctx.persona}"]
        
        if hasattr(self.ctx, 'preferences') and self.ctx.preferences:
            prefs = self.ctx.preferences
            if isinstance(prefs, dict):
                tone = prefs.get('tone', 'neutral')
                length = prefs.get('length', 'medium')
                technicality = prefs.get('technicality', 'medium')
            else:
                tone = 'neutral'
                length = 'medium'
                technicality = 'medium'
        else:
            tone = 'neutral'
            length = 'medium'
            technicality = 'medium'

        prompt = f"""You are a research assistant creating a personalized brief.

Topic: {topic}
{', '.join(context_parts)}

Tone: {tone}
Length preference: {length}
Technical level: {technicality}

Based on the research synthesis below, create a comprehensive, engaging brief that includes:

1. **Executive Summary** (2-3 sentences) - The absolute key takeaway

2. **Main Findings** (3-5 bullet points) - Core discoveries with specific details, numbers, or examples

3. **Deep Dive** (2-4 paragraphs) - Detailed analysis of the most interesting/important aspects:
   - Explain the context and background
   - Highlight trends, patterns, or surprising insights
   - Include specific examples, quotes, or data points
   - Connect different pieces of information

4. **Practical Implications** - How this affects the reader:
   - What actions they might take
   - What to watch for
   - How this changes understanding of the topic

5. **Key Quotes & Sources** - 2-3 notable quotes or findings with attribution

6. **What's Next** - Future trends, unanswered questions, or areas to explore

Research Synthesis:
{synthesis}

Guidelines:
- Write in {tone} tone
- Aim for {length} length (short=400 words, medium=800 words, long=1200+ words)
- Use {technicality} technical language
- Be specific and concrete, avoid vague statements
- Include numbers, data, and examples when available
- Make it engaging and easy to scan
- Use clear section headers (##)
- Connect insights meaningfully

Format the brief in clean Markdown with proper headers (##), bullet points, and emphasis.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2500
            )

            brief = response.choices[0].message.content.strip()
            print("[BriefAgent] ✓ Brief created successfully")
            return brief

        except Exception as e:
            print(f"[BriefAgent] ✗ Error creating brief: {e}")
            return f"**Error generating brief:** {str(e)}\n\n**Raw synthesis:**\n{synthesis}"