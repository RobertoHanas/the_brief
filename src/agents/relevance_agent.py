from openai import OpenAI
from rich import print
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed


class RelevanceAgent:
    def __init__(self, client: OpenAI, ctx, output_dir: str = "output/relevance"):
        self.client = client
        self.ctx = ctx
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's embedding model."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",  # Cheaper and faster
                input=text[:8000]  # Limit input length
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"[RelevanceAgent] Embedding error: {e}")
            return None

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def batch_embeddings(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Get embeddings for multiple texts efficiently."""
        embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"[RelevanceAgent] 🔄 Processing embeddings batch {batch_num}/{total_batches} ({len(batch)} items)...")
            
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch
                )
                embeddings.extend([data.embedding for data in response.data])
                print(f"[RelevanceAgent] ✓ Batch {batch_num}/{total_batches} complete")
            except Exception as e:
                print(f"[RelevanceAgent] ✗ Batch {batch_num} embedding error: {e}")
                # Return zero vectors for failed batch
                embeddings.extend([[0.0] * 1536 for _ in batch])
        
        return embeddings

    def score_with_llm(self, topic: str, text: str) -> float:
        """Fallback: Use LLM to score relevance (more expensive, use sparingly)."""
        prompt = f"""
        Rate how relevant this content is to the topic: '{topic}'
        Consider the persona: {self.ctx.persona}

        Respond with ONLY a number between 0.0 and 1.0.
        
        Content:
        {text[:1000]}
        """

        try:
            resp = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for cost savings
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10
            )
            
            content = resp.choices[0].message.content.strip()
            # Extract first number found
            import re
            match = re.search(r'0?\.\d+|[01]\.?\d*', content)
            if match:
                return float(match.group())
            return 0.0
        except Exception as e:
            print(f"[RelevanceAgent] LLM scoring error: {e}")
            return 0.0

    def run(self, topic: str, chunks: List[Dict[str, Any]], 
            method: str = "embedding", 
            threshold: float = 0.4,
            save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Score and filter chunks by relevance.
        
        Args:
            topic: The research topic
            chunks: List of content items to score
            method: 'embedding' (fast, cheap) or 'llm' (slow, expensive)
            threshold: Minimum score to keep (0.0 to 1.0)
            save_results: Whether to save scoring results to file
        """
        print(f"[RelevanceAgent] Scoring {len(chunks)} items using {method} method…")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if method == "embedding":
            # FAST METHOD: Use embeddings (recommended)
            print("[RelevanceAgent] Getting topic embedding…")
            # Build context from available attributes
            context_parts = [topic, f"Persona: {self.ctx.persona}"]
            
            # Add interests if available
            if hasattr(self.ctx, 'interests') and self.ctx.interests:
                context_parts.append(f"Interests: {', '.join(self.ctx.interests)}")
            
            # Add preferences if available
            if hasattr(self.ctx, 'preferences') and self.ctx.preferences:
                context_parts.append(f"Preferences: {self.ctx.preferences}")
            
            topic_context = ". ".join(context_parts)
            topic_embedding = self.get_embedding(topic_context)
            
            if not topic_embedding:
                print("[RelevanceAgent] Failed to get topic embedding, falling back to LLM")
                return self.run(topic, chunks, method="llm", threshold=threshold)
            
            # Get content texts
            print("[RelevanceAgent] Extracting content…")
            content_texts = []
            for ch in chunks:
                content = ch.get("content") or ch.get("text") or ch.get("summary") or ""
                title = ch.get("title", "")
                # Combine title and content for better matching
                combined = f"{title}. {content}"[:8000]
                content_texts.append(combined)
            
            # Get embeddings in batches
            print("[RelevanceAgent] Computing embeddings…")
            content_embeddings = self.batch_embeddings(content_texts)
            
            # Calculate similarities
            print("[RelevanceAgent] Computing similarity scores…")
            results = []
            scores_data = []
            accepted_count = 0
            
            for i, (ch, emb) in enumerate(zip(chunks, content_embeddings)):
                if emb and emb[0] != 0.0:  # Check if valid embedding
                    score = self.cosine_similarity(topic_embedding, emb)
                    # Normalize from [-1, 1] to [0, 1]
                    score = (score + 1) / 2
                else:
                    score = 0.0
                
                scores_data.append({
                    "index": i,
                    "title": ch.get("title", "")[:100],
                    "source": ch.get("source", "")[:100],
                    "score": round(score, 4),
                    "accepted": score >= threshold
                })
                
                if score >= threshold:
                    ch["relevance_score"] = score
                    results.append(ch)
                    accepted_count += 1
                
                # Progress updates every 50 items
                if (i + 1) % 50 == 0:
                    acceptance_rate = (accepted_count / (i + 1)) * 100
                    print(f"[RelevanceAgent] 📊 Progress: {i + 1}/{len(chunks)} | Accepted: {accepted_count} ({acceptance_rate:.1f}%)")
            
            # Final progress
            final_rate = (accepted_count / len(chunks)) * 100 if chunks else 0
            print(f"[RelevanceAgent] ✓ Scoring complete: {accepted_count}/{len(chunks)} accepted ({final_rate:.1f}%)")
        
        else:
            # SLOW METHOD: Use LLM for each item
            print("[RelevanceAgent] Using LLM scoring (this will take a while)…")
            print("[RelevanceAgent] ⏳ Starting parallel LLM scoring with 10 workers...")
            results = []
            scores_data = []
            
            # Use threading for parallel API calls
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for i, ch in enumerate(chunks):
                    content = ch.get("content") or ch.get("text") or ch.get("summary") or ""
                    future = executor.submit(self.score_with_llm, topic, content)
                    futures[future] = (i, ch)
                
                completed = 0
                accepted_count = 0
                
                for future in as_completed(futures):
                    i, ch = futures[future]
                    try:
                        score = future.result()
                        
                        scores_data.append({
                            "index": i,
                            "title": ch.get("title", "")[:100],
                            "source": ch.get("source", "")[:100],
                            "score": round(score, 4),
                            "accepted": score >= threshold
                        })
                        
                        if score >= threshold:
                            ch["relevance_score"] = score
                            results.append(ch)
                            accepted_count += 1
                        
                        completed += 1
                        
                        # Progress every 10 items
                        if completed % 10 == 0:
                            acceptance_rate = (accepted_count / completed) * 100
                            print(f"[RelevanceAgent] 📊 Progress: {completed}/{len(chunks)} | Accepted: {accepted_count} ({acceptance_rate:.1f}%)")
                            
                    except Exception as e:
                        print(f"[RelevanceAgent] ✗ Error scoring item {i}: {e}")
                
                final_rate = (accepted_count / len(chunks)) * 100 if chunks else 0
                print(f"[RelevanceAgent] ✓ LLM scoring complete: {accepted_count}/{len(chunks)} accepted ({final_rate:.1f}%)")
        
        # Sort by score
        results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        # Save results to file
        if save_results:
            print("[RelevanceAgent] 💾 Saving results to files...")
            self._save_results(topic, scores_data, results, threshold, method, timestamp)
        
        acceptance_rate = (len(results) / len(chunks) * 100) if chunks else 0
        print(f"[RelevanceAgent] ✅ Complete! Accepted {len(results)} of {len(chunks)} items (threshold: {threshold}, rate: {acceptance_rate:.1f}%)")
        return results

    def _save_results(self, topic: str, scores_data: List[Dict], 
                     results: List[Dict], threshold: float, 
                     method: str, timestamp: str):
        """Save relevance scoring results to files."""
        
        # Create summary
        summary = {
            "topic": topic,
            "persona": str(self.ctx.persona),
            "timestamp": timestamp,
            "method": method,
            "threshold": float(threshold),
            "total_items": len(scores_data),
            "accepted_items": len(results),
            "acceptance_rate": round(len(results) / len(scores_data) * 100, 2) if scores_data else 0,
            "score_stats": {
                "mean": round(np.mean([s["score"] for s in scores_data]), 4) if scores_data else 0,
                "median": round(np.median([s["score"] for s in scores_data]), 4) if scores_data else 0,
                "min": round(min([s["score"] for s in scores_data]), 4) if scores_data else 0,
                "max": round(max([s["score"] for s in scores_data]), 4) if scores_data else 0,
            }
        }
        
        # Convert booleans explicitly in scores_data
        clean_scores_data = []
        for item in scores_data:
            clean_item = {
                "index": int(item["index"]),
                "title": str(item["title"]),
                "source": str(item["source"]),
                "score": float(item["score"]),
                "accepted": bool(item["accepted"])  # Explicitly convert to bool
            }
            clean_scores_data.append(clean_item)
        
        # Save detailed scores
        scores_file = self.output_dir / f"scores_{timestamp}.json"
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "all_scores": clean_scores_data
            }, f, indent=2, ensure_ascii=False)
        
        # Save accepted items - only include JSON-serializable fields
        accepted_file = self.output_dir / f"accepted_{timestamp}.json"
        clean_results = []
        for item in results:
            clean_result = {
                "title": str(item.get("title", "")),
                "source": str(item.get("source", "")),
                "link": str(item.get("link", "")),
                "score": float(item.get("relevance_score", 0)),
                "content": str(item.get("content", ""))[:500] + "..."
            }
            clean_results.append(clean_result)
        
        with open(accepted_file, 'w', encoding='utf-8') as f:
            json.dump({
                "summary": summary,
                "items": clean_results
            }, f, indent=2, ensure_ascii=False)
        
        print(f"[RelevanceAgent] 💾 Results saved to {self.output_dir}")
        print(f"[RelevanceAgent] 📊 Acceptance rate: {summary['acceptance_rate']}%")