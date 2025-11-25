from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass, field
from rich import print


@dataclass
class APICall:
    """Record of a single API call"""
    agent: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)


class CostTracker:
    """
    Tracks API usage, costs, and agent activity throughout the pipeline.
    """
    
    # OpenAI pricing (as of Nov 2024)
    PRICING = {
        "gpt-4o": {
            "prompt": 0.0025 / 1000,      # $2.50 per 1M tokens
            "completion": 0.010 / 1000     # $10.00 per 1M tokens
        },
        "gpt-4o-mini": {
            "prompt": 0.00015 / 1000,      # $0.15 per 1M tokens
            "completion": 0.0006 / 1000    # $0.60 per 1M tokens
        },
        "text-embedding-3-small": {
            "prompt": 0.00002 / 1000,      # $0.02 per 1M tokens
            "completion": 0.0                # No completion tokens
        }
    }
    
    def __init__(self):
        self.calls: List[APICall] = []
        self.agent_logs: List[str] = []
        self.start_time = datetime.now()
    
    def log_agent_activity(self, message: str):
        """Log agent activity for display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.agent_logs.append(log_entry)
        print(log_entry)  # Also print to console
    
    def track_call(self, agent: str, model: str, response) -> float:
        """
        Track an API call and calculate cost.
        
        Args:
            agent: Name of the agent making the call
            model: Model being used
            response: OpenAI response object
        
        Returns:
            Cost of this call in USD
        """
        try:
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Calculate cost
            if model in self.PRICING:
                pricing = self.PRICING[model]
                cost = (prompt_tokens * pricing["prompt"] + 
                       completion_tokens * pricing["completion"])
            else:
                # Unknown model, use gpt-4o pricing as fallback
                pricing = self.PRICING["gpt-4o"]
                cost = (prompt_tokens * pricing["prompt"] + 
                       completion_tokens * pricing["completion"])
            
            # Record the call
            call = APICall(
                agent=agent,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost
            )
            self.calls.append(call)
            
            self.log_agent_activity(
                f"{agent}: {model} - {total_tokens:,} tokens (${cost:.4f})"
            )
            
            return cost
            
        except Exception as e:
            print(f"[CostTracker] Error tracking call: {e}")
            return 0.0
    
    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        if not self.calls:
            return {
                "total_calls": 0,
                "total_cost": 0.0,
                "total_tokens": 0,
                "by_agent": {},
                "by_model": {},
                "duration": 0
            }
        
        total_cost = sum(call.cost for call in self.calls)
        total_tokens = sum(call.total_tokens for call in self.calls)
        
        # Group by agent
        by_agent = {}
        for call in self.calls:
            if call.agent not in by_agent:
                by_agent[call.agent] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            by_agent[call.agent]["calls"] += 1
            by_agent[call.agent]["tokens"] += call.total_tokens
            by_agent[call.agent]["cost"] += call.cost
        
        # Group by model
        by_model = {}
        for call in self.calls:
            if call.model not in by_model:
                by_model[call.model] = {
                    "calls": 0,
                    "tokens": 0,
                    "cost": 0.0
                }
            by_model[call.model]["calls"] += 1
            by_model[call.model]["tokens"] += call.total_tokens
            by_model[call.model]["cost"] += call.cost
        
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "total_calls": len(self.calls),
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "prompt_tokens": sum(call.prompt_tokens for call in self.calls),
            "completion_tokens": sum(call.completion_tokens for call in self.calls),
            "by_agent": by_agent,
            "by_model": by_model,
            "duration": duration
        }
    
    def format_report(self) -> str:
        """Generate a formatted report for display"""
        stats = self.get_stats()
        
        if stats["total_calls"] == 0:
            return "No API calls made yet."
        
        report = f"""## 💰 Cost & Usage Report

**Total Cost:** ${stats['total_cost']:.4f}
**Total Tokens:** {stats['total_tokens']:,} ({stats['prompt_tokens']:,} prompt + {stats['completion_tokens']:,} completion)
**API Calls:** {stats['total_calls']}
**Duration:** {stats['duration']:.1f}s

### 📊 By Agent
"""
        
        for agent, data in sorted(stats['by_agent'].items(), key=lambda x: x[1]['cost'], reverse=True):
            report += f"\n**{agent}**\n"
            report += f"- Calls: {data['calls']}\n"
            report += f"- Tokens: {data['tokens']:,}\n"
            report += f"- Cost: ${data['cost']:.4f}\n"
        
        report += "\n### 🤖 By Model\n"
        
        for model, data in sorted(stats['by_model'].items(), key=lambda x: x[1]['cost'], reverse=True):
            report += f"\n**{model}**\n"
            report += f"- Calls: {data['calls']}\n"
            report += f"- Tokens: {data['tokens']:,}\n"
            report += f"- Cost: ${data['cost']:.4f}\n"
        
        return report
    
    def get_activity_log(self) -> str:
        """Get formatted activity log"""
        if not self.agent_logs:
            return "No activity logged yet."
        
        return "\n".join(self.agent_logs[-50:])  # Last 50 entries