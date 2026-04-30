import os
import logging
import asyncio
from typing import Dict, Any
from dotenv import load_dotenv

# Suppress loud HTTP logs
logging.getLogger("httpx").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Reasoning Engine powered by Groq LLMs.
    Provides fast, dynamic explanations for AI deployment strategies.
    Falls back gracefully if API keys are missing.
    """
    def __init__(self, mode: str = "llama-3.1-8b-instant"):
        self.model_name = mode
        self.client = None
        self._init_client()

    def _init_client(self):
        load_dotenv()
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            self.client = None
            return
            
        try:
            from groq import AsyncGroq
            self.client = AsyncGroq(api_key=api_key)
        except ImportError:
            self.client = None

    async def generate_explanation_async(self, strategy_metrics: Dict[str, Any], use_llm: bool = False) -> str:
        fallback_text = self._generate_fallback(strategy_metrics)
        
        if not use_llm or not self.client:
            return fallback_text
            
        try:
            system_prompt = (
                "You are a Staff-Level AI Infrastructure Engineer. "
                "Provide a brief, professional, non-technical explanation for the selected deployment strategy. "
                "Do NOT use markdown headers, bold text, emojis, or bullet points. "
                "Explain the choice in 2-3 clean, readable sentences focusing on memory, latency, and cost limits."
            )
            
            metrics_str = "Strategy Metrics:\n"
            for key, value in strategy_metrics.items():
                metrics_str += f"- {key}: {value}\n"

            user_prompt = f"Based on these metrics, provide a highly professional explanation for why this is the optimal deployment strategy:\n\n{metrics_str}"
            
            completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.1,
                max_tokens=150,
                timeout=5.0
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception:
            return fallback_text

    def generate_explanation(self, strategy_metrics: Dict[str, Any], use_llm: bool = False) -> str:
        """Synchronous wrapper for pipeline engine"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            import threading
            result = [None]
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                result[0] = new_loop.run_until_complete(self.generate_explanation_async(strategy_metrics, use_llm))
                new_loop.close()
            t = threading.Thread(target=run_in_thread)
            t.start()
            t.join()
            return result[0]
        else:
            return loop.run_until_complete(self.generate_explanation_async(strategy_metrics, use_llm))

    def _generate_fallback(self, metrics: Dict[str, Any]) -> str:
        """Template-based reasoning fallback."""
        if not self.client and os.environ.get("GROQ_API_KEY") is None:
            prefix = "AI reasoning running in offline mode. "
        else:
            prefix = ""
            
        ai = metrics.get('Prefill AI', 0.0)
        ridge = metrics.get('Ridge Point', 0.0)
        
        regime = "compute-bound" if ai >= ridge else "memory-bound"
            
        explanation = (
            f"{prefix}This strategy operates in a {regime} regime during prefill. "
            f"The selected precision reduces memory transfer overhead, "
            f"improving overall throughput while maintaining strict latency and cost constraints."
        )
        return explanation
