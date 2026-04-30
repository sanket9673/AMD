import os
import logging
import asyncio
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Reasoning Engine powered by Groq LLMs.
    Provides fast, dynamic explanations for AI deployment strategies.
    Falls back to a robust template if API keys are missing or calls fail.
    """
    def __init__(self, mode: str = "llama-3.1-8b-instant"):
        self.model_name = mode
        self.client = None
        self._init_client()

    def _init_client(self):
        try:
            from groq import AsyncGroq
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                self.client = AsyncGroq(api_key=api_key)
            else:
                logger.warning("GROQ_API_KEY not found. Reasoning engine will use fallback template.")
        except ImportError:
            logger.warning("groq package not installed. Reasoning engine will use fallback template.")
            self.client = None

    async def generate_explanation_async(self, strategy_metrics: Dict[str, Any], use_llm: bool = False) -> str:
        fallback_text = self._generate_fallback(strategy_metrics)
        
        if not use_llm or not self.client:
            return fallback_text
            
        try:
            system_prompt = "You are an expert AI deployment architect. Provide a brief, professional explanation for the selected deployment strategy."
            
            metrics_str = "Strategy Metrics:\n"
            for key, value in strategy_metrics.items():
                metrics_str += f"- {key}: {value}\n"

            user_prompt = f"Based on these metrics, explain why this is the optimal deployment strategy. Mention hardware tradeoffs and accuracy preservation.\n\n{metrics_str}"
            
            logger.info(f"Generating insight using Groq ({self.model_name})...")
            completion = await self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model_name,
                temperature=0.3,
                max_tokens=256,
                timeout=10.0
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq LLM Generation failed: {e}")
            return f"LLM Generation Failed. Fallback insight: {fallback_text}"

    def generate_explanation(self, strategy_metrics: Dict[str, Any], use_llm: bool = False) -> str:
        """Synchronous wrapper for pipeline engine"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # Create a new thread to run the async function if loop is running
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
        prec = metrics.get('Precision', 'Unknown')
        hw = metrics.get('Hardware', 'Unknown')
        lat = metrics.get('Latency (ms)', 0)
        acc_pen = metrics.get('Accuracy Penalty', 0)
        
        explanation = (
            f"The selected {prec} strategy on {hw} was chosen because it optimally balances "
            f"latency ({lat:.1f}ms) and infrastructure cost while staying within memory bounds. "
        )
        if acc_pen > 0:
            explanation += f"Note: An estimated accuracy degradation penalty of {acc_pen*100:.0f}% is factored into the efficiency score."
        else:
            explanation += "This strategy preserves near-baseline accuracy."
            
        return explanation
