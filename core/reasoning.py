import logging
import os
import streamlit as st

logger = logging.getLogger(__name__)

# Safe environment loading
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class ReasoningEngine:
    def __init__(self, mode: str = "llama-3.1-8b-instant"):
        self.mode = mode
        self.client = None
        
        # Production Safe ENV Handling
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY", None)
            except Exception:
                pass
                
        if groq_api_key:
            try:
                from groq import Groq
                self.client = Groq(api_key=groq_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")

    def generate_explanation(self, metrics: dict, use_llm: bool = True) -> str:
        # Fallback heuristic explanation
        base_exp = (
            "This strategy reduces memory transfer overhead by leveraging lower precision types, "
            "shifting the bottleneck towards compute. This achieves significant latency improvements "
            "while strictly abiding by your cost and constraint targets."
        )

        if not use_llm or not self.client:
            return base_exp

        try:
            prompt = (
                f"You are a Staff-Level AI Infrastructure Engineer at AMD. Explain in 2-3 short, professional sentences "
                f"why this deployment strategy is mathematically optimal based on Roofline Performance Modeling.\n\n"
                f"Metrics: {metrics}\n\n"
                f"Focus on the exact metrics (e.g. Ridge Point, Precision). No emojis. No fluff. Be highly technical."
            )

            completion = self.client.chat.completions.create(
                model=self.mode,
                messages=[
                    {"role": "system", "content": "You are an expert AI infrastructure engineer. Be precise and concise."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150,
            )

            return completion.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return base_exp
