import logging
import os

logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import streamlit as st
    _HAS_STREAMLIT = True
except ImportError:
    _HAS_STREAMLIT = False


class ReasoningEngine:
    def __init__(self, mode: str = "llama-3.1-8b-instant"):
        self.mode = mode
        self.client = None

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key and _HAS_STREAMLIT:
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
        base_exp = (
            "This strategy reduces memory transfer overhead by leveraging lower precision types, "
            "shifting the bottleneck towards compute. This achieves significant latency improvements "
            "while strictly abiding by your cost and constraint targets."
        )

        if not use_llm or not self.client:
            return base_exp

        try:
            prompt = (
                f"You are a Staff-Level AI Infrastructure Engineer at AMD. "
                f"Explain in 2-3 short, precise sentences why this deployment strategy is optimal "
                f"based on Roofline Performance Modeling.\n\n"
                f"Metrics: {metrics}\n\n"
                f"STRICT RULES you MUST follow:\n"
                f"1. The Ridge Point ({metrics.get('Ridge Point', 'N/A')}) is a FIXED hardware constant "
                f"equal to Peak FLOPs divided by Peak Bandwidth. Do NOT say that precision changes "
                f"or minimizes the Ridge Point. It is immutable.\n"
                f"2. Explain that lower precision (e.g. INT8) reduces bytes-per-parameter, which "
                f"INCREASES the model Arithmetic Intensity, pushing the workload further into the "
                f"compute-bound region above the Ridge Point.\n"
                f"3. Reference the exact Prefill AI value ({metrics.get('Prefill AI', 'N/A')}) "
                f"compared to the Ridge Point ({metrics.get('Ridge Point', 'N/A')}) to justify "
                f"why the prefill phase is compute-bound.\n"
                f"4. No emojis. No fluff. Be highly technical and concise."
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
