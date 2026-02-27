import os
import torch
import warnings
from typing import Dict, Any, List

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class ReasoningEngine:
    """
    Reasoning Engine powered by TinyLlama.
    Generates human-readable explanations for AI deployment strategies.
    Runs entirely locally on CPU.
    """
    def __init__(self, model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """
        Initializes the ReasoningEngine, loading the TinyLlama model on CPU.
        """
        self.device = "cpu"
        self.model_id = model_id
        
        print(f"Loading {self.model_id} on {self.device}... This may take a moment if downloading.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32, # CPU friendly
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1, # -1 maps to CPU in pipeline
                torch_dtype=torch.float32
            )
            print("TinyLlama loaded successfully on CPU.")
        except Exception as e:
            print(f"Failed to load TinyLlama: {e}")
            self.generator = None

    def generate_explanation(self, strategy_metrics: Dict[str, Any]) -> str:
        """
        Generates a comprehensive explanation using TinyLlama locally.
        
        The explanation includes:
        - Why the best strategy was selected
        - Hardware tradeoffs
        - Cost implications
        - Future scaling suggestions
        
        Args:
            strategy_metrics: A dictionary containing the metrics of the chosen strategy.
            
        Returns:
            A well-formatted string containing the generated explanation.
        """
        if not self.generator:
            return "Error: Reasoning Engine (TinyLlama) could not be loaded."

        system_prompt = "You are an expert AI deployment architect. Provide a well-formatted, professional explanation."
        
        # Format the metrics beautifully into the prompt to give the LLm context
        metrics_str = "Strategy Metrics:\n"
        for key, value in strategy_metrics.items():
            metrics_str += f"- {key}: {value}\n"

        user_prompt = f"""
Based on the following deployment strategy metrics, generate a comprehensive explanation covering exactly these 4 points:
1. Why this strategy was selected as the best
2. Hardware tradeoffs considered
3. Cost implications
4. Future scaling suggestions

{metrics_str}

Return a well-formatted explanation text.
"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Apply the ChatML/TinyLlama specific template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        print("ReasoningEngine: Generating insight using local TinyLlama... ")
        outputs = self.generator(
            prompt, 
            max_new_tokens=512, 
            do_sample=True, 
            temperature=0.3, # Keep it professional and focused
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False # Returns only the newly generated text
        )
        
        explanation = outputs[0]["generated_text"].strip()
        
        return explanation
