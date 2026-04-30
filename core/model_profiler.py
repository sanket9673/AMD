import os
import logging
from transformers import AutoConfig
from dotenv import load_dotenv

# Suppress loud HTTP logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

class ModelProfiler:
    """
    Extracts core architectural dimensions from a HuggingFace model without 
    loading full weights into VRAM. Designed to safely handle gated models.
    """
    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.config = None
        self.model_id = None
        self.status = "unknown"

    def load_model(self, model_id: str):
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        try:
            self.config = AutoConfig.from_pretrained(model_id, token=hf_token)
            self.model_id = model_id
            self.status = "ready"
        except Exception:
            # Silently fallback without printing raw HTTP traces
            try:
                fallback_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                self.config = AutoConfig.from_pretrained(fallback_id, token=hf_token)
                self.model_id = fallback_id
                self.status = "fallback"
            except Exception:
                # Absolute worst-case scenario: create dummy config
                class DummyConfig:
                    hidden_size = 4096
                    num_hidden_layers = 32
                    num_attention_heads = 32
                    vocab_size = 32000
                self.config = DummyConfig()
                self.model_id = "Generic-LLM (Offline)"
                self.status = "fallback"

    def generate_profile(self) -> dict:
        if not self.config:
            raise ValueError("Model not loaded. Call load_model() first.")

        hidden_size = getattr(self.config, "hidden_size", 4096)
        num_layers = getattr(self.config, "num_hidden_layers", 32)
        
        # Mathematical estimation of parameters (FFN is usually 4x hidden)
        params = (hidden_size * hidden_size * 4 + hidden_size * (hidden_size * 4) * 2) * num_layers
        params += getattr(self.config, "vocab_size", 32000) * hidden_size # Embedding
        
        return {
            "model_id": self.model_id,
            "status": self.status,
            "hidden_size": hidden_size,
            "model_depth": num_layers,
            "total_parameters": params
        }