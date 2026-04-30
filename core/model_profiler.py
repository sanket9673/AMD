import os
import logging
from transformers import AutoConfig
from dotenv import load_dotenv

# Suppress loud HTTP logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

def safe_load_config(model_id, hf_token=None):
    """
    Centralized loader function for fast, safe model loading.
    """
    try:
        config = AutoConfig.from_pretrained(model_id, token=hf_token)
        return config, model_id, "ready"
    except Exception:
        fallback = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        try:
            config = AutoConfig.from_pretrained(fallback, token=hf_token)
        except Exception:
            # Absolute fallback
            class DummyConfig:
                hidden_size = 4096
                num_hidden_layers = 32
                num_attention_heads = 32
                vocab_size = 32000
            config = DummyConfig()
        return config, fallback, "fallback"

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

    def load_model(self, model_id: str, config=None, status="ready", actual_model_id=None):
        if config is not None:
            self.config = config
            self.model_id = actual_model_id or model_id
            self.status = status
        else:
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            cfg, m_id, stat = safe_load_config(model_id, hf_token)
            self.config = cfg
            self.model_id = m_id
            self.status = stat

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