import os
import json
import logging
from typing import Dict, Any, Optional
from transformers import AutoConfig

logger = logging.getLogger(__name__)

class ModelProfiler:
    """
    Lightweight Model Profiler.
    Uses HuggingFace AutoConfig to mathematically estimate model parameters,
    memory footprint, and FLOPs WITHOUT loading model weights into memory.
    """

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.model_id: Optional[str] = None
        self.config = None
        self.stats = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self, model_id: str) -> None:
        """
        Loads the model configuration without downloading or loading weights.
        Falls back to a default configuration if the model is gated or fails to load.
        """
        logger.info(f"Lightweight profiling mode (no weights loaded) for {model_id}...")
        self.model_id = model_id

        try:
            self.config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            logger.info(f"Successfully loaded configuration for {model_id}.")
        except Exception as e:
            logger.error(f"Failed to load config for {model_id} (e.g. gated model). Using fallback TinyLlama profile. Error: {e}")
            self.model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            try:
                self.config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            except Exception as e2:
                logger.error(f"Fallback also failed! Generating dummy config. Error: {e2}")
                # Create a dummy class that behaves like config
                class DummyConfig:
                    hidden_size = 2048
                    intermediate_size = 8192
                    num_hidden_layers = 16
                    vocab_size = 32000
                    tie_word_embeddings = False
                self.config = DummyConfig()

    def estimate_parameters(self) -> Dict[str, int]:
        hidden_size = getattr(self.config, "hidden_size", 4096)
        intermediate_size = getattr(self.config, "intermediate_size", hidden_size * 4)
        num_layers = getattr(self.config, "num_hidden_layers", 32)
        vocab_size = getattr(self.config, "vocab_size", 32000)

        emb_params = vocab_size * hidden_size
        attn_params_per_layer = 4 * (hidden_size * hidden_size)
        mlp_params_per_layer = 3 * (hidden_size * intermediate_size)
        ln_params = (num_layers * 2 + 1) * hidden_size
        tie_word_embeddings = getattr(self.config, "tie_word_embeddings", False)
        head_params = 0 if tie_word_embeddings else (vocab_size * hidden_size)

        total_params = emb_params + (attn_params_per_layer + mlp_params_per_layer) * num_layers + ln_params + head_params

        self.stats["total_parameters"] = total_params
        return {"total_parameters": total_params}

    def estimate_memory(self, batch_size: int = 1, sequence_length: int = 1024) -> float:
        if "total_parameters" not in self.stats:
            self.estimate_parameters()

        params = self.stats["total_parameters"]
        param_memory_mb = (params * 2) / (1024 ** 2)

        hidden_size = getattr(self.config, "hidden_size", 4096)
        num_layers = getattr(self.config, "num_hidden_layers", 32)
        activation_memory_mb = (2 * 2 * num_layers * batch_size * sequence_length * hidden_size) / (1024 ** 2)

        total_mb = param_memory_mb + activation_memory_mb
        return total_mb

    def estimate_flops(self, batch_size: int = 1, sequence_length: int = 1024) -> int:
        if "total_parameters" not in self.stats:
            self.estimate_parameters()
            
        params = self.stats["total_parameters"]
        return 2 * params * sequence_length * batch_size

    def generate_profile(self) -> Dict[str, Any]:
        self.estimate_parameters()

        profile = {
            "model_id": self.model_id,
            "total_parameters": self.stats["total_parameters"],
            "trainable_parameters": self.stats["total_parameters"],
            "estimated_memory_mb": self.estimate_memory(),
            "estimated_flops": self.estimate_flops(),
            "model_depth": getattr(self.config, "num_hidden_layers", 32),
            "hidden_size": getattr(self.config, "hidden_size", 4096)
        }

        profile_path = os.path.join(self.output_dir, "model_profile.json")
        try:
            with open(profile_path, "w") as f:
                json.dump(profile, f, indent=4)
        except IOError:
            pass

        return profile