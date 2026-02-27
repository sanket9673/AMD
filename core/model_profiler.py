import os
import json
import logging
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelProfiler:

    def __init__(self, output_dir: str = "data"):
        self.output_dir = output_dir
        self.model_id: Optional[str] = None
        self.config = None
        self.model = None

        os.makedirs(self.output_dir, exist_ok=True)

    def load_model(self, model_id: str) -> None:
        logger.info(f"Loading model {model_id} for profiling on CPU...")
        import torch
        from transformers import AutoConfig, AutoModel

        self.model_id = model_id

        try:
            self.config = AutoConfig.from_pretrained(model_id)

            self.model = AutoModel.from_pretrained(
                model_id,
                config=self.config,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )

            self.model.to(torch.device("cpu"))
            self.model.eval()

            logger.info(f"Successfully loaded {model_id} on CPU.")

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def count_parameters(self) -> Dict[str, int]:
        if self.model is None:
            raise ValueError("Model not loaded.")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params
        }

    def estimate_memory(self, batch_size: int = 1, sequence_length: int = 512) -> float:
        if self.model is None:
            raise ValueError("Model not loaded.")

        params = self.count_parameters()["total_parameters"]
        param_memory = params * 4  # float32

        hidden_size = getattr(self.config, "hidden_size", 768)
        num_layers = getattr(self.config, "num_hidden_layers", 12)

        activation_memory = (
            hidden_size * sequence_length * batch_size * num_layers * 4 * 2
        )

        total_mb = (param_memory + activation_memory) / (1024 ** 2)

        return total_mb

    def estimate_flops(self, batch_size: int = 1, sequence_length: int = 512) -> int:
        params = self.count_parameters()["total_parameters"]
        return 2 * params * sequence_length * batch_size

    def generate_profile(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not loaded.")

        params = self.count_parameters()

        profile = {
            "model_id": self.model_id,
            "total_parameters": params["total_parameters"],
            "trainable_parameters": params["trainable_parameters"],
            "estimated_memory_mb": self.estimate_memory(),
            "estimated_flops": self.estimate_flops(),
            "model_depth": getattr(self.config, "num_hidden_layers", 0),
            "hidden_size": getattr(self.config, "hidden_size", None)
        }

        profile_path = os.path.join(self.output_dir, "model_profile.json")
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=4)

        logger.info(f"Profile saved to {profile_path}")

        return profile