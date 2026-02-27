import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class DeploymentStrategy:
    """
    Structured object representing a model deployment strategy.
    Defines precision, pruning, performance multipliers, and deployment mode.
    """
    precision: str
    prune_ratio: float
    expected_memory_multiplier: float
    expected_latency_multiplier: float
    deployment_mode: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy object to dictionary."""
        return asdict(self)

class StrategyGenerator:
    """
    Generates realistic deployment strategies based on varying precision,
    pruning ratios, and deployment modes.
    """
    
    def __init__(self):
        pass

    def generate_strategies(self) -> List[DeploymentStrategy]:
        """
        Generate 10 predefined but realistic deployment strategies.
        Returns a list of structured Strategy objects.
        """
        strategies = [
            # 1. Baseline: Full precision, no pruning
            DeploymentStrategy(
                precision="fp32", 
                prune_ratio=0.0, 
                expected_memory_multiplier=1.0, 
                expected_latency_multiplier=1.0, 
                deployment_mode="high_performance"
            ),
            # 2. FP16: Half precision, no pruning
            DeploymentStrategy(
                precision="fp16", 
                prune_ratio=0.0, 
                expected_memory_multiplier=0.5, 
                expected_latency_multiplier=0.6, 
                deployment_mode="high_performance"
            ),
            # 3. FP16 with mild pruning (20%)
            DeploymentStrategy(
                precision="fp16", 
                prune_ratio=0.2, 
                expected_memory_multiplier=0.4, 
                expected_latency_multiplier=0.5, 
                deployment_mode="balanced"
            ),
            # 4. FP32 with aggressive pruning (40%)
            DeploymentStrategy(
                precision="fp32", 
                prune_ratio=0.4, 
                expected_memory_multiplier=0.6, 
                expected_latency_multiplier=0.75, 
                deployment_mode="balanced"
            ),
            # 5. FP16 with aggressive pruning (40%)
            DeploymentStrategy(
                precision="fp16", 
                prune_ratio=0.4, 
                expected_memory_multiplier=0.3, 
                expected_latency_multiplier=0.45, 
                deployment_mode="high_performance"
            ),
            # 6. INT8: Quarter precision, no pruning
            DeploymentStrategy(
                precision="int8", 
                prune_ratio=0.0, 
                expected_memory_multiplier=0.25, 
                expected_latency_multiplier=0.4, 
                deployment_mode="balanced"
            ),
            # 7. INT8 with mild pruning (20%)
            DeploymentStrategy(
                precision="int8", 
                prune_ratio=0.2, 
                expected_memory_multiplier=0.2, 
                expected_latency_multiplier=0.35, 
                deployment_mode="balanced"
            ),
            # 8. INT8 with aggressive pruning (40%)
            DeploymentStrategy(
                precision="int8", 
                prune_ratio=0.4, 
                expected_memory_multiplier=0.15, 
                expected_latency_multiplier=0.25, 
                deployment_mode="low_power"
            ),
            # 9. INT4: Eighth precision, no pruning
            DeploymentStrategy(
                precision="int4", 
                prune_ratio=0.0, 
                expected_memory_multiplier=0.125, 
                expected_latency_multiplier=0.3, 
                deployment_mode="low_power"
            ),
            # 10. INT4 with aggressive pruning (40%)
            DeploymentStrategy(
                precision="int4", 
                prune_ratio=0.4, 
                expected_memory_multiplier=0.075, 
                expected_latency_multiplier=0.2, 
                deployment_mode="low_power"
            )
        ]
        return strategies

if __name__ == "__main__":
    generator = StrategyGenerator()
    strats = generator.generate_strategies()
    print(f"Generated {len(strats)} strategies.")
    for strat in strats:
        print(strat)
