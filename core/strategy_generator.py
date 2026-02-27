import math
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable

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
    Implements self-improving adaptive strategy search.
    """
    
    def __init__(self):
        self.precisions = ["fp32", "fp16", "int8", "int4"]
        self.prune_ratios = [0.0, 0.2, 0.4, 0.6]
        self.deployment_modes = ["balanced", "high_performance", "low_power"]

    def generate_all_combinations(self) -> List[DeploymentStrategy]:
        """
        Generate all base combinations of strategies based on parameter space.
        """
        import itertools
        strategies = []
        for p, pr, dm in itertools.product(self.precisions, self.prune_ratios, self.deployment_modes):
            
            # Simple heuristic scaling
            base_mem = 1.0
            if p == "fp16": base_mem = 0.5
            elif p == "int8": base_mem = 0.25
            elif p == "int4": base_mem = 0.125
            
            base_lat = 1.0
            if p == "fp16": base_lat = 0.6
            elif p == "int8": base_lat = 0.4
            elif p == "int4": base_lat = 0.3
            
            strategies.append(DeploymentStrategy(
                precision=p,
                prune_ratio=pr,
                expected_memory_multiplier=base_mem * (1.0 - pr),
                expected_latency_multiplier=base_lat * (1.0 - pr * 0.5),
                deployment_mode=dm
            ))
        return strategies

    def search_best_strategies(self, evaluator_callback: Callable[[List[DeploymentStrategy]], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Evaluate all combinations and prune bottom 50% after scoring.
        Iteratively refine top 10%. Creates adaptive search.
        
        Args:
            evaluator_callback: A function that takes a list of strategies and returns sorted evaluations.
        """
        all_combinations = self.generate_all_combinations()
        evaluations = evaluator_callback(all_combinations)
        
        # Prune bottom 50%
        num_keep = int(max(1, len(evaluations) // 2))
        top_half_evals = evaluations[:num_keep]
        
        # Iteratively refine top 10%
        num_refine = int(max(1, len(top_half_evals) // 10))
        top_10_percent = top_half_evals[:num_refine]
        
        refined_strategies = []
        # Keep all the top half first
        for ev in top_half_evals:
            refined_strategies.append(ev["strategy"])
            
        # Add refinements for the top 10%
        for ev in top_10_percent:
            strat: DeploymentStrategy = ev["strategy"]
            # Create a refined child strategy: slightly higher prune ratio
            new_prune_val = min(0.9, round(strat.prune_ratio + 0.05, 2))
                
            child = DeploymentStrategy(
                precision=strat.precision,
                prune_ratio=new_prune_val,
                expected_memory_multiplier=float(strat.expected_memory_multiplier * 0.95),
                expected_latency_multiplier=float(strat.expected_latency_multiplier * 0.98),
                deployment_mode=strat.deployment_mode
            )
            refined_strategies.append(child)
            
        final_evaluations = evaluator_callback(refined_strategies)
        return final_evaluations

    def generate_strategies(self) -> List[DeploymentStrategy]:
        """
        Fallback for the old pipeline that expected a base list of strategies.
        """
        return self.generate_all_combinations()

if __name__ == "__main__":
    generator = StrategyGenerator()
    strats = generator.generate_all_combinations()
    print(f"Generated {len(strats)} baseline strategies.")
