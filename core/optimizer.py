from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class DeploymentStrategy:
    """
    Structured object representing a model deployment strategy.
    """
    precision: str
    prune_ratio: float
    expected_memory_multiplier: float
    expected_latency_multiplier: float
    deployment_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyGenerator:
    """
    Generates realistic deployment strategies based on varying precision,
    pruning ratios, and deployment modes.
    Implements early filtering and Pareto-style pruning.
    """
    
    def __init__(self):
        self.precisions = ["fp32", "fp16", "int8", "int4"]
        self.prune_ratios = [0.0, 0.2, 0.4, 0.6]
        self.deployment_modes = ["balanced", "high_performance", "low_power"]

    def generate_all_combinations(self) -> List[DeploymentStrategy]:
        """
        Generate all base combinations of strategies, filtering out explicitly unrealistic ones.
        """
        import itertools
        strategies = []
        for p, pr, dm in itertools.product(self.precisions, self.prune_ratios, self.deployment_modes):
            
            # EARLY FILTERING: Remove unrealistic combinations
            if p == "int4" and pr > 0.4:
                continue # int4 with high pruning is mathematically destroyed
            
            if p == "fp32" and dm == "low_power":
                continue # FP32 is never low power
                
            # Base logic multipliers
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
                expected_latency_multiplier=base_lat * (1.0 - pr * 0.5), # Unstructured pruning inefficiency
                deployment_mode=dm
            ))
        return strategies

    def pareto_filter(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes strictly dominated strategies. 
        A strategy is dominated if another strategy is better in ALL key metrics.
        """
        filtered = []
        for i, ev_A in enumerate(evaluations):
            dominated = False
            for j, ev_B in enumerate(evaluations):
                if i == j: continue
                
                # We check if B dominates A. 
                # (Lower latency is better, Lower Memory is better, Lower Cost is better, Lower accuracy penalty is better)
                sim_A = ev_A['simulation']
                sim_B = ev_B['simulation']
                
                if (sim_B['latency_estimate'] <= sim_A['latency_estimate'] and
                    sim_B['adjusted_memory'] <= sim_A['adjusted_memory'] and
                    sim_B['cost_estimate'] <= sim_A['cost_estimate'] and
                    sim_B.get('accuracy_penalty', 0) <= sim_A.get('accuracy_penalty', 0)):
                    
                    # Ensure B is strictly better in at least one
                    if (sim_B['latency_estimate'] < sim_A['latency_estimate'] or
                        sim_B['adjusted_memory'] < sim_A['adjusted_memory'] or
                        sim_B['cost_estimate'] < sim_A['cost_estimate'] or
                        sim_B.get('accuracy_penalty', 0) < sim_A.get('accuracy_penalty', 0)):
                        dominated = True
                        break
            
            if not dominated:
                filtered.append(ev_A)
                
        return filtered

    def search_best_strategies(self, evaluator_callback: Callable[[List[DeploymentStrategy]], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Evaluate and filter strategies smartly.
        """
        all_combinations = self.generate_all_combinations()
        evaluations = evaluator_callback(all_combinations)
        
        # Apply Pareto filtering
        pareto_front = self.pareto_filter(evaluations)
        
        # If pareto front is too small, fallback to top 50%
        if len(pareto_front) < 5:
            num_keep = max(5, len(evaluations) // 2)
            top_evals = evaluations[:num_keep]
            return top_evals
            
        # Re-sort just to be sure
        pareto_front.sort(key=lambda x: x['score'], reverse=True)
        return pareto_front

    def generate_strategies(self) -> List[DeploymentStrategy]:
        return self.generate_all_combinations()
