import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class StrategyScorer:
    """
    Evaluates and scores deployment strategies across multiple hardware profiles.
    Uses Absolute Baselines instead of Relative Normalization.
    Incorporates an Accuracy Penalty to prevent over-pruning/quantizing.
    """
    def __init__(self):
        # Default weights
        self.weights = {
            'latency': 0.20,
            'memory_efficiency': 0.20,
            'energy_efficiency': 0.20,
            'cost_efficiency': 0.20,
            'accuracy_preservation': 0.20
        }

    def estimate_accuracy_penalty(self, strategy: Any) -> float:
        """
        Estimates the perplexity/accuracy degradation of a given strategy.
        Returns a penalty value between 0.0 (no degradation) and 1.0 (completely broken).
        """
        if isinstance(strategy, dict):
            precision = strategy.get("precision", "fp16")
            prune_ratio = strategy.get("prune_ratio", 0.0)
        else:
            precision = getattr(strategy, "precision", "fp16")
            prune_ratio = getattr(strategy, "prune_ratio", 0.0)

        penalty = 0.0
        
        # Precision penalties
        if precision == "int4":
            penalty += 0.35
        elif precision == "int8":
            penalty += 0.10
        elif precision == "fp16":
            penalty += 0.02
            
        # Pruning penalties (Unstructured pruning destroys accuracy rapidly)
        if prune_ratio > 0.6:
            penalty += 0.50 # Severe penalty, likely unusable
        elif prune_ratio > 0.3:
            penalty += 0.25
        elif prune_ratio > 0.1:
            penalty += 0.05
            
        return min(penalty, 1.0)

    def evaluate(self, 
                 model_profile: Dict[str, Any], 
                 strategies: List[Any], 
                 hardware_profiles: List[Any], 
                 simulator: Any,
                 weights: Optional[Dict[str, float]] = None,
                 workload_type: str = "chat_inference") -> List[Dict[str, Any]]:
        """
        Simulate all combinations of strategies and hardware profiles, score them,
        and return the ranked results.
        """
        if weights:
            # Rebalance dynamically
            self.weights.update(weights)

        evaluations = []
        for hw in hardware_profiles:
            for strat in strategies:
                sim_result = simulator.simulate(model_profile, strat, hw, workload_type=workload_type)
                
                # Calculate accuracy penalty
                accuracy_penalty = self.estimate_accuracy_penalty(strat)
                sim_result["accuracy_penalty"] = accuracy_penalty
                
                evaluations.append({
                    "strategy": strat,
                    "hardware": hw,
                    "simulation": sim_result
                })
        
        return self.score_and_rank(evaluations)
        
    def score_and_rank(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score using absolute thresholds to prevent bad batches from skewing results.
        """
        if not evaluations:
            return []
            
        for e in evaluations:
            sim = e['simulation']
            
            # Latency Score: 0 seconds = 1.0, >2.0 seconds = 0.0
            lat = sim.get('latency_estimate', 2.0)
            norm_latency = max(0.0, 1.0 - (lat / 2.0))
            
            # Memory Efficiency: Using the fit score directly (0 to 1)
            norm_memory = min(100.0, max(0.0, sim.get('hardware_fit_score', 0))) / 100.0
            
            # Energy Efficiency: 0W = 1.0, >500Wh = 0.0
            eng = sim.get('energy_consumption', 500.0)
            norm_energy = max(0.0, 1.0 - (eng / 500.0))

            # Cost Efficiency: $0 = 1.0, >$0.01 per inference = 0.0
            cost = sim.get('cost_estimate', 0.01)
            norm_cost = max(0.0, 1.0 - (cost / 0.01))
            
            # Accuracy Preservation Score
            acc_penalty = sim.get('accuracy_penalty', 0.0)
            norm_acc = 1.0 - acc_penalty

            # Weighted sum
            final_score = (
                norm_latency * self.weights.get('latency', 0.20) +
                norm_memory * self.weights.get('memory_efficiency', 0.20) +
                norm_energy * self.weights.get('energy_efficiency', 0.20) +
                norm_cost * self.weights.get('cost_efficiency', 0.20) +
                norm_acc * self.weights.get('accuracy_preservation', 0.20)
            )
            
            e['score'] = final_score * 100.0  # Scale to 0-100 percentage
            e['simulation']['accuracy_degradation_note'] = f"Estimated {acc_penalty*100:.0f}% accuracy degradation applied."
            
        # Sort from highest to lowest score
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        return evaluations

    def get_best_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        ranked = self.evaluate(*args, **kwargs)
        return ranked[0] if ranked else {}
