from typing import List, Dict, Any, Optional

class StrategyScorer:
    """
    Evaluates and scores deployment strategies across multiple hardware profiles.
    Computes a weighted score and normalizes values to return the best strategy.
    """
    def __init__(self):
        # Default weights, can be dynamically overwritten
        self.weights = {
            'latency': 0.25,
            'memory_efficiency': 0.25,
            'energy_efficiency': 0.25,
            'cost_efficiency': 0.25
        }

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
            # allow dynamic weights passed from UI
            self.weights.update(weights)

        evaluations = []
        for hw in hardware_profiles:
            for strat in strategies:
                sim_result = simulator.simulate(model_profile, strat, hw, workload_type=workload_type)
                evaluations.append({
                    "strategy": strat,
                    "hardware": hw,
                    "simulation": sim_result
                })
        
        return self.score_and_rank(evaluations)
        
    def score_and_rank(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize all metrics before scoring and compute weighted final scores.
        """
        if not evaluations:
            return []
            
        # Extract min and max bounds for normalization
        max_latency = max(e['simulation']['latency_estimate'] for e in evaluations)
        min_latency = min(e['simulation']['latency_estimate'] for e in evaluations)
        
        max_energy = max(e['simulation']['energy_consumption'] for e in evaluations)
        min_energy = min(e['simulation']['energy_consumption'] for e in evaluations)
        
        max_cost = max(e['simulation'].get('cost_estimate', 0) for e in evaluations)
        min_cost = min(e['simulation'].get('cost_estimate', 0) for e in evaluations)
        
        latency_range = max(max_latency - min_latency, 1e-9)
        energy_range = max(max_energy - min_energy, 1e-9)
        cost_range = max(max_cost - min_cost, 1e-9)
        
        for e in evaluations:
            sim = e['simulation']
            
            # Latency (lower is better, 1.0 is min latency)
            norm_latency = 1.0 - ((sim['latency_estimate'] - min_latency) / latency_range)
            
            # Memory Efficiency (1.0 is max fit score)
            norm_memory = min(100.0, max(0.0, sim['hardware_fit_score'])) / 100.0
            
            # Energy Efficiency (lower is better)
            norm_energy = 1.0 - ((sim['energy_consumption'] - min_energy) / energy_range)

            # Cost Efficiency (lower is better)
            sim_cost = sim.get('cost_estimate', 0)
            norm_cost = 1.0 - ((sim_cost - min_cost) / cost_range)

            final_score = (
                norm_latency * self.weights.get('latency', 0.25) +
                norm_memory * self.weights.get('memory_efficiency', 0.25) +
                norm_energy * self.weights.get('energy_efficiency', 0.25) +
                norm_cost * self.weights.get('cost_efficiency', 0.25)
            )
            
            e['score'] = final_score * 100.0  # Scale to 0-100 percentage
            
        # Sort from highest to lowest score
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        return evaluations

    def get_best_strategy(self, 
                          model_profile: Dict[str, Any], 
                          strategies: List[Any], 
                          hardware_profiles: List[Any], 
                          simulator: Any,
                          weights: Dict[str, float] = None,
                          workload_type: str = "chat_inference") -> Dict[str, Any]:
        """
        Return the absolute best deployment strategy.
        """
        ranked = self.evaluate(model_profile, strategies, hardware_profiles, simulator, weights, workload_type)
        return ranked[0] if ranked else {}
