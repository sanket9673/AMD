from typing import List, Dict, Any

class StrategyScorer:
    """
    Evaluates and scores deployment strategies across multiple hardware profiles.
    Computes a weighted score and normalizes values to return the best strategy.
    """
    def __init__(self):
        # Weighted scoring requirements
        self.weights = {
            'latency': 0.35,             # 35%
            'memory_fit': 0.25,          # 25%
            'energy_efficiency': 0.20,   # 20%
            'throughput': 0.10,          # 10%
            'deployment_mode': 0.10      # 10%
        }
        
        # Priority mapping for deployment modes
        self.mode_priorities = {
            'high_performance': 1.0,
            'balanced': 0.8,
            'low_power': 0.6
        }

    def evaluate(self, 
                 model_profile: Dict[str, Any], 
                 strategies: List[Any], 
                 hardware_profiles: List[Any], 
                 simulator: Any) -> List[Dict[str, Any]]:
        """
        Simulate all combinations of strategies and hardware profiles, score them,
        and return the ranked results.
        
        Args:
            model_profile: Dict containing model details (e.g. estimated_memory_mb, estimated_flops)
            strategies: List of DeploymentStrategy objects
            hardware_profiles: List of HardwareProfile objects
            simulator: PerformanceSimulator instance
            
        Returns:
            List of evaluated options sorted by score descending.
        """
        evaluations = []
        for hw in hardware_profiles:
            for strat in strategies:
                sim_result = simulator.simulate(model_profile, strat, hw)
                evaluations.append({
                    "strategy": strat,
                    "hardware": hw,
                    "simulation": sim_result
                })
        
        return self.score_and_rank(evaluations)
        
    def score_and_rank(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize values across all evaluations and compute weighted final scores.
        """
        if not evaluations:
            return []
            
        # Extract min and max bounds for normalization
        max_latency = max(e['simulation']['latency_estimate'] for e in evaluations)
        min_latency = min(e['simulation']['latency_estimate'] for e in evaluations)
        
        max_energy = max(e['simulation']['energy_consumption'] for e in evaluations)
        min_energy = min(e['simulation']['energy_consumption'] for e in evaluations)
        
        max_throughput = max(e['simulation']['throughput_estimate'] for e in evaluations)
        min_throughput = min(e['simulation']['throughput_estimate'] for e in evaluations)
        
        latency_range = max(max_latency - min_latency, 1e-9)
        energy_range = max(max_energy - min_energy, 1e-9)
        throughput_range = max(max_throughput - min_throughput, 1e-9)
        
        for e in evaluations:
            sim = e['simulation']
            strat = e['strategy']
            
            # 1. Latency (35%) - lower is better
            if max_latency == min_latency:
                norm_latency = 1.0
            else:
                norm_latency = 1.0 - ((sim['latency_estimate'] - min_latency) / latency_range)
                
            # 2. Memory fit (25%) - from 0-100 directly from simulator, convert to 0-1.0
            norm_memory = min(100.0, max(0.0, sim['hardware_fit_score'])) / 100.0
            
            # 3. Energy efficiency (20%) - lower is better
            if max_energy == min_energy:
                norm_energy = 1.0
            else:
                norm_energy = 1.0 - ((sim['energy_consumption'] - min_energy) / energy_range)
                
            # 4. Throughput (10%) - higher is better
            if max_throughput == min_throughput:
                norm_throughput = 1.0
            else:
                norm_throughput = (sim['throughput_estimate'] - min_throughput) / throughput_range
            
            # 5. Deployment_mode priority (10%)
            mode = getattr(strat, 'deployment_mode', 'balanced') if not isinstance(strat, dict) else strat.get('deployment_mode', 'balanced')
            norm_mode = self.mode_priorities.get(mode, 0.5)
            
            final_score = (
                norm_latency * self.weights['latency'] +
                norm_memory * self.weights['memory_fit'] +
                norm_energy * self.weights['energy_efficiency'] +
                norm_throughput * self.weights['throughput'] +
                norm_mode * self.weights['deployment_mode']
            )
            
            e['score'] = final_score * 100.0  # Scale to 0-100 percentage
            
        # Sort from highest to lowest score
        evaluations.sort(key=lambda x: x['score'], reverse=True)
        return evaluations

    def get_best_strategy(self, 
                          model_profile: Dict[str, Any], 
                          strategies: List[Any], 
                          hardware_profiles: List[Any], 
                          simulator: Any) -> Dict[str, Any]:
        """
        Return the absolute best deployment strategy across multiple hardware profiles.
        """
        ranked = self.evaluate(model_profile, strategies, hardware_profiles, simulator)
        return ranked[0] if ranked else {}
