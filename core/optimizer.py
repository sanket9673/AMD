from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

@dataclass
class DeploymentStrategy:
    precision: str
    prune_ratio: float
    expected_memory_multiplier: float
    expected_latency_multiplier: float
    deployment_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StrategyGenerator:
    def __init__(self):
        self.precisions = ["fp32", "fp16", "int8", "int4"]
        self.prune_ratios = [0.0, 0.2, 0.4, 0.6]
        self.deployment_modes = ["balanced", "high_performance", "low_power"]

    def generate_all_combinations(self) -> List[DeploymentStrategy]:
        import itertools
        strategies = []
        for p, pr, dm in itertools.product(self.precisions, self.prune_ratios, self.deployment_modes):
            if p == "int4" and pr > 0.4:
                continue
            if p == "fp32" and dm == "low_power":
                continue
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

    def pareto_filter(self, evaluations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered = []
        for i, ev_A in enumerate(evaluations):
            dominated = False
            for j, ev_B in enumerate(evaluations):
                if i == j:
                    continue
                sim_A = ev_A['simulation']
                sim_B = ev_B['simulation']
                if (sim_B['latency_ms'] <= sim_A['latency_ms'] and
                        sim_B['memory_mb'] <= sim_A['memory_mb'] and
                        sim_B['cost_usd'] <= sim_A['cost_usd'] and
                        sim_B.get('accuracy_penalty', 0) <= sim_A.get('accuracy_penalty', 0)):
                    if (sim_B['latency_ms'] < sim_A['latency_ms'] or
                            sim_B['memory_mb'] < sim_A['memory_mb'] or
                            sim_B['cost_usd'] < sim_A['cost_usd'] or
                            sim_B.get('accuracy_penalty', 0) < sim_A.get('accuracy_penalty', 0)):
                        dominated = True
                        break
            if not dominated:
                filtered.append(ev_A)
        return filtered

    def search_best_strategies(self, evaluator_callback: Callable[[List[DeploymentStrategy]], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        all_combinations = self.generate_all_combinations()
        evaluations = evaluator_callback(all_combinations)
        pareto_front = self.pareto_filter(evaluations)
        if len(pareto_front) < 5:
            num_keep = max(5, len(evaluations) // 2)
            return evaluations[:num_keep]
        pareto_front.sort(key=lambda x: x.get('score', 0), reverse=True)
        return pareto_front

    def generate_strategies(self) -> List[DeploymentStrategy]:
        return self.generate_all_combinations()
