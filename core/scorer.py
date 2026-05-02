import logging
from typing import List, Dict, Any
from utils.normalization import clamp

logger = logging.getLogger(__name__)


class Scorer:
    def __init__(self):
        pass

    def evaluate(self, evaluations: List[Dict[str, Any]], weights: Dict[str, float]) -> List[Dict[str, Any]]:
        if not evaluations:
            return []

        total_weight = sum(weights.values())
        if total_weight <= 0:
            logger.warning("All weights are 0 or negative. Auto-assigning equal weights.")
            num_metrics = len(weights) if len(weights) > 0 else 5
            weights = {k: 1.0 / num_metrics for k in weights} if len(weights) > 0 else {
                'latency': 0.2, 'memory_efficiency': 0.2, 'cost_efficiency': 0.2,
                'energy_efficiency': 0.2, 'accuracy_preservation': 0.2
            }
            total_weight = 1.0
        else:
            weights = {k: v / total_weight for k, v in weights.items()}

        for e in evaluations:
            sim = e['simulation']
            lat = sim.get('latency_ms', 1000.0)
            mem = sim.get('memory_mb', 16000.0)
            cost = sim.get('cost_usd', 1.0)
            eng = sim.get('energy_kwh', 1.0)
            acc_pen = sim.get('accuracy_penalty', 0.0)

            lat_score = clamp(1.0 - (lat / 500.0), 0.0, 1.0)
            mem_score = clamp(1.0 - (mem / 16000.0), 0.0, 1.0)
            cost_score = clamp(1.0 - (cost / 10.0), 0.0, 1.0)
            eng_score = clamp(1.0 - (eng / 5.0), 0.0, 1.0)
            acc_score = clamp(1.0 - acc_pen, 0.0, 1.0)

            weighted_sum = (
                lat_score * weights.get('latency', 0.2) +
                mem_score * weights.get('memory_efficiency', 0.2) +
                cost_score * weights.get('cost_efficiency', 0.2) +
                eng_score * weights.get('energy_efficiency', 0.2) +
                acc_score * weights.get('accuracy_preservation', 0.2)
            )

            logger.debug(f"Metrics: lat={lat}, mem={mem}, cost={cost}, eng={eng}, acc_pen={acc_pen}")
            logger.debug(f"Scores: lat={lat_score}, mem={mem_score}, cost={cost_score}, eng={eng_score}, acc={acc_score}")

            e['score'] = max(0.0, min(weighted_sum * 100.0, 100.0))
            e['scoring_breakdown'] = {
                "efficiency_score": e['score'],
                "latency_score": lat_score * 100.0,
                "cost_score": cost_score * 100.0,
                "energy_score": eng_score * 100.0,
                "accuracy_score": acc_score * 100.0,
                "memory_score": mem_score * 100.0
            }

        evaluations.sort(key=lambda x: x['score'], reverse=True)
        return evaluations
