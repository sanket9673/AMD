import os
import json
import logging
from typing import Dict, Any

from core.model_profiler import ModelProfiler
from core.optimizer import StrategyGenerator
from core.performance_simulator import PerformanceSimulator
from core.scorer import Scorer
from core.reasoning import ReasoningEngine
from core.hardware import HARDWARE_DATABASE

logger = logging.getLogger(__name__)


class DeploymentPipeline:
    def __init__(self):
        self.profiler = ModelProfiler(output_dir="data")
        self.strategy_gen = StrategyGenerator()
        self.simulator = PerformanceSimulator()
        self.scorer = Scorer()
        os.makedirs("results", exist_ok=True)

    def run_pipeline(
        self,
        model_id: str,
        hardware_type: str = None,
        workload_type: str = "chat_inference",
        weights: dict = None,
        constraints: dict = None,
        llm_mode: str = "llama-3.1-8b-instant",
        use_llm_reasoning: bool = False,
        config: Any = None,
        status: str = "ready",
        actual_model_id: str = None
    ) -> Dict[str, Any]:
        try:
            self.profiler.load_model(model_id, config=config, status=status, actual_model_id=actual_model_id)
            model_profile = self.profiler.generate_profile()

            hardware_profiles = [HARDWARE_DATABASE.get("AMD_MI250"), HARDWARE_DATABASE.get("AMD_MI300X")]
            hardware_profiles = [hw for hw in hardware_profiles if hw is not None]
            if hardware_type and hardware_type != "None":
                hw = HARDWARE_DATABASE.get(hardware_type)
                if hw:
                    hardware_profiles = [hw]

            all_strats = self.strategy_gen.generate_all_combinations()

            evaluations = []
            for hw in hardware_profiles:
                for strat in all_strats:
                    try:
                        sim_result = self.simulator.simulate(model_profile, strat, hw, workload_type=workload_type)
                        acc_penalty = 0.0
                        if strat.precision == "int4":
                            acc_penalty += 0.35
                        elif strat.precision == "int8":
                            acc_penalty += 0.10
                        if strat.prune_ratio > 0.6:
                            acc_penalty += 0.50
                        elif strat.prune_ratio > 0.3:
                            acc_penalty += 0.25
                        elif strat.prune_ratio > 0.1:
                            acc_penalty += 0.05
                        acc_penalty = min(acc_penalty, 1.0)
                        sim_result["accuracy_penalty"] = acc_penalty
                        evaluations.append({
                            "strategy": strat.to_dict(),
                            "hardware": hw.name,
                            "simulation": sim_result
                        })
                    except Exception:
                        continue

            baseline_hw = hardware_profiles[0] if hardware_profiles else HARDWARE_DATABASE.get("AMD_MI250")
            baseline_strat = next(
                (s for s in all_strats if s.precision == "fp32" and s.prune_ratio == 0.0 and s.deployment_mode == "balanced"),
                all_strats[0]
            )
            baseline_eval = None
            try:
                sim_result = self.simulator.simulate(model_profile, baseline_strat, baseline_hw, workload_type=workload_type)
                sim_result["accuracy_penalty"] = 0.0
                baseline_eval = {
                    "strategy": baseline_strat.to_dict(),
                    "hardware": baseline_hw.name,
                    "simulation": sim_result
                }
            except Exception:
                pass

            if constraints:
                max_lat = constraints.get("max_latency", float('inf'))
                max_cost = constraints.get("max_cost", float('inf'))
                filtered_evals = [
                    ev for ev in evaluations
                    if ev['simulation']['latency_ms'] <= max_lat and ev['simulation']['cost_usd'] <= max_cost
                ]
                if filtered_evals:
                    evaluations = filtered_evals

            if not weights:
                weights = {
                    'latency': 0.2, 'memory_efficiency': 0.2, 'cost_efficiency': 0.2,
                    'energy_efficiency': 0.2, 'accuracy_preservation': 0.2
                }

            ranked_evaluations = self.scorer.evaluate(evaluations, weights)

            if not ranked_evaluations:
                raise ValueError("No viable strategies could be scored within the given constraints.")

            pareto_front = self.strategy_gen.pareto_filter(ranked_evaluations)
            if len(pareto_front) >= 3:
                ranked_evaluations = pareto_front

            best_evaluation = ranked_evaluations[0]

            self.reasoner = ReasoningEngine(mode=llm_mode)
            telem = best_evaluation["simulation"].get("roofline_telemetry", {})
            strategy_metrics = {
                "Precision": best_evaluation["strategy"]["precision"],
                "Prune Ratio": best_evaluation["strategy"]["prune_ratio"],
                "Hardware": best_evaluation["hardware"],
                "Latency (ms)": round(best_evaluation["simulation"]["latency_ms"], 1),
                "Cost ($)": round(best_evaluation["simulation"].get("cost_usd", 0), 5),
                "Prefill AI": round(telem.get("prefill_ai", 0.0), 2),
                "Ridge Point": round(telem.get("ridge", 0.0), 2)
            }

            reasoning = self.reasoner.generate_explanation(strategy_metrics, use_llm=use_llm_reasoning)
            best_evaluation["reasoning"] = reasoning

            final_output = {
                "original_request": model_id,
                "model_id": model_profile["model_id"],
                "status": model_profile.get("status", "unknown"),
                "baseline_evaluation": baseline_eval,
                "best_strategy": best_evaluation["strategy"],
                "best_evaluation": best_evaluation,
                "hardware_comparison": ranked_evaluations[:20],
                "scoring_breakdown": best_evaluation.get("scoring_breakdown", {}),
                "workload_type": workload_type,
            }
            final_output["scoring_breakdown"]["reasoning"] = reasoning

            results_path = os.path.join("results", "hardware_comparison.json")
            try:
                with open(results_path, "w") as f:
                    json.dump(final_output, f, indent=4)
            except IOError:
                pass

            return final_output

        except Exception as e:
            return {
                "error": str(e),
                "best_evaluation": None,
                "hardware_comparison": []
            }
