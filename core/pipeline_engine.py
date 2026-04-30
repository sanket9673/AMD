import os
import json
import logging
import traceback
from typing import Dict, Any

from core.model_profiler import ModelProfiler
from core.optimizer import StrategyGenerator
from core.performance_simulator import PerformanceSimulator
from core.scorer import Scorer
from core.reasoning import ReasoningEngine
from core.hardware import HARDWARE_DATABASE

logger = logging.getLogger(__name__)

class DeploymentPipeline:
    """
    Main orchestration class handling the end-to-end intelligence cycle.
    Ensures ZERO crashes by wrapping executions in robust try-catch blocks.
    """
    def __init__(self):
        self.profiler = ModelProfiler(output_dir="data")
        self.strategy_gen = StrategyGenerator()
        self.simulator = PerformanceSimulator()
        self.scorer = Scorer()
        
        os.makedirs("results", exist_ok=True)
        
    def run_pipeline(self, 
                     model_id: str, 
                     hardware_type: str = None, 
                     workload_type: str = "chat_inference", 
                     weights: dict = None,
                     constraints: dict = None,
                     llm_mode: str = "FAST",
                     use_llm_reasoning: bool = False) -> Dict[str, Any]:
        """
        Runs the full pipeline robustly. Guarantees a valid return object.
        """
        logger.info(f"--- Starting Enterprise Deployment Intelligence Pipeline for {model_id} ---")
        
        try:
            # 1. Profile model (with gated model fallback inside profiler)
            logger.info("1. Profiling model (Mathematical bounds, NO weights loaded)...")
            self.profiler.load_model(model_id)
            model_profile = self.profiler.generate_profile()
            
            # 2. Setup Hardware Targets
            logger.info("2. Setting up hardware comparison...")
            hardware_profiles = [HARDWARE_DATABASE.get("AMD_MI250"), HARDWARE_DATABASE.get("AMD_MI300X")]
            hardware_profiles = [hw for hw in hardware_profiles if hw is not None]
            
            # 3. Generating Strategies
            logger.info("3. Generating and simulating strategies...")
            all_strats = self.strategy_gen.generate_all_combinations()
            
            evaluations = []
            for hw in hardware_profiles:
                for strat in all_strats:
                    try:
                        sim_result = self.simulator.simulate(model_profile, strat, hw, workload_type=workload_type)
                        # Estimate accuracy penalty
                        acc_penalty = 0.0
                        if strat.precision == "int4": acc_penalty += 0.35
                        elif strat.precision == "int8": acc_penalty += 0.10
                        if strat.prune_ratio > 0.6: acc_penalty += 0.50
                        elif strat.prune_ratio > 0.3: acc_penalty += 0.25
                        elif strat.prune_ratio > 0.1: acc_penalty += 0.05
                        acc_penalty = min(acc_penalty, 1.0)
                        
                        sim_result["accuracy_penalty"] = acc_penalty
                        
                        evaluations.append({
                            "strategy": strat.to_dict(),
                            "hardware": hw.name,
                            "simulation": sim_result
                        })
                    except Exception as e:
                        logger.warning(f"Simulation failed for {hw.name} with {strat.precision}: {e}")
                        continue
                        
            # Apply Constraints
            if constraints:
                max_lat = constraints.get("max_latency", float('inf'))
                max_cost = constraints.get("max_cost", float('inf'))
                
                filtered_evals = [
                    ev for ev in evaluations
                    if ev['simulation']['latency_ms'] <= max_lat and ev['simulation']['cost_usd'] <= max_cost
                ]
                if filtered_evals:
                    evaluations = filtered_evals
                else:
                    logger.warning("No strategies met constraints. Using all available to prevent crash.")

            # 4. Score and Rank (New Scorer)
            logger.info("4. Scoring and ranking strategies...")
            if not weights:
                weights = {'latency': 0.2, 'memory_efficiency': 0.2, 'cost_efficiency': 0.2, 'energy_efficiency': 0.2, 'accuracy_preservation': 0.2}
                
            ranked_evaluations = self.scorer.evaluate(evaluations, weights)
            
            if not ranked_evaluations:
                raise ValueError("No viable strategies could be scored!")
                
            best_evaluation = ranked_evaluations[0]
            
            # 5. Generate reasoning
            logger.info("5. Generating reasoning...")
            self.reasoner = ReasoningEngine(mode=llm_mode)
            
            strategy_metrics = {
                "Precision": best_evaluation["strategy"]["precision"],
                "Prune Ratio": best_evaluation["strategy"]["prune_ratio"],
                "Hardware": best_evaluation["hardware"],
                "Latency (ms)": round(best_evaluation["simulation"]["latency_ms"], 4),
                "Throughput (req/s)": round(best_evaluation["simulation"]["throughput"], 2),
                "Energy (kWh)": round(best_evaluation["simulation"]["energy_kwh"], 4),
                "Cost ($)": round(best_evaluation["simulation"].get("cost_usd", 0), 5),
                "Accuracy Penalty": round(best_evaluation["simulation"].get("accuracy_penalty", 0), 2)
            }
            
            reasoning = self.reasoner.generate_explanation(strategy_metrics, use_llm=use_llm_reasoning)
            best_evaluation["reasoning"] = reasoning
            
            # 6. Output
            logger.info("6. Preparing final output...")
            final_output = {
                "model_id": model_id,
                "best_strategy": best_evaluation["strategy"],
                "best_evaluation": best_evaluation,
                "hardware_comparison": ranked_evaluations[:20], # top 20 for charts
                "scoring_breakdown": best_evaluation.get("scoring_breakdown", {}),
                "workload_type": workload_type,
            }
            final_output["scoring_breakdown"]["reasoning"] = reasoning
                
            results_path = os.path.join("results", "hardware_comparison.json")
            try:
                with open(results_path, "w") as f:
                    json.dump(final_output, f, indent=4)
            except IOError as e:
                logger.warning(f"Could not save results to disk: {e}")
            
            logger.info(f"Pipeline execution completed successfully.")
            return final_output
            
        except Exception as e:
            logger.error(f"Critical Pipeline Failure: {e}\n{traceback.format_exc()}")
            # Return a safe fallback object to prevent UI crash
            return {
                "error": str(e),
                "traceback": traceback.format_exc(),
                "best_evaluation": None,
                "hardware_comparison": []
            }
