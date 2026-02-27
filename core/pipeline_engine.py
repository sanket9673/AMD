import os
import json
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from .model_profiler import ModelProfiler
from .strategy_generator import StrategyGenerator
from .performance_simulator import PerformanceSimulator
from .scoring_engine import StrategyScorer
from .reasoning_engine import ReasoningEngine
from .hardware_profiles import HARDWARE_DATABASE

class DeploymentPipeline:
    """
    Main orchestration class handling the end-to-end intelligence cycle.
    """
    def __init__(self):
        self.profiler = ModelProfiler(output_dir="data")
        self.strategy_gen = StrategyGenerator()
        self.simulator = PerformanceSimulator()
        self.scorer = StrategyScorer()
        self.reasoner = ReasoningEngine()
        
        os.makedirs("results", exist_ok=True)
        
    def run_pipeline(self, model_id: str, hardware_type: str = None, workload_type: str = "chat_inference", weights: dict = None) -> Dict[str, Any]:
        """
        Runs the full pipeline across AMD_MI250 and AMD_MI300X as required by Phase 5.
        """
        print(f"--- Starting Enterprise Deployment Intelligence Pipeline for {model_id} ---")
        
        # 1. Profile model
        print("1. Profiling model...")
        self.profiler.load_model(model_id)
        model_profile = self.profiler.generate_profile()
        
        # 2. Setup Hardware Targets (Phase 5)
        print("2. Setting up hardware comparison (MI250 vs MI300X)...")
        if "AMD_MI250" not in HARDWARE_DATABASE or "AMD_MI300X" not in HARDWARE_DATABASE:
            # Fallback if somehow not defined
            hardware_profiles = list(HARDWARE_DATABASE.values())[:2]
        else:
            hardware_profiles = [HARDWARE_DATABASE["AMD_MI250"], HARDWARE_DATABASE["AMD_MI300X"]]
        
        # 3. & 4. Adaptive Strategy Search (Phase 6)
        print("3. Generating and optimizing strategies adaptively...")
        
        def evaluate_strategies(strats):
            return self.scorer.evaluate(
                model_profile=model_profile,
                strategies=strats,
                hardware_profiles=hardware_profiles,
                simulator=self.simulator,
                weights=weights,
                workload_type=workload_type
            )
            
        ranked_evaluations = self.strategy_gen.search_best_strategies(evaluate_strategies)
        
        # 5. Select Best
        print("4. Selecting best strategy and comparing hardware...")
        if not ranked_evaluations:
            raise ValueError("No viable strategies found!")
            
        best_evaluation = ranked_evaluations[0]
        best_strategy = best_evaluation["strategy"]
        best_score = best_evaluation["score"]
        
        # Create hardware comparison matrix
        hardware_comparison = []
        for ev in ranked_evaluations[:10]: # Top 10 for comparison
            strat = ev["strategy"]
            hw = ev["hardware"]
            hardware_comparison.append({
                "hardware": hw.name,
                "strategy": strat.to_dict(),
                "latency_estimate": ev["simulation"]["latency_estimate"],
                "throughput_estimate": ev["simulation"]["throughput_estimate"],
                "energy_consumption": ev["simulation"]["energy_consumption"],
                "cost_estimate": ev["simulation"].get("cost_estimate", 0),
                "adjusted_memory": ev["simulation"].get("adjusted_memory", 0),
                "hardware_fit_score": ev["simulation"].get("hardware_fit_score", 0),
                "score": ev["score"]
            })
            
        # 6. Generate reasoning
        print("5. Generating reasoning...")
        strategy_metrics = {
            "Precision": getattr(best_strategy, "precision", "unknown"),
            "Prune Ratio": getattr(best_strategy, "prune_ratio", 0.0),
            "Mode": getattr(best_strategy, "deployment_mode", "unknown"),
            "Hardware": best_evaluation["hardware"].name,
            "Latency (s)": best_evaluation["simulation"]["latency_estimate"],
            "Throughput (req/s)": best_evaluation["simulation"]["throughput_estimate"],
            "Energy (W)": best_evaluation["simulation"]["energy_consumption"],
            "Cost": best_evaluation["simulation"].get("cost_estimate", 0),
            "Overall Score": best_score
        }
        
        reasoning = self.reasoner.generate_explanation(strategy_metrics)
        best_evaluation["reasoning"] = reasoning
        
        # 7. Output Structure (Phase 7)
        print("6. Saving results...")
        final_output = {
            "best_strategy": best_strategy.to_dict(),
            "hardware_comparison": hardware_comparison,
            "scoring_breakdown": {
                "score": best_evaluation["score"],
                "hardware_fit_score": best_evaluation["simulation"]["hardware_fit_score"],
                "reasoning": reasoning
            },
            "cost_analysis": {
                "per_inference_cost": best_evaluation["simulation"]["cost_estimate"],
                "recommended_hardware": best_evaluation["hardware"].name
            },
            "energy_analysis": {
                "energy_consumed_wh": best_evaluation["simulation"]["energy_consumption"]
            },
            "workload_type": workload_type
        }
            
        results_path = os.path.join("results", "hardware_comparison.json")
        with open(results_path, "w") as f:
            json.dump(final_output, f, indent=4)
            
        # 8. Generate comparison graphs
        print("7. Generating comparison graphs...")
        self.generate_comparison_graphs(ranked_evaluations)
        
        print(f"Pipeline execution completed successfully. Results saved to {results_path}")
        
        # also return an object that helps run_pipeline.py remain compatible
        final_output["model_id"] = model_id
        final_output["best_evaluation"] = {
            "strategy": best_strategy.to_dict(),
            "hardware": best_evaluation["hardware"].name,
            "simulation": best_evaluation["simulation"]
        }
        return final_output
        
    def generate_comparison_graphs(self, evaluations: List[Dict[str, Any]]) -> None:
        """
        Generates bar charts comparing the top strategies.
        """
        top_evals = evaluations[:10]
        
        labels = []
        latencies = []
        memories = []
        scores = []
        
        for i, ev in enumerate(top_evals):
            strat = ev["strategy"]
            hw = ev["hardware"].name
            prec = getattr(strat, "precision", "unk")
            prune = getattr(strat, "prune_ratio", 0.0)
            labels.append(f"#{i+1} {hw}\n{prec} (p={prune})")
            
            latencies.append(ev["simulation"]["latency_estimate"])
            memories.append(ev["simulation"]["adjusted_memory"])
            scores.append(ev["score"])
            
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 3, 1)
        plt.bar(labels, scores, color='skyblue')
        plt.title('Overall Score')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Score')
        
        plt.subplot(1, 3, 2)
        plt.bar(labels, latencies, color='salmon')
        plt.title('Latency Estimate')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Seconds')

        plt.subplot(1, 3, 3)
        plt.bar(labels, memories, color='lightgreen')
        plt.title('Memory Footprint')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('MB')
        
        plt.tight_layout()
        viz_path = os.path.join("results", "strategy_comparison.png")
        plt.savefig(viz_path)
        plt.close()
