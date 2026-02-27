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
        
        # Ensure output directories exist
        os.makedirs("results", exist_ok=True)
        
    def run_pipeline(self, model_id: str, hardware_type: str) -> Dict[str, Any]:
        """
        Runs the full pipeline.
        
        Args:
            model_id: The identifier for the model (e.g. HuggingFace ID).
            hardware_type: Key in the HARDWARE_DATABASE for target hardware.
            
        Returns:
            A dictionary containing the pipeline results.
        """
        print(f"--- Starting Deployment Pipeline for {model_id} on {hardware_type} ---")
        
        # 1. Profile model
        print("1. Profiling model...")
        self.profiler.load_model(model_id)
        model_profile = self.profiler.generate_profile()
        
        # 2. Generate strategies
        print("2. Generating strategies...")
        strategies = self.strategy_gen.generate_strategies()
        
        # 3. Simulate performance on selected hardware
        print("3. Simulating performance...")
        if hardware_type not in HARDWARE_DATABASE:
            raise ValueError(f"Unknown hardware type: {hardware_type}")
            
        hardware = HARDWARE_DATABASE[hardware_type]
        hardware_profiles = [hardware]
        
        # 4. Score strategies
        print("4. Scoring strategies...")
        ranked_evaluations = self.scorer.evaluate(
            model_profile=model_profile,
            strategies=strategies,
            hardware_profiles=hardware_profiles,
            simulator=self.simulator
        )
        
        # 5. Select best
        print("5. Selecting best strategy...")
        if not ranked_evaluations:
            raise ValueError("No viable strategies found!")
            
        best_evaluation = ranked_evaluations[0]
        best_strategy = best_evaluation["strategy"]
        best_score = best_evaluation["score"]
        
        # 6. Generate reasoning
        print("6. Generating reasoning...")
        strategy_metrics = {
            "Precision": getattr(best_strategy, "precision", "unknown"),
            "Prune Ratio": getattr(best_strategy, "prune_ratio", 0.0),
            "Mode": getattr(best_strategy, "deployment_mode", "unknown"),
            "Hardware": hardware.name,
            "Latency (s)": best_evaluation["simulation"]["latency_estimate"],
            "Throughput (req/s)": best_evaluation["simulation"]["throughput_estimate"],
            "Energy (W)": best_evaluation["simulation"]["energy_consumption"],
            "Memory Fit Score": best_evaluation["simulation"]["hardware_fit_score"],
            "Overall Score": best_score
        }
        
        reasoning = self.reasoner.generate_explanation(strategy_metrics)
        best_evaluation["reasoning"] = reasoning
        
        # 7. Save results JSON
        print("7. Saving results...")
        final_output = []
        for eval_idx, ev in enumerate(ranked_evaluations):
            strat = ev["strategy"]
            hw = ev["hardware"]
            ev_dict = {
                "rank": eval_idx + 1,
                "score": ev["score"],
                "strategy": {
                    "precision": getattr(strat, "precision", ""),
                    "prune_ratio": getattr(strat, "prune_ratio", 0.0),
                    "deployment_mode": getattr(strat, "deployment_mode", "")
                },
                "hardware": hw.name,
                "simulation": ev["simulation"]
            }
            if "reasoning" in ev:
                ev_dict["reasoning"] = ev["reasoning"]
            final_output.append(ev_dict)
            
        results_path = os.path.join("results", "final_output.json")
        with open(results_path, "w") as f:
            json.dump({
                "model_id": model_id,
                "target_hardware": hardware.name,
                "ranked_strategies": final_output
            }, f, indent=4)
            
        # 8. Generate comparison graphs
        print("8. Generating comparison graphs...")
        self.generate_comparison_graphs(ranked_evaluations)
        
        print(f"Pipeline execution completed successfully. Results saved to {results_path}")
        return {
            "model_id": model_id,
            "target_hardware": hardware.name,
            "best_evaluation": final_output[0],
            "results_path": results_path
        }
        
    def generate_comparison_graphs(self, evaluations: List[Dict[str, Any]]) -> None:
        """
        Generates bar charts comparing the top strategies for latency, memory, and score.
        Saves the resulting plot as a PNG.
        """
        # Take up to top 10 for visualization to avoid crowding
        top_evals = evaluations[:10]
        
        labels = []
        latencies = []
        memories = []
        scores = []
        
        for i, ev in enumerate(top_evals):
            strat = ev["strategy"]
            prec = getattr(strat, "precision", "unk")
            prune = getattr(strat, "prune_ratio", 0.0)
            labels.append(f"#{i+1} {prec} (p={prune})")
            
            latencies.append(ev["simulation"]["latency_estimate"])
            memories.append(ev["simulation"]["adjusted_memory"])
            scores.append(ev["score"])
            
        plt.figure(figsize=(15, 6))
        
        # Plot 1: Scores (Higher is better)
        plt.subplot(1, 3, 1)
        plt.bar(labels, scores, color='skyblue')
        plt.title('Overall Score (Higher is Better)')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Score')
        
        # Plot 2: Latencies (Lower is better)
        plt.subplot(1, 3, 2)
        plt.bar(labels, latencies, color='salmon')
        plt.title('Latency Estimate (Lower is Better)')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('Seconds')

        # Plot 3: Projected Memory (Lower is better)
        plt.subplot(1, 3, 3)
        plt.bar(labels, memories, color='lightgreen')
        plt.title('Memory Footprint (Lower is Better)')
        plt.xticks(rotation=45, ha="right")
        plt.ylabel('MB')
        
        plt.tight_layout()
        viz_path = os.path.join("results", "strategy_comparison.png")
        plt.savefig(viz_path)
        plt.close()
        print(f"Comparison graphs saved to {viz_path}")
