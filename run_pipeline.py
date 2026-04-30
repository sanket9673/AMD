# run_pipeline.py

from core.pipeline_engine import DeploymentPipeline
import json

def main():
    print("🚀 Starting Slingshot-AI Deployment Intelligence Pipeline...\n")

    pipeline = DeploymentPipeline()

    try:
        results = pipeline.run_pipeline(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            hardware_type="AMD_MI300X",
            workload_type="chat_inference",
            weights={'latency': 0.25, 'memory_efficiency': 0.25, 'cost_efficiency': 0.25, 'energy_efficiency': 0.25, 'accuracy_preservation': 0.0},
            llm_mode="FAST",
            use_llm_reasoning=False
        )

        if "error" in results:
            print(f"❌ Pipeline failed: {results['error']}")
        else:
            print("\n✅ Pipeline Completed Successfully.")
            print("🏆 Best Strategy Selected:")
            print(json.dumps(results["best_strategy"], indent=2))
            print("\nScore Breakdown:")
            print(json.dumps(results["scoring_breakdown"], indent=2))
            
    except Exception as e:
        print(f"Failed to execute pipeline: {e}")

if __name__ == "__main__":
    main()