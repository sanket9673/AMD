# run_pipeline.py

from core.pipeline_engine import DeploymentPipeline

def main():
    print("🚀 Starting Slingshot-AI Deployment Intelligence Pipeline...\n")

    pipeline = DeploymentPipeline()

    results = pipeline.run_pipeline(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        hardware_type="AMD_MI300X"
    )

    print("\n✅ Pipeline Completed Successfully.")
    print("🏆 Best Strategy Selected:")
    print(results["best_evaluation"]["strategy"])

if __name__ == "__main__":
    main()