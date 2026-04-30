# 🎯 Slingshot-AI
**Enterprise Deployment Intelligence System for AI Inference**

Slingshot-AI is a production-grade infrastructure intelligence platform that programmatically determines the optimal hardware and software strategies for deploying Large Language Models (LLMs). It uses mathematically rigorous Roofline performance modeling to solve the multi-objective optimization problem between latency, cost, energy, and accuracy.

---

## 1. Problem Statement
Deploying LLMs into production is fundamentally difficult because of the massive search space of optimizations. 
Infrastructure engineers must balance competing constraints:
- Should we use `int8` or `int4` quantization?
- Will unstructured pruning at 40% actually speed up the model, or just degrade accuracy?
- Is it cheaper to run a memory-bound decode phase on an AMD MI250 or an AMD MI300X?

Guessing the answers to these questions in production leads to **OOM crashes, massive cloud bills, and missed latency SLAs.**

---

## 2. Solution
Slingshot-AI eliminates the guesswork. Instead of running expensive, time-consuming grid searches on physical hardware, Slingshot-AI simulates the entire hardware-software interaction using **Roofline Performance Modeling**. 

It automatically calculates the exact memory bandwidth limit and compute limit of different target hardware, evaluates thousands of permutations of precision and pruning combinations, and outputs the mathematically proven Pareto-optimal deployment strategy.

---

## 3. Features
- **Multi-Objective Optimization**: Ranks strategies using a customized weighted sum across Latency, Memory, Cost, Energy, and Accuracy constraints.
- **True Roofline Modeling**: Simulates prefill (compute-bound) and decode (memory-bound) phases independently based on arithmetic intensity (FLOPs/byte) and hardware limits (ridge points).
- **Hardware-Aware Decisions**: Built-in AMD MI250 and MI300X hardware profiles containing exact memory capacity, memory bandwidth, peak compute, and TDP constraints.
- **LLM Reasoning Engine**: Integrates asynchronously with Groq's high-speed inference endpoints to generate dynamic, human-readable explanations of *why* the strategy was selected.

---

## 4. Architecture
The system is divided into modular, highly-cohesive components:

* `core/pipeline_engine.py`: The robust orchestrator that drives the intelligence cycle and safely captures failures.
* `core/model_profiler.py`: Analyzes the LLM architecture using `AutoConfig` to determine parameter counts and sequence constraints without loading the full weights into RAM.
* `core/performance_simulator.py`: Implements the core Roofline math to calculate precise latency, energy, and memory throughput limits.
* `core/optimizer.py`: Generates all permutations of quantization and pruning strategies.
* `core/scorer.py`: Applies Min-Max normalization to standard bounds and generates an `efficiency_score` from `[0, 100]`.
* `core/hardware.py`: Encapsulates the AMD device profiles (TDP, TFLOPS, Bandwidth).
* `core/reasoning.py`: Generates the automated LLM insights.
* `ui/dashboard.py`: A fully reactive, YC-level Streamlit interface displaying Executive Decisions and Data Visualizations.
* `utils/plotting.py`: Generates Log-scale Roofline models, Radar charts, and Pareto Frontiers using Plotly.

---

## 5. Installation

To install Slingshot-AI, we recommend using a Conda environment:

```bash
conda create -n slingshot python=3.10
conda activate slingshot
pip install -r requirements.txt
```

---

## 6. Run

To execute the pipeline via CLI and generate the final output JSON:
```bash
python run_pipeline.py
```

To boot up the interactive executive dashboard:
```bash
streamlit run ui/dashboard.py
```

*(Optional)* To enable LLM insights, set your Groq API key:
```bash
export GROQ_API_KEY="your-api-key"
```

---

## 7. Example Output
Slingshot-AI produces a rich evaluation dictionary. Key outputs include:
- `latency_ms`: Computed total execution time based on the Roofline limits.
- `energy_kwh`: Calculated directly from execution time and hardware Wattage (TDP).
- `cost_usd`: Estimated per-request inference cost.
- `accuracy_penalty`: A modeled penalty assigned for destructive compressions (e.g., int4 with heavy pruning).
- `roofline_telemetry`: Detailed breakdown of the Prefill and Decode arithmetic intensities.

---

## 8. Limitations
- **Accuracy Penalty Math**: Currently, the accuracy penalty from pruning/quantization is approximated via a static penalty step-function. Future iterations require mapping against real-world perplexity degradation datasets.
- **Static Batching**: The simulator currently defaults to static batch definitions. Dynamic Continuous Batching (vLLM style) alters arithmetic intensity over time and is not fully modeled.

---

## 9. Future Work
- Integrate direct PyTorch benchmarking to validate simulator estimations automatically.
- Support multi-node deployment strategies (Tensor Parallelism & Pipeline Parallelism overheads).
- Expand hardware support to upcoming generation accelerators.
