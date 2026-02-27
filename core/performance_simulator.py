from typing import Dict, Any
from .cost_model import CostModel

class PerformanceSimulator:
    """
    Performance Simulator for AI Deployment Engine.
    Simulates performance of a model deployment strategy on a specific hardware.
    """
    def __init__(self):
        self.cost_model = CostModel()

    def simulate(self, profile: Dict[str, Any], strategy: Any, hardware: Any, workload_type: str = "chat_inference") -> Dict[str, Any]:
        """
        Compute performance metrics dynamically scaled by workload type.
        """
        estimated_memory_mb = profile.get("estimated_memory_mb", 1000.0)
        
        # Extract strategy metrics
        if isinstance(strategy, dict):
            expected_memory_multiplier = strategy.get("expected_memory_multiplier", 1.0)
            latency_multiplier = strategy.get("expected_latency_multiplier", 1.0)
            precision = strategy.get("precision", "fp16")
        else:
            expected_memory_multiplier = getattr(strategy, "expected_memory_multiplier", 1.0)
            latency_multiplier = getattr(strategy, "expected_latency_multiplier", 1.0)
            precision = getattr(strategy, "precision", "fp16")
            
        # Extract hardware metrics
        if isinstance(hardware, dict):
            power_watts = hardware.get("power_watts", 1.0)
            memory_gb = hardware.get("memory_gb", 1.0)
            cost_per_hour = hardware.get("cost_per_hour", 1.0)
            fp16_tflops = hardware.get("fp16_tflops", hardware.get("compute_score", 1.0))
            int8_tops = hardware.get("int8_tops", hardware.get("compute_score", 1.0))
        else:
            power_watts = getattr(hardware, "power_watts", 1.0)
            memory_gb = getattr(hardware, "memory_gb", 1.0)
            cost_per_hour = getattr(hardware, "cost_per_hour", 1.0)
            fp16_tflops = getattr(hardware, "fp16_tflops", getattr(hardware, "compute_score", 1.0))
            int8_tops = getattr(hardware, "int8_tops", getattr(hardware, "compute_score", 1.0))

        # Workload sizing
        if workload_type == "chat_inference":
            batch_size = 1
            sequence_length = 1024
        elif workload_type == "batch_inference":
            batch_size = 32
            sequence_length = 512
        elif workload_type == "fine_tuning":
            batch_size = 16
            sequence_length = 2048
        else:
            batch_size = 1
            sequence_length = 1024

        # Compute scaling based on precision
        if precision in ["int8", "int4"]:
            hardware_compute = int8_tops
        else:
            hardware_compute = fp16_tflops

        safe_compute_score = max(hardware_compute, 1e-9)
            
        # Mathematical modeling
        # Latency ∝ sequence_length × batch_size / hardware_compute
        latency_estimate = (sequence_length * batch_size / safe_compute_score) * latency_multiplier
        
        # Memory ∝ model_memory + activation_memory
        model_memory = estimated_memory_mb * expected_memory_multiplier
        activation_memory = (sequence_length * batch_size * 2) / 1024.0 # Estimate in MB
        adjusted_memory = model_memory + activation_memory
        
        # Energy ∝ power_watts × execution_time
        execution_time = latency_estimate
        energy_consumption = power_watts * execution_time
        
        # Cost ∝ cost_per_hour × execution_time
        cost_estimate = self.cost_model.estimate_total_deployment_cost(
            nodes_required=1, 
            execution_time=execution_time, 
            power_watts=power_watts, 
            cost_per_hour=cost_per_hour
        )
        
        # Derived metrics
        safe_latency = max(latency_estimate, 1e-9)
        throughput_estimate = 1.0 / safe_latency
        
        hardware_memory_mb = memory_gb * 1024.0
        # Fit score out of 100 based on memory fit
        if adjusted_memory > hardware_memory_mb:
            hardware_fit_score = (hardware_memory_mb / adjusted_memory) * 50.0  # Penalty for exceeding memory
        else:
            hardware_fit_score = 100.0
            
        # Structured performance dictionary
        return {
            "adjusted_memory": adjusted_memory,
            "latency_estimate": latency_estimate,
            "throughput_estimate": throughput_estimate,
            "energy_consumption": energy_consumption,
            "hardware_fit_score": hardware_fit_score,
            "cost_estimate": cost_estimate
        }
