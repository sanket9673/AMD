from typing import Dict, Any

class PerformanceSimulator:
    """
    Performance Simulator for AI Deployment Engine.
    Simulates performance of a model deployment strategy on a specific hardware.
    """
    def __init__(self):
        pass

    def simulate(self, profile: Dict[str, Any], strategy: Any, hardware: Any) -> Dict[str, Any]:
        """
        Compute performance metrics.
        
        Args:
            profile: Profile output from ModelProfiler (dict)
            strategy: DeploymentStrategy object or dict
            hardware: HardwareProfile object or dict
            
        Returns:
            Dict containing computed performance metrics.
        """
        # Extract profile metrics
        estimated_memory_mb = profile.get("estimated_memory_mb", 1000.0)
        base_flops = profile.get("estimated_flops", 1e9)
        
        # Extract strategy metrics
        if isinstance(strategy, dict):
            expected_memory_multiplier = strategy.get("expected_memory_multiplier", 1.0)
            latency_multiplier = strategy.get("expected_latency_multiplier", 1.0)
        else:
            expected_memory_multiplier = getattr(strategy, "expected_memory_multiplier", 1.0)
            latency_multiplier = getattr(strategy, "expected_latency_multiplier", 1.0)
            
        # Extract hardware metrics
        if isinstance(hardware, dict):
            compute_score = hardware.get("compute_score", 1.0)
            power_watts = hardware.get("power_watts", 1.0)
            memory_gb = hardware.get("memory_gb", 1.0)
        else:
            compute_score = getattr(hardware, "compute_score", 1.0)
            power_watts = getattr(hardware, "power_watts", 1.0)
            memory_gb = getattr(hardware, "memory_gb", 1.0)

        # Ensure safe values to avoid division by zero
        safe_compute_score = max(compute_score, 1e-9)
            
        # Mathematical modeling based on requirements
        adjusted_memory = estimated_memory_mb * expected_memory_multiplier
        latency_estimate = (base_flops / safe_compute_score) * latency_multiplier
        energy_consumption = latency_estimate * power_watts
        
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
        result = {
            "adjusted_memory": adjusted_memory,
            "latency_estimate": latency_estimate,
            "throughput_estimate": throughput_estimate,
            "energy_consumption": energy_consumption,
            "hardware_fit_score": hardware_fit_score
        }
        
        return result
