import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PerformanceSimulator:
    """
    Performance Simulator using a True Roofline Model.
    """
    def __init__(self):
        pass

    def simulate(self, profile: Dict[str, Any], strategy: Any, hardware: Any, workload_type: str = "chat_inference") -> Dict[str, Any]:
        """
        Compute performance metrics dynamically scaled by the true roofline model.
        """
        try:
            # Extract basic profile data
            params = profile.get("total_parameters", 1e9)
            
            # Extract strategy metrics
            if isinstance(strategy, dict):
                precision = strategy.get("precision", "fp16")
                prune_ratio = strategy.get("prune_ratio", 0.0)
                mode = strategy.get("deployment_mode", "balanced")
            else:
                precision = getattr(strategy, "precision", "fp16")
                prune_ratio = getattr(strategy, "prune_ratio", 0.0)
                mode = getattr(strategy, "deployment_mode", "balanced")
                
            # Extract hardware metrics
            if isinstance(hardware, dict):
                power_watts = hardware.get("power_watts", 500.0)
                memory_gb = hardware.get("memory_gb", 128.0)
                cost_per_hour = hardware.get("cost_per_hour", 3.0)
                compute_score = hardware.get("compute_score", 100.0) # TFLOPs
                bandwidth_gbps = hardware.get("bandwidth_gbps", 1000.0) # GB/s
            else:
                power_watts = getattr(hardware, "power_watts", 500.0)
                memory_gb = getattr(hardware, "memory_gb", 128.0)
                cost_per_hour = getattr(hardware, "cost_per_hour", 3.0)
                compute_score = getattr(hardware, "fp16_tflops", getattr(hardware, "compute_score", 100.0))
                bandwidth_gbps = getattr(hardware, "bandwidth_gbps", 1000.0)

            # Apply Deployment Mode modifiers
            if mode == "high_performance":
                power_watts *= 1.2
                compute_score *= 1.1
            elif mode == "low_power":
                power_watts *= 0.6
                compute_score *= 0.7

            # Workload sizing
            if workload_type == "chat_inference":
                batch_size = 1
                prefill_seq = 512
                decode_seq = 512
            elif workload_type == "batch_inference":
                batch_size = 32
                prefill_seq = 512
                decode_seq = 512
            elif workload_type == "fine_tuning":
                batch_size = 16
                prefill_seq = 2048
                decode_seq = 0
            else:
                batch_size = 1
                prefill_seq = 512
                decode_seq = 512

            # Determine precision bytes per param
            if precision == "fp32": bytes_per_param = 4.0
            elif precision == "fp16": bytes_per_param = 2.0
            elif precision == "int8": bytes_per_param = 1.0
            elif precision == "int4": bytes_per_param = 0.5
            else: bytes_per_param = 2.0
            
            active_params = params * (1.0 - prune_ratio)
            compute_params = params * (1.0 - (prune_ratio * 0.5))

            # ---------------------------------------------------------
            # TRUE ROOFLINE MODEL MATH
            # ---------------------------------------------------------
            bandwidth_bytes_per_sec = bandwidth_gbps * 1e9
            
            # Hardware TOPS typically scale with reduced precision
            if precision == "fp32": compute_flops_per_sec = (compute_score / 2.0) * 1e12
            elif precision == "int8": compute_flops_per_sec = (compute_score * 2.0) * 1e12
            elif precision == "int4": compute_flops_per_sec = (compute_score * 4.0) * 1e12
            else: compute_flops_per_sec = compute_score * 1e12

            hardware_ridge_point = compute_flops_per_sec / bandwidth_bytes_per_sec
            
            # -- PREFILL PHASE --
            prefill_flops = 2 * compute_params * prefill_seq * batch_size
            prefill_bytes_moved = (active_params * bytes_per_param) + (prefill_seq * batch_size * 2 * 2 * profile.get("hidden_size", 4096))
            
            if prefill_bytes_moved == 0: prefill_bytes_moved = 1e-9
            prefill_arithmetic_intensity = prefill_flops / prefill_bytes_moved
            
            prefill_bandwidth_limit = bandwidth_bytes_per_sec * prefill_arithmetic_intensity
            prefill_compute_limit = compute_flops_per_sec
            prefill_performance = min(prefill_bandwidth_limit, prefill_compute_limit)
            
            prefill_latency = prefill_flops / prefill_performance

            # -- DECODE PHASE --
            decode_latency = 0.0
            decode_arithmetic_intensity = 0.0
            decode_performance = 0.0
            
            if decode_seq > 0:
                decode_flops_per_step = 2 * compute_params * 1 * batch_size
                decode_bytes_moved_per_step = (active_params * bytes_per_param) + (batch_size * 2 * 2 * profile.get("hidden_size", 4096))
                
                decode_arithmetic_intensity = decode_flops_per_step / decode_bytes_moved_per_step
                
                decode_bandwidth_limit = bandwidth_bytes_per_sec * decode_arithmetic_intensity
                decode_compute_limit = compute_flops_per_sec
                decode_performance = min(decode_bandwidth_limit, decode_compute_limit)
                
                decode_latency_per_step = decode_flops_per_step / decode_performance
                decode_latency = decode_latency_per_step * decode_seq

            # Total Latency
            total_latency_sec = max(prefill_latency + decode_latency, 1e-6)
            latency_ms = total_latency_sec * 1000.0
            
            # Memory Model (Static + KV Cache)
            model_memory_bytes = active_params * bytes_per_param
            kv_cache_bytes = batch_size * (prefill_seq + decode_seq) * 2 * 2 * profile.get("hidden_size", 4096) * getattr(profile, "model_depth", 32)
            memory_mb = (model_memory_bytes + kv_cache_bytes) / (1024**2)
            
            # Cost and Energy
            energy_kwh = (power_watts * total_latency_sec) / 3600000.0
            cost_usd = cost_per_hour * (total_latency_sec / 3600.0)
            
            throughput = batch_size / total_latency_sec
            
            hardware_memory_mb = memory_gb * 1024.0
            hardware_fit_score = 100.0
            if memory_mb > hardware_memory_mb:
                hardware_fit_score = (hardware_memory_mb / memory_mb) * 50.0
                
            return {
                "memory_mb": memory_mb,
                "latency_ms": latency_ms,
                "throughput": throughput,
                "energy_kwh": energy_kwh,
                "hardware_fit_score": hardware_fit_score,
                "cost_usd": cost_usd,
                "roofline_info": f"AI: {prefill_arithmetic_intensity:.2f} (P), {decode_arithmetic_intensity:.2f} (D) | Ridge: {hardware_ridge_point:.2f}",
                # Additional telemetry for UI plotting
                "roofline_telemetry": {
                    "bandwidth": bandwidth_bytes_per_sec,
                    "compute": compute_flops_per_sec,
                    "ridge": hardware_ridge_point,
                    "prefill_ai": prefill_arithmetic_intensity,
                    "prefill_perf": prefill_performance,
                    "decode_ai": decode_arithmetic_intensity,
                    "decode_perf": decode_performance
                }
            }
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                "memory_mb": 1000.0,
                "latency_ms": 1000.0,
                "throughput": 1.0,
                "energy_kwh": 0.0001,
                "hardware_fit_score": 0.0,
                "cost_usd": 0.1,
                "roofline_info": "Failed",
                "roofline_telemetry": None
            }
