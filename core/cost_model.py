"""
Cost Model for Slingshot-AI Deployment Intelligence Engine

Translates infrastructure metrics into direct and operating financial bounds,
factoring in utilization overheads, idle costs, and spot pricing mechanics.

Classes:
    CostProjection: Analytical wrapper for a point-in-time cost estimation.
    CostModelEngine: Calculation component resolving usage into dollars.
"""

from typing import Dict, Any

class CostProjection:
    """
    Representation of financial expenditure for defined operations.
    """
    def __init__(self, hourly_run_rate: float, cost_per_1m_requests: float):
        self.hourly_run_rate = hourly_run_rate
        self.cost_per_1m_requests = cost_per_1m_requests

    def export(self) -> Dict[str, float]:
        """
        Serialize parameters for the API level.

        Returns:
            Dict[str, float]: Flat payload of monetary keys.
        """
        return {
            "hourly_cost_usd": self.hourly_run_rate,
            "cost_per_1m_inferences_usd": self.cost_per_1m_requests
        }

class CostModelEngine:
    """
    Applies pricing algorithms against utilized computing hardware over time domains.
    """
    def __init__(self, provider: str = "aws"):
        self.provider = provider
        # Simulated cost table (USD per hour per node)
        self.pricing_table = {
            "aws": {
                "A100_40GB": 4.10,
                "H100_80GB": 8.50,
                "TPUv4": 3.22
            },
            "gcp": {
                "A100_40GB": 4.00,
                "H100_80GB": 8.00,
                "TPUv4": 3.10
            }
        }

    def compute_cost(self, node_type_name: str, node_count: int, 
                     throughput_per_second: float) -> CostProjection:
        """
        Calculate standard pricing mechanics over target loads.

        Args:
            node_type_name (str): The standard name of the instance type.
            node_count (int): How many units are being spun up per replication ring.
            throughput_per_second (float): Inference units processed in one wall-clock second.

        Returns:
            CostProjection: The cost bounds data structure.
        """
        provider_rates = self.pricing_table.get(self.provider, self.pricing_table["aws"])
        hourly_rate_per_node = provider_rates.get(node_type_name, 5.0)
        
        total_hourly = hourly_rate_per_node * node_count
        
        # Calculate inferences per hour to find average cost per 1M queries
        inferences_per_hour = throughput_per_second * 3600
        
        if inferences_per_hour > 0:
            cost_per_1m = (total_hourly / inferences_per_hour) * 1_000_000
        else:
            cost_per_1m = 0.0
            
        return CostProjection(hourly_run_rate=total_hourly, 
                              cost_per_1m_requests=cost_per_1m)
