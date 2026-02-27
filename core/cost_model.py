class CostModel:
    """
    Computes infrastructure and energy costs for AI deployments.
    """
    def __init__(self):
        # We can store some base rates if needed
        pass

    def compute_energy_cost(self, power_watts: float, execution_time_seconds: float) -> float:
        """
        Computes the cost of energy consumed.
        Assuming an average cost of $0.12 per kWh.
        """
        kwh = (power_watts / 1000.0) * (execution_time_seconds / 3600.0)
        return kwh * 0.12

    def compute_infrastructure_cost(self, cost_per_hour: float, execution_time_seconds: float) -> float:
        """
        Computes the cost of reserving the hardware infra.
        """
        return cost_per_hour * (execution_time_seconds / 3600.0)

    def estimate_total_deployment_cost(self, nodes_required: int, execution_time: float, 
                                       power_watts: float = 0.0, cost_per_hour: float = 0.0) -> float:
        """
        Total estimated cost for a given time across N nodes.
        Note: The prompt specified exactly this signature minus the optional parameters, we include them for actual use.
        """
        energy_cost = self.compute_energy_cost(power_watts, execution_time)
        infra_cost = self.compute_infrastructure_cost(cost_per_hour, execution_time)
        return nodes_required * (energy_cost + infra_cost)
