import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.performance_simulator import PerformanceSimulator
from core.hardware_profiles import HARDWARE_DATABASE
from core.strategy_generator import DeploymentStrategy

class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.simulator = PerformanceSimulator()
        self.mock_profile = {
            "total_parameters": 7_000_000_000, # 7B model
            "hidden_size": 4096,
            "model_depth": 32
        }
        self.mi250 = HARDWARE_DATABASE["AMD_MI250"]
        self.mi300x = HARDWARE_DATABASE["AMD_MI300X"]

    def test_mi300x_vs_mi250(self):
        """Test that MI300X is faster than MI250 for the same strategy (Roofline validity)"""
        strat = DeploymentStrategy("fp16", 0.0, 0.5, 0.6, "balanced")
        
        sim_250 = self.simulator.simulate(self.mock_profile, strat, self.mi250)
        sim_300 = self.simulator.simulate(self.mock_profile, strat, self.mi300x)
        
        self.assertLess(sim_300["latency_estimate"], sim_250["latency_estimate"], "MI300X should have lower latency than MI250")

    def test_pruning_reduces_memory(self):
        """Test that higher pruning reduces memory footprint"""
        strat_no_prune = DeploymentStrategy("fp16", 0.0, 0.5, 0.6, "balanced")
        strat_prune = DeploymentStrategy("fp16", 0.5, 0.25, 0.3, "balanced")
        
        sim_no = self.simulator.simulate(self.mock_profile, strat_no_prune, self.mi300x)
        sim_yes = self.simulator.simulate(self.mock_profile, strat_prune, self.mi300x)
        
        self.assertLess(sim_yes["adjusted_memory"], sim_no["adjusted_memory"], "Pruned model should use less memory")

    def test_int4_reduces_memory_vs_fp16(self):
        """Test that INT4 quant reduces memory compared to FP16"""
        strat_fp16 = DeploymentStrategy("fp16", 0.0, 0.5, 0.6, "balanced")
        strat_int4 = DeploymentStrategy("int4", 0.0, 0.125, 0.3, "balanced")
        
        sim_fp16 = self.simulator.simulate(self.mock_profile, strat_fp16, self.mi300x)
        sim_int4 = self.simulator.simulate(self.mock_profile, strat_int4, self.mi300x)
        
        self.assertLess(sim_int4["adjusted_memory"], sim_fp16["adjusted_memory"], "INT4 should use less memory than FP16")

if __name__ == '__main__':
    unittest.main()
