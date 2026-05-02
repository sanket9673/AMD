import unittest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.performance_simulator import PerformanceSimulator
from core.hardware import HARDWARE_DATABASE
from core.optimizer import DeploymentStrategy


class TestSimulator(unittest.TestCase):

    def setUp(self):
        self.simulator = PerformanceSimulator()
        self.mock_profile = {
            "total_parameters": 7_000_000_000,
            "hidden_size": 4096,
            "model_depth": 32
        }
        self.mi250 = HARDWARE_DATABASE["AMD_MI250"]
        self.mi300x = HARDWARE_DATABASE["AMD_MI300X"]

    def test_mi300x_vs_mi250(self):
        strat = DeploymentStrategy("fp16", 0.0, 0.5, 0.6, "balanced")
        sim_250 = self.simulator.simulate(self.mock_profile, strat, self.mi250)
        sim_300 = self.simulator.simulate(self.mock_profile, strat, self.mi300x)
        self.assertLess(sim_300["latency_ms"], sim_250["latency_ms"],
                        "MI300X should have lower latency than MI250")

    def test_pruning_reduces_memory(self):
        strat_no_prune = DeploymentStrategy("fp16", 0.0, 0.5, 0.6, "balanced")
        strat_prune = DeploymentStrategy("fp16", 0.5, 0.25, 0.3, "balanced")
        sim_no = self.simulator.simulate(self.mock_profile, strat_no_prune, self.mi300x)
        sim_yes = self.simulator.simulate(self.mock_profile, strat_prune, self.mi300x)
        self.assertLess(sim_yes["memory_mb"], sim_no["memory_mb"],
                        "Pruned model should use less memory")

    def test_int4_reduces_memory_vs_fp16(self):
        strat_fp16 = DeploymentStrategy("fp16", 0.0, 0.5, 0.6, "balanced")
        strat_int4 = DeploymentStrategy("int4", 0.0, 0.125, 0.3, "balanced")
        sim_fp16 = self.simulator.simulate(self.mock_profile, strat_fp16, self.mi300x)
        sim_int4 = self.simulator.simulate(self.mock_profile, strat_int4, self.mi300x)
        self.assertLess(sim_int4["memory_mb"], sim_fp16["memory_mb"],
                        "INT4 should use less memory than FP16")

    def test_pareto_filter_removes_dominated_strategy(self):
        from core.optimizer import StrategyGenerator
        gen = StrategyGenerator()
        ev_A = {"simulation": {"latency_ms": 100.0, "memory_mb": 2000.0, "cost_usd": 0.05, "accuracy_penalty": 0.2}}
        ev_B = {"simulation": {"latency_ms": 50.0, "memory_mb": 1000.0, "cost_usd": 0.01, "accuracy_penalty": 0.1}}
        filtered = gen.pareto_filter([ev_A, ev_B])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0], ev_B)

    def test_dummy_config_fallback(self):
        from core.model_profiler import ModelProfiler
        import transformers
        original = transformers.AutoConfig.from_pretrained
        def raise_exc(*args, **kwargs):
            raise Exception("Force fallback")
        transformers.AutoConfig.from_pretrained = raise_exc
        try:
            profiler = ModelProfiler()
            profiler.load_model("invalid_model_that_does_not_exist")
            profile = profiler.generate_profile()
            self.assertIn("model_id", profile)
            self.assertIn("hidden_size", profile)
            self.assertIn("model_depth", profile)
            self.assertIn("total_parameters", profile)
        finally:
            transformers.AutoConfig.from_pretrained = original

    def test_constraint_filter_empty_result(self):
        from core.pipeline_engine import DeploymentPipeline
        pipeline = DeploymentPipeline()
        res = pipeline.run_pipeline(
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            constraints={"max_latency": 0.001, "max_cost": 0.000001}
        )
        self.assertIn("error", res)

    def test_cost_model_zero_power(self):
        from core.cost_model import CostModel
        cm = CostModel()
        result = cm.compute_energy_cost(power_watts=0.0, execution_time_seconds=3600.0)
        self.assertEqual(result, 0.0)

    def test_scorer_zero_weights(self):
        from core.scorer import Scorer
        scorer = Scorer()
        evals = [{
            "simulation": {
                "latency_ms": 100.0,
                "memory_mb": 2000.0,
                "cost_usd": 0.05,
                "energy_kwh": 0.1,
                "accuracy_penalty": 0.1
            }
        }]
        weights = {
            'latency': 0,
            'memory_efficiency': 0,
            'cost_efficiency': 0,
            'energy_efficiency': 0,
            'accuracy_preservation': 0
        }
        result = scorer.evaluate(evals, weights)
        self.assertEqual(len(result), 1)
        self.assertIn("score", result[0])


if __name__ == '__main__':
    unittest.main()
