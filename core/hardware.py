from typing import Dict, Any

class HardwareProfile:
    def __init__(
        self,
        name: str,
        memory_gb: int,
        compute_score: float,
        bandwidth_gbps: float,
        power_watts: float,
        architecture_type: str,
        fp16_tflops: float = 0.0,
        int8_tops: float = 0.0,
        cost_per_hour: float = 0.0,
        bandwidth_tbps: float = 0.0
    ):
        self.name = name
        self.memory_gb = memory_gb
        self.compute_score = compute_score
        self.power_watts = power_watts
        self.architecture_type = architecture_type
        self.fp16_tflops = fp16_tflops
        self.int8_tops = int8_tops
        self.cost_per_hour = cost_per_hour

        if bandwidth_gbps > 0:
            self.bandwidth_gbps = bandwidth_gbps
            self.bandwidth_tbps = round(bandwidth_gbps / 1000.0, 4)
        elif bandwidth_tbps > 0:
            self.bandwidth_tbps = bandwidth_tbps
            self.bandwidth_gbps = bandwidth_tbps * 1000.0
        else:
            raise ValueError(
                f"HardwareProfile '{name}': at least one of bandwidth_gbps or bandwidth_tbps must be > 0"
            )

        if self.compute_score == 0 and self.fp16_tflops > 0:
            self.compute_score = self.fp16_tflops

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "memory_gb": self.memory_gb,
            "compute_score": self.compute_score,
            "bandwidth_gbps": self.bandwidth_gbps,
            "bandwidth_tbps": self.bandwidth_tbps,
            "power_watts": self.power_watts,
            "architecture_type": self.architecture_type,
            "fp16_tflops": self.fp16_tflops,
            "int8_tops": self.int8_tops,
            "cost_per_hour": self.cost_per_hour
        }

    def __repr__(self) -> str:
        return f"HardwareProfile(name={self.name}, architecture={self.architecture_type})"


HARDWARE_DATABASE: Dict[str, HardwareProfile] = {
    "AMD_MI250": HardwareProfile(
        name="AMD_MI250",
        memory_gb=128,
        compute_score=383.0,
        bandwidth_gbps=3200.0,
        bandwidth_tbps=3.2,
        power_watts=500.0,
        architecture_type="CDNA_2",
        fp16_tflops=383.0,
        int8_tops=766.0,
        cost_per_hour=3.5
    ),
    "AMD_MI300X": HardwareProfile(
        name="AMD_MI300X",
        memory_gb=192,
        compute_score=1307.0,
        bandwidth_gbps=5300.0,
        bandwidth_tbps=5.3,
        power_watts=750.0,
        architecture_type="CDNA_3",
        fp16_tflops=1307.0,
        int8_tops=2600.0,
        cost_per_hour=6.5
    )
}

def get_hardware_profile(name: str) -> HardwareProfile:
    if name in HARDWARE_DATABASE:
        return HARDWARE_DATABASE[name]
    raise ValueError(f"Hardware profile {name} not found.")
