from typing import Dict, Any

class HardwareProfile:
    """
    Encapsulates the performance characteristics and constraints of an AMD target device.
    """
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
        """
        Initialize an AMD Hardware Profile.
        """
        self.name = name
        self.memory_gb = memory_gb
        self.compute_score = compute_score
        self.bandwidth_gbps = bandwidth_gbps
        self.power_watts = power_watts
        self.architecture_type = architecture_type
        
        self.fp16_tflops = fp16_tflops
        self.int8_tops = int8_tops
        self.cost_per_hour = cost_per_hour
        
        # Populate tbps or gbps based on what was given
        if bandwidth_tbps > 0 and bandwidth_gbps == 0:
            self.bandwidth_tbps = bandwidth_tbps
            self.bandwidth_gbps = bandwidth_tbps * 1000.0
        else:
            self.bandwidth_gbps = bandwidth_gbps
            self.bandwidth_tbps = bandwidth_gbps / 1000.0 if bandwidth_gbps > 0 else 0.0
            
        # Default compute score fallback
        if self.compute_score == 0 and self.fp16_tflops > 0:
            self.compute_score = self.fp16_tflops

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary format."""
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


# Dictionary defining the AMD hardware configurations
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
    """Returns the requested hardware profile by name."""
    if name in HARDWARE_DATABASE:
        return HARDWARE_DATABASE[name]
    raise ValueError(f"Hardware profile {name} not found.")
