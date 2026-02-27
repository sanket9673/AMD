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
    ):
        """
        Initialize an AMD Hardware Profile.
        
        Args:
            name (str): Identifier for the hardware.
            memory_gb (int): Total available memory in gigabytes.
            compute_score (float): Theoretical peak performance metric (e.g., TFLOPS/TOPS).
            bandwidth_gbps (float): Peak memory bandwidth in gigabytes per second.
            power_watts (float): Thermal Design Power (TDP) in watts.
            architecture_type (str): The underlying architecture (e.g., CDNA_3, XDNA).
        """
        self.name = name
        self.memory_gb = memory_gb
        self.compute_score = compute_score
        self.bandwidth_gbps = bandwidth_gbps
        self.power_watts = power_watts
        self.architecture_type = architecture_type

    def to_dict(self) -> Dict[str, Any]:
        """Convert the profile to a dictionary format."""
        return {
            "name": self.name,
            "memory_gb": self.memory_gb,
            "compute_score": self.compute_score,
            "bandwidth_gbps": self.bandwidth_gbps,
            "power_watts": self.power_watts,
            "architecture_type": self.architecture_type,
        }

    def __repr__(self) -> str:
        return f"HardwareProfile(name={self.name}, architecture={self.architecture_type})"


# Dictionary defining the 4 AMD hardware configurations with realistic scaling differences
HARDWARE_DATABASE: Dict[str, HardwareProfile] = {
    "AMD_MI300X": HardwareProfile(
        name="AMD_MI300X",
        memory_gb=192,
        compute_score=1300.0,  # ~1.3 PFLOPS (FP16/BF16 with sparsity)
        bandwidth_gbps=5300.0, # 5.3 TB/s HBM3
        power_watts=750.0,
        architecture_type="CDNA_3"
    ),
    "AMD_MI210": HardwareProfile(
        name="AMD_MI210",
        memory_gb=64,
        compute_score=181.0,   # ~181 TFLOPS (FP16/BF16)
        bandwidth_gbps=3200.0, # 3.2 TB/s HBM2e
        power_watts=300.0,
        architecture_type="CDNA_2"
    ),
    "AMD_Ryzen_AI_Edge": HardwareProfile(
        name="AMD_Ryzen_AI_Edge",
        memory_gb=32,          # Shared system memory typical for premium edge
        compute_score=39.0,    # ~39 TOPS (XDNA NPU)
        bandwidth_gbps=100.0,  # LPDDR5x bandwidth
        power_watts=54.0,      # Typical APU TDP
        architecture_type="XDNA"
    ),
    "AMD_Embedded_8GB": HardwareProfile(
        name="AMD_Embedded_8GB",
        memory_gb=8,           # Constrained embedded memory
        compute_score=10.0,    # Generic compute score for embedded RDNA/Zen
        bandwidth_gbps=64.0,   # Standard DDR4/LPDDR4 bandwidth
        power_watts=15.0,      # Low-power embedded TDP
        architecture_type="RDNA_3"
    )
}
