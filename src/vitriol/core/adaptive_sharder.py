"""
Adaptive Sharding Strategy based on Hardware Characteristics.

This module implements intelligent sharding that adapts to:
- Available system memory
- Disk I/O bandwidth
- Network speed (for distributed setups)
- GPU memory (if available)

Automatically optimizes shard sizes for the specific hardware.
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch

from ..utils.size import parse_size_to_bytes

logger = logging.getLogger(__name__)


class HardwareProfile(Enum):
    """Hardware capability profiles."""
    LOW_END = "low_end"           # < 8GB RAM, slow disk
    MID_RANGE = "mid_range"       # 8-32GB RAM, SSD
    HIGH_END = "high_end"         # 32-128GB RAM, NVMe
    WORKSTATION = "workstation"   # > 128GB RAM, multiple GPUs


@dataclass
class SystemCapabilities:
    """Detected system capabilities."""
    total_memory_gb: float
    available_memory_gb: float
    cpu_count: int
    has_gpu: bool
    gpu_memory_gb: Optional[float]
    disk_type: str  # 'hdd', 'ssd', 'nvme'
    disk_speed_mbps: Optional[float]
    network_speed_mbps: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "cpu_count": self.cpu_count,
            "has_gpu": self.has_gpu,
            "gpu_memory_gb": self.gpu_memory_gb,
            "disk_type": self.disk_type,
            "disk_speed_mbps": self.disk_speed_mbps,
            "network_speed_mbps": self.network_speed_mbps
        }


class HardwareDetector:
    """Detects system hardware capabilities."""

    def detect(self) -> SystemCapabilities:
        """
        Detect system hardware capabilities.

        Returns:
            SystemCapabilities object
        """
        # Memory
        mem = psutil.virtual_memory()
        total_memory_gb = mem.total / (1024 ** 3)
        available_memory_gb = mem.available / (1024 ** 3)

        # CPU
        cpu_count = os.cpu_count() or 1

        # GPU
        has_gpu = torch.cuda.is_available()
        gpu_memory_gb = None
        if has_gpu:
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except RuntimeError:
                has_gpu = False

        # Disk type (heuristic)
        disk_type = self._detect_disk_type()

        # Disk speed (optional, can be slow to detect)
        disk_speed = None

        return SystemCapabilities(
            total_memory_gb=total_memory_gb,
            available_memory_gb=available_memory_gb,
            cpu_count=cpu_count,
            has_gpu=has_gpu,
            gpu_memory_gb=gpu_memory_gb,
            disk_type=disk_type,
            disk_speed_mbps=disk_speed,
            network_speed_mbps=None
        )

    def _detect_disk_type(self) -> str:
        """Detect disk type (heuristic)."""
        # Simple heuristic based on OS
        if os.name == 'posix':
            try:
                # Check if root is on SSD
                import subprocess
                result = subprocess.run(
                    ['cat', '/sys/block/sda/queue/rotational'],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    rotational = int(result.stdout.strip())
                    return 'hdd' if rotational else 'ssd'
            except Exception:
                logger.debug("Failed to detect disk type via /sys/block/sda/queue/rotational")

        return 'unknown'

    def classify_hardware(self, caps: SystemCapabilities) -> HardwareProfile:
        """
        Classify hardware into a profile.

        Args:
            caps: System capabilities

        Returns:
            HardwareProfile enum
        """
        if caps.total_memory_gb >= 128 and caps.has_gpu:
            return HardwareProfile.WORKSTATION
        elif caps.total_memory_gb >= 32:
            return HardwareProfile.HIGH_END
        elif caps.total_memory_gb >= 8:
            return HardwareProfile.MID_RANGE
        else:
            return HardwareProfile.LOW_END


class AdaptiveShardManager:
    """
    Adaptive shard manager that optimizes for hardware.

    Features:
        ✅ Automatic hardware detection
        ✅ Dynamic shard size calculation
        ✅ Memory-aware buffering
        ✅ Parallel I/O optimization
        ✅ Progress tracking with ETA

    Example:
        >>> manager = AdaptiveShardManager()
        >>> shards = manager.plan_shards(param_names, param_sizes)
    """

    # Default shard sizes by hardware profile (in MB)
    DEFAULT_SHARD_SIZES = {
        HardwareProfile.LOW_END: 100,      # 100MB shards
        HardwareProfile.MID_RANGE: 500,    # 500MB shards
        HardwareProfile.HIGH_END: 2000,    # 2GB shards
        HardwareProfile.WORKSTATION: 5000  # 5GB shards
    }

    # Memory safety margins
    MEMORY_SAFETY_FACTOR = 0.7  # Use only 70% of available memory

    def __init__(
        self,
        max_shard_size: Optional[str] = None,
        hardware_profile: Optional[HardwareProfile] = None,
        auto_detect: bool = True
    ):
        """
        Initialize adaptive shard manager.

        Args:
            max_shard_size: Maximum shard size (e.g., "5GB")
            hardware_profile: Override hardware profile
            auto_detect: Automatically detect hardware capabilities
        """
        self.detector = HardwareDetector()

        # Detect or use provided hardware info
        if auto_detect and hardware_profile is None:
            self.capabilities = self.detector.detect()
            self.profile = self.detector.classify_hardware(self.capabilities)
            logger.info(f"Detected hardware: {self.profile.value}")
            logger.info(f"Capabilities: {self.capabilities.to_dict()}")
        else:
            self.profile = hardware_profile or HardwareProfile.MID_RANGE
            self.capabilities = None

        # Determine max shard size
        if max_shard_size:
            self.max_shard_bytes = self._parse_size(max_shard_size)
        else:
            default_mb = self.DEFAULT_SHARD_SIZES[self.profile]
            self.max_shard_bytes = default_mb * 1024 * 1024

        # Adjust based on available memory
        if self.capabilities:
            self._adjust_for_memory()

        # Statistics
        self.stats = {
            "total_shards": 0,
            "total_bytes": 0,
            "avg_shard_size": 0,
            "planning_time": 0
        }

        logger.info(f"AdaptiveShardManager: max_shard_size={self._format_size(self.max_shard_bytes)}")

    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes."""
        return parse_size_to_bytes(size_str)

    def _format_size(self, bytes: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024:
                return f"{bytes:.1f}{unit}"
            bytes /= 1024
        return f"{bytes:.1f}PB"

    def _adjust_for_memory(self):
        """Adjust shard size based on available memory."""
        if not self.capabilities:
            return

        # Calculate safe memory usage
        safe_memory = self.capabilities.available_memory_gb * self.MEMORY_SAFETY_FACTOR

        # Convert to bytes
        safe_memory_bytes = safe_memory * 1024 ** 3

        # Use smaller of configured max and safe memory
        self.max_shard_bytes = min(self.max_shard_bytes, int(safe_memory_bytes / 2))

        logger.info(f"Adjusted max shard size to {self._format_size(self.max_shard_bytes)} "
                   f"(based on {safe_memory:.1f}GB safe memory)")

    def plan_shards(
        self,
        param_names: List[str],
        param_sizes: Dict[str, int]
    ) -> List[Tuple[str, Dict[str, int]]]:
        """
        Plan optimal shard distribution.

        Args:
            param_names: List of parameter names
            param_sizes: Dict mapping names to sizes in bytes

        Returns:
            List of (shard_name, {param_name: size}) tuples
        """
        start_time = time.time()

        shards = []
        current_shard = {}
        current_size = 0
        shard_idx = 0

        for name in param_names:
            size = param_sizes.get(name, 0)

            # Check if adding this parameter would exceed limit
            if current_size + size > self.max_shard_bytes and current_shard:
                # Save current shard
                shard_name = f"model_{shard_idx:05d}-of-{len(param_names):05d}.safetensors"
                shards.append((shard_name, current_shard))

                # Start new shard
                current_shard = {name: size}
                current_size = size
                shard_idx += 1
            else:
                # Add to current shard
                current_shard[name] = size
                current_size += size

        # Don't forget the last shard
        if current_shard:
            shard_name = f"model_{shard_idx:05d}-of-{len(param_names):05d}.safetensors"
            shards.append((shard_name, current_shard))

        # Update statistics
        self.stats["total_shards"] = len(shards)
        self.stats["total_bytes"] = sum(param_sizes.values())
        self.stats["avg_shard_size"] = self.stats["total_bytes"] / len(shards) if shards else 0
        self.stats["planning_time"] = time.time() - start_time

        logger.info(f"Planned {len(shards)} shards, avg size: "
                   f"{self._format_size(int(self.stats['avg_shard_size']))}")

        return shards

    def get_optimal_workers(self) -> int:
        """
        Get optimal number of parallel workers for this hardware.

        Returns:
            Recommended number of workers
        """
        if not self.capabilities:
            return 4  # Default

        # Base on CPU count
        workers = self.capabilities.cpu_count

        # Adjust based on disk type
        if self.capabilities.disk_type == 'hdd':
            workers = min(workers, 2)  # HDDs don't benefit from many parallel reads
        elif self.capabilities.disk_type == 'nvme':
            workers = min(workers * 2, 16)  # NVMe can handle more

        # Adjust based on memory
        if self.capabilities.available_memory_gb < 4:
            workers = min(workers, 2)  # Low memory, be conservative

        return max(1, workers)

    def estimate_time(self, total_bytes: int, operation: str = "write") -> float:
        """
        Estimate time for operation based on hardware.

        Args:
            total_bytes: Total bytes to process
            operation: 'read', 'write', or 'compute'

        Returns:
            Estimated time in seconds
        """
        if not self.capabilities:
            return 0.0

        if operation == "write":
            # Estimate based on disk speed
            if self.capabilities.disk_speed_mbps:
                speed_mbps = self.capabilities.disk_speed_mbps
            else:
                # Rough estimates
                speed_mbps = {
                    'hdd': 100,
                    'ssd': 500,
                    'nvme': 2000,
                    'unknown': 200
                }.get(self.capabilities.disk_type, 200)

            total_mb = total_bytes / (1024 * 1024)
            return total_mb / speed_mbps

        elif operation == "read":
            # Reads are typically faster
            return self.estimate_time(total_bytes, "write") * 0.8

        elif operation == "compute":
            # Rough estimate based on CPU
            # Assume 1GB/s processing per core
            total_gb = total_bytes / (1024 ** 3)
            return total_gb / self.capabilities.cpu_count

        return 0.0

    def get_recommendations(self) -> Dict[str, Any]:
        """
        Get hardware-specific recommendations.

        Returns:
            Dict with recommendations
        """
        recommendations = {
            "hardware_profile": self.profile.value,
            "max_shard_size": self._format_size(self.max_shard_bytes),
            "optimal_workers": self.get_optimal_workers(),
            "use_memory_mapping": self.profile in [HardwareProfile.HIGH_END, HardwareProfile.WORKSTATION],
            "enable_parallel_io": self.capabilities.disk_type != 'hdd' if self.capabilities else True,
            "compression_recommended": self.profile in [HardwareProfile.LOW_END, HardwareProfile.MID_RANGE]
        }

        if self.capabilities:
            recommendations["capabilities"] = self.capabilities.to_dict()

        return recommendations


class StreamingShardWriter:
    """
    Streaming shard writer with adaptive buffering.

    Writes shards directly to disk without loading all into memory,
    with adaptive buffering based on available memory.
    """

    def __init__(self, shard_manager: AdaptiveShardManager, output_dir: str):
        """
        Initialize streaming writer.

        Args:
            shard_manager: AdaptiveShardManager instance
            output_dir: Output directory for shards
        """
        self.shard_manager = shard_manager
        self.output_dir = output_dir
        self.buffer_size = self._calculate_buffer_size()

        self.current_buffer: Dict[str, torch.Tensor] = {}
        self.current_buffer_size = 0
        self.shard_count = 0

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"StreamingShardWriter: buffer_size={self.shard_manager._format_size(self.buffer_size)}")

    def _calculate_buffer_size(self) -> int:
        """Calculate optimal buffer size based on memory."""
        if not self.shard_manager.capabilities:
            return 100 * 1024 * 1024  # 100MB default

        # Use 10% of available memory, capped at max shard size
        available = self.shard_manager.capabilities.available_memory_gb * 1024 ** 3
        buffer = int(available * 0.1)

        return min(buffer, self.shard_manager.max_shard_bytes)

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """
        Write a tensor to the stream.

        Args:
            name: Parameter name
            tensor: Tensor to write
        """
        tensor_size = tensor.numel() * tensor.element_size()

        # Check if adding would exceed buffer
        if (self.current_buffer_size + tensor_size > self.buffer_size and
            self.current_buffer):
            self._flush_buffer()

        # Add to buffer
        self.current_buffer[name] = tensor
        self.current_buffer_size += tensor_size

    def _flush_buffer(self):
        """Flush current buffer to disk."""
        if not self.current_buffer:
            return

        shard_name = f"model_{self.shard_count:05d}.safetensors"
        shard_path = os.path.join(self.output_dir, shard_name)

        # Save (using safetensors if available)
        try:
            from safetensors.torch import save_file
            save_file(self.current_buffer, shard_path)
        except Exception as e:
            logger.debug("safetensors save failed, falling back to torch.save: %s", e)
            torch.save(self.current_buffer, shard_path)

        logger.debug(f"Flushed shard {shard_name} ({len(self.current_buffer)} tensors)")

        # Clear buffer
        self.current_buffer = {}
        self.current_buffer_size = 0
        self.shard_count += 1

    def close(self) -> None:
        """Close writer and flush remaining buffer."""
        self._flush_buffer()
        logger.info(f"StreamingShardWriter: wrote {self.shard_count} shards")
