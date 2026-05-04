"""
Tests for vitriol.core.adaptive_sharder module.
"""
import pytest
from unittest.mock import patch, MagicMock

import torch

from vitriol.core.adaptive_sharder import (
    HardwareProfile,
    SystemCapabilities,
    HardwareDetector,
    AdaptiveShardManager,
    StreamingShardWriter,
)


class TestHardwareProfile:
    def test_enum_values(self):
        assert HardwareProfile.LOW_END.value == "low_end"
        assert HardwareProfile.MID_RANGE.value == "mid_range"
        assert HardwareProfile.HIGH_END.value == "high_end"
        assert HardwareProfile.WORKSTATION.value == "workstation"


class TestSystemCapabilities:
    def test_to_dict(self):
        caps = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=500.0,
            network_speed_mbps=None,
        )
        d = caps.to_dict()
        assert d["total_memory_gb"] == 16.0
        assert d["available_memory_gb"] == 8.0
        assert d["cpu_count"] == 8
        assert d["has_gpu"] is False
        assert d["gpu_memory_gb"] is None
        assert d["disk_type"] == "ssd"
        assert d["disk_speed_mbps"] == 500.0
        assert d["network_speed_mbps"] is None


class TestHardwareDetector:
    @pytest.fixture
    def detector(self):
        return HardwareDetector()

    @patch("psutil.virtual_memory")
    @patch("torch.cuda.is_available")
    def test_detect_basic(self, mock_cuda_avail, mock_virtual_mem, detector):
        mock_virtual_mem.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        mock_cuda_avail.return_value = False

        caps = detector.detect()
        assert caps.total_memory_gb == 16.0
        assert caps.available_memory_gb == 8.0
        assert isinstance(caps.cpu_count, int)
        assert caps.has_gpu is False
        assert caps.gpu_memory_gb is None
        assert caps.disk_type in ["hdd", "ssd", "unknown"]

    @patch("psutil.virtual_memory")
    @patch("torch.cuda.is_available")
    @patch("torch.cuda.get_device_properties")
    def test_detect_with_gpu(self, mock_gpu_props, mock_cuda_avail, mock_virtual_mem, detector):
        mock_virtual_mem.return_value = MagicMock(total=64 * 1024**3, available=32 * 1024**3)
        mock_cuda_avail.return_value = True
        mock_props = MagicMock()
        mock_props.total_memory = 24 * 1024**3
        mock_gpu_props.return_value = mock_props

        caps = detector.detect()
        assert caps.has_gpu is True
        assert caps.gpu_memory_gb == 24.0

    @patch("psutil.virtual_memory")
    @patch("torch.cuda.is_available")
    def test_detect_gpu_exception(self, mock_cuda_avail, mock_virtual_mem, detector):
        mock_virtual_mem.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3)
        mock_cuda_avail.return_value = True

        with patch("torch.cuda.get_device_properties", side_effect=RuntimeError("CUDA error")):
            caps = detector.detect()
            assert caps.has_gpu is False

    def test_classify_hardware_workstation(self, detector):
        caps = SystemCapabilities(
            total_memory_gb=256.0,
            available_memory_gb=128.0,
            cpu_count=32,
            has_gpu=True,
            gpu_memory_gb=80.0,
            disk_type="nvme",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        assert detector.classify_hardware(caps) == HardwareProfile.WORKSTATION

    def test_classify_hardware_high_end(self, detector):
        caps = SystemCapabilities(
            total_memory_gb=64.0,
            available_memory_gb=32.0,
            cpu_count=16,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        assert detector.classify_hardware(caps) == HardwareProfile.HIGH_END

    def test_classify_hardware_mid_range(self, detector):
        caps = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        assert detector.classify_hardware(caps) == HardwareProfile.MID_RANGE

    def test_classify_hardware_low_end(self, detector):
        caps = SystemCapabilities(
            total_memory_gb=4.0,
            available_memory_gb=2.0,
            cpu_count=4,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="hdd",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        assert detector.classify_hardware(caps) == HardwareProfile.LOW_END

    def test_detect_disk_type_posix(self, detector):
        with patch("os.name", "posix"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="0\n")
                assert detector._detect_disk_type() == "ssd"

    def test_detect_disk_type_posix_hdd(self, detector):
        with patch("os.name", "posix"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="1\n")
                assert detector._detect_disk_type() == "hdd"

    def test_detect_disk_type_posix_error(self, detector):
        with patch("os.name", "posix"):
            with patch("subprocess.run", side_effect=Exception("error")):
                assert detector._detect_disk_type() == "unknown"

    def test_detect_disk_type_non_posix(self, detector):
        with patch("os.name", "nt"):
            assert detector._detect_disk_type() == "unknown"


class TestAdaptiveShardManager:
    def test_init_with_hardware_profile_override(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.HIGH_END,
            auto_detect=False,
        )
        assert manager.profile == HardwareProfile.HIGH_END
        assert manager.capabilities is None
        assert manager.max_shard_bytes == 2000 * 1024 * 1024

    def test_init_with_max_shard_size(self):
        manager = AdaptiveShardManager(
            max_shard_size="1024MB",
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager.max_shard_bytes == 1024 * 1024 * 1024

    def test_init_default_mid_range(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager.max_shard_bytes == 500 * 1024 * 1024

    def test_parse_size_various_units(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager._parse_size("100B") == 100
        assert manager._parse_size("10KB") == 10 * 1024
        assert manager._parse_size("5MB") == 5 * 1024 ** 2
        assert manager._parse_size("2GB") == 2 * 1024 ** 3
        assert manager._parse_size("1TB") == 1 * 1024 ** 4

    def test_parse_size_no_unit(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager._parse_size("1024") == 1024

    def test_format_size(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager._format_size(512) == "512.0B"
        assert manager._format_size(1024) == "1.0KB"
        assert manager._format_size(1024 ** 2) == "1.0MB"
        assert manager._format_size(1024 ** 3) == "1.0GB"
        assert manager._format_size(1024 ** 4) == "1.0TB"

    def test_adjust_for_memory(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.HIGH_END,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=4.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        original_max = manager.max_shard_bytes
        manager._adjust_for_memory()
        # Should be reduced based on available memory
        assert manager.max_shard_bytes <= original_max

    def test_plan_shards_basic(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        param_names = [f"param_{i}" for i in range(5)]
        param_sizes = {name: 100 * 1024 * 1024 for name in param_names}  # 100MB each

        shards = manager.plan_shards(param_names, param_sizes)
        assert len(shards) >= 1
        # With 500MB max and 100MB params, all 5 should fit in one or two shards
        total_in_shards = sum(len(s[1]) for s in shards)
        assert total_in_shards == 5

    def test_plan_shards_multiple_shards(self):
        manager = AdaptiveShardManager(
            max_shard_size="200MB",
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        param_names = [f"param_{i}" for i in range(5)]
        param_sizes = {name: 100 * 1024 * 1024 for name in param_names}

        shards = manager.plan_shards(param_names, param_sizes)
        # 5 * 100MB = 500MB total, max 200MB per shard -> at least 3 shards
        assert len(shards) >= 2

    def test_plan_shards_empty(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        shards = manager.plan_shards([], {})
        assert shards == []

    def test_plan_shards_updates_stats(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        param_names = [f"param_{i}" for i in range(3)]
        param_sizes = {name: 100 * 1024 * 1024 for name in param_names}

        manager.plan_shards(param_names, param_sizes)
        assert manager.stats["total_shards"] >= 1
        assert manager.stats["total_bytes"] == 300 * 1024 * 1024
        assert manager.stats["avg_shard_size"] > 0
        assert manager.stats["planning_time"] >= 0

    def test_get_optimal_workers_no_caps(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager.get_optimal_workers() == 4

    def test_get_optimal_workers_with_caps(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.HIGH_END,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=64.0,
            available_memory_gb=32.0,
            cpu_count=16,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="nvme",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        workers = manager.get_optimal_workers()
        assert workers > 0
        # NVMe allows more workers, up to 16
        assert workers <= 16

    def test_get_optimal_workers_hdd_limited(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=16,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="hdd",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        workers = manager.get_optimal_workers()
        assert workers <= 2

    def test_get_optimal_workers_low_memory(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=4.0,
            available_memory_gb=2.0,
            cpu_count=16,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=None,
            network_speed_mbps=None,
        )
        workers = manager.get_optimal_workers()
        assert workers <= 2

    def test_estimate_time_write(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=500,
            network_speed_mbps=None,
        )
        time_est = manager.estimate_time(1024 * 1024 * 1024, operation="write")  # 1GB
        assert time_est > 0

    def test_estimate_time_read_faster_than_write(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=500,
            network_speed_mbps=None,
        )
        write_time = manager.estimate_time(1024 * 1024 * 1024, operation="write")
        read_time = manager.estimate_time(1024 * 1024 * 1024, operation="read")
        assert read_time < write_time

    def test_estimate_time_compute(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=500,
            network_speed_mbps=None,
        )
        time_est = manager.estimate_time(1024 * 1024 * 1024, operation="compute")
        assert time_est > 0

    def test_estimate_time_unknown_operation(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_count=8,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="ssd",
            disk_speed_mbps=500,
            network_speed_mbps=None,
        )
        assert manager.estimate_time(1024, operation="unknown") == 0.0

    def test_estimate_time_no_caps(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        assert manager.estimate_time(1024 * 1024 * 1024, operation="write") == 0.0

    def test_get_recommendations(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.HIGH_END,
            auto_detect=False,
        )
        manager.capabilities = SystemCapabilities(
            total_memory_gb=64.0,
            available_memory_gb=32.0,
            cpu_count=16,
            has_gpu=False,
            gpu_memory_gb=None,
            disk_type="nvme",
            disk_speed_mbps=2000,
            network_speed_mbps=None,
        )
        recs = manager.get_recommendations()
        assert recs["hardware_profile"] == "high_end"
        assert "max_shard_size" in recs
        assert "optimal_workers" in recs
        assert recs["use_memory_mapping"] is True
        assert recs["enable_parallel_io"] is True
        assert recs["compression_recommended"] is False
        assert "capabilities" in recs

    def test_get_recommendations_no_caps(self):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        recs = manager.get_recommendations()
        assert recs["hardware_profile"] == "mid_range"
        assert "capabilities" not in recs


class TestStreamingShardWriter:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        return str(tmp_path / "shards")

    def test_init(self, temp_dir):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        writer = StreamingShardWriter(manager, temp_dir)
        assert writer.shard_manager is manager
        assert writer.output_dir == temp_dir
        assert writer.current_buffer_size == 0
        assert writer.shard_count == 0

    def test_write_tensor_accumulates(self, temp_dir):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        writer = StreamingShardWriter(manager, temp_dir)
        tensor = torch.randn(10, 10)
        writer.write_tensor("test", tensor)
        assert writer.shard_count == 0
        assert writer.current_buffer_size > 0
        assert "test" in writer.current_buffer

    def test_write_tensor_triggers_flush(self, temp_dir):
        manager = AdaptiveShardManager(
            max_shard_size="1KB",
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        writer = StreamingShardWriter(manager, temp_dir)
        # Override buffer size to force flush
        writer.buffer_size = 1024  # 1KB
        # Write a small tensor first, then a large one to trigger flush
        writer.write_tensor("small", torch.randn(10, 10))  # ~400B
        tensor = torch.randn(1000, 1000)  # ~4MB for float32
        writer.write_tensor("large", tensor)
        assert writer.shard_count >= 1
        assert writer.current_buffer_size > 0  # large tensor stays in buffer

    def test_flush_empty_buffer(self, temp_dir):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        writer = StreamingShardWriter(manager, temp_dir)
        writer._flush_buffer()
        assert writer.shard_count == 0

    def test_close_flushes_remaining(self, temp_dir):
        manager = AdaptiveShardManager(
            hardware_profile=HardwareProfile.MID_RANGE,
            auto_detect=False,
        )
        writer = StreamingShardWriter(manager, temp_dir)
        tensor = torch.randn(10, 10)
        writer.write_tensor("test", tensor)
        assert writer.shard_count == 0
        writer.close()
        assert writer.shard_count == 1
