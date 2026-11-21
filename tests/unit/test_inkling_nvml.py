from unittest.mock import Mock

import pytest

from datagoblin.training_gym.rlt_gym import inkling_nvml


def test_nvml_unknown_error_retries_within_sample_deadline(monkeypatch):
    sample = {"vram_free_mib": 9000, "gpu_util_percent": 1, "compute_pids": [123]}
    device = Mock(spec=inkling_nvml._Device)
    device.sample.side_effect = [inkling_nvml._NvmlError("device utilization", 999), sample]
    pause = Mock()
    monkeypatch.setattr(inkling_nvml, "monotonic", lambda: 10.0)
    monkeypatch.setattr(inkling_nvml, "sleep", pause)

    assert inkling_nvml._read_fresh(device, deadline=15.0) == sample
    assert device.sample.call_count == 2
    pause.assert_called_once_with(0.2)


@pytest.mark.parametrize(("status", "deadline"), [(15, 15.0), (999, 10.1)])
def test_nvml_retry_preserves_unavailable_telemetry(status, deadline, monkeypatch):
    device = Mock(spec=inkling_nvml._Device)
    device.sample.side_effect = inkling_nvml._NvmlError("device utilization", status)
    pause = Mock()
    monkeypatch.setattr(inkling_nvml, "monotonic", lambda: 10.0)
    monkeypatch.setattr(inkling_nvml, "sleep", pause)

    with pytest.raises(inkling_nvml._NvmlError, match=f"NVML error {status}"):
        inkling_nvml._read_fresh(device, deadline)

    device.sample.assert_called_once()
    pause.assert_not_called()
