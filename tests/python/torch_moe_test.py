import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import flash_train.torch as ftrain
from flash_train.torch.moe import _integer, _partition, _version, _MoeOperator


ROUTES = (
    "ag_nn_a2a", "ag_nn_ring", "ag_nn_ring_chunk", "ag_nt_a2a", "ag_nt_a2a_chunk", "ag_nt_ring_chunk",
    "ag_nt_a2a_chunk_32k", "ag_nt_ring_chunk_32k", "rs_nn", "rs_nt",
)


def test_moe_exports():
    assert callable(ftrain.AGGroupGemm)
    assert callable(ftrain.MoEReduceRS)
    assert isinstance(ftrain.moe_is_available(), bool)


@pytest.mark.parametrize("operation", [ftrain.AGGroupGemm, ftrain.MoEReduceRS])
def test_constructor_requires_distributed_context(operation):
    if torch.distributed.is_initialized():
        pytest.skip("requires an uninitialized process group")
    message = "initialize torch.distributed" if ftrain.moe_is_available() else "no MoE backend"
    with pytest.raises(RuntimeError, match=message):
        operation(128, 256, 1024, 8)


@pytest.mark.parametrize("value", [0, -1, 1.5, True, 2**31])
def test_dimension_validation(value):
    with pytest.raises(ValueError):
        _integer(value, "dimension")


@pytest.mark.parametrize("m_dim,sizes", [(129, None), (128, [16] * 7), (128, [0] + [16] * 7),
                                        (128, [15] * 8), (128, [16.0] * 8)])
def test_partition_validation(m_dim, sizes):
    with pytest.raises(ValueError):
        _partition(m_dim, sizes, 8)


def test_uniform_and_ragged_partitions():
    assert _partition(128, None, 8) == (16,) * 8
    assert _partition(152, [12, 14, 16, 18, 20, 22, 24, 26], 8) == (12, 14, 16, 18, 20, 22, 24, 26)


def test_versions_track_inplace_changes_and_explicit_overrides():
    value = torch.ones(2)
    previous = _version(value, None, "weight")
    value.add_(1)
    assert _version(value, None, "weight") > previous
    assert _version(value, 17, "weight") == 17
    for invalid in (-1, True, 1.5, 2**64):
        with pytest.raises(ValueError):
            _version(value, invalid, "weight")


def test_inference_tensors_require_an_explicit_version():
    with torch.inference_mode():
        value = torch.ones(2)
    with pytest.raises(ValueError, match="required for inference tensors"):
        _version(value, None, "weight")
    assert _version(value, 4, "weight") == 4


def test_closed_operator_and_concurrent_access_are_rejected():
    import threading
    operation = object.__new__(_MoeOperator)
    operation._session = None
    operation._lock = threading.Lock()
    operation.close()
    assert operation.closed
    with pytest.raises(RuntimeError, match="closed"):
        operation.synchronize()
    operation._lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="concurrent access"):
            operation.close()
    finally:
        operation._lock.release()


@pytest.mark.parametrize("callback", [lambda payload: None, lambda payload: b"wrong byte count"])
def test_bootstrap_callback_errors_are_reported(callback):
    if not torch.cuda.is_available() or not ftrain.moe_is_available():
        pytest.skip("requires an enabled MoE backend and one device")
    from flash_train.torch._flash_train_torch import _MoeSession
    device = torch.cuda.current_device()
    if not getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").startswith("gfx936"):
        pytest.skip("MoE bootstrap requires gfx936")
    with pytest.raises(RuntimeError, match="bootstrap: allgather"):
        _MoeSession(True, 0, 2, device, callback, 32, 256, 64, 8, 1, 32, "ag_nn_a2a", [16, 16], 0,
                    torch.cuda.current_stream().cuda_stream)


@pytest.mark.multirank
@pytest.mark.parametrize("route", ROUTES)
def test_moe_multirank_numerics(route, tmp_path):
    if os.environ.get("FTRAIN_TEST_MOE_MULTIRANK", "0").lower() not in ("1", "on", "true"):
        pytest.skip("set FTRAIN_TEST_MOE_MULTIRANK=1 to run eight-device MoE tests")
    if not ftrain.moe_is_available():
        pytest.skip("MoE backend is disabled")
    runner = Path(__file__).resolve().parents[1] / "support" / "run_multirank.py"
    worker = Path(__file__).with_name("moe_numerics_worker.py")
    output = Path(os.environ["FTRAIN_MOE_TEST_LOG_DIR"]) / route if "FTRAIN_MOE_TEST_LOG_DIR" in os.environ else tmp_path
    command = [sys.executable, str(runner), str(worker), "ag" if route.startswith("ag_") else "rs", route,
               str(output), "--world-size", "8", "--timeout", "360", "--interpreter", sys.executable]
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=390)
    if result.returncode == 77:
        pytest.skip(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
