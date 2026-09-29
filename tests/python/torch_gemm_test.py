# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Black-box check of the flash_train.torch binding against torch.matmul."""

import pytest
import torch

import flash_train.torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="no visible HCU device")


def make_inputs():
    generator = torch.Generator(device="cuda").manual_seed(7)
    a = torch.randn(128, 64, device="cuda", generator=generator)
    b = torch.randn(64, 96, device="cuda", generator=generator)
    c = torch.randn(128, 96, device="cuda", generator=generator)
    return a, b, c


def test_gemm_matches_matmul():
    a, b, c = make_inputs()
    alpha, beta = 1.5, 0.25
    d = flash_train.torch.gemm(a, b, c, alpha=alpha, beta=beta)
    torch.testing.assert_close(d, alpha * (a @ b) + beta * c, rtol=1e-4, atol=1e-3)


def test_gemm_without_c():
    a, b, _ = make_inputs()
    d = flash_train.torch.gemm(a, b, alpha=1.5)
    torch.testing.assert_close(d, 1.5 * (a @ b), rtol=1e-4, atol=1e-3)


def test_gemm_defaults():
    a, b, _ = make_inputs()
    d = flash_train.torch.gemm(a, b)
    torch.testing.assert_close(d, a @ b, rtol=1e-4, atol=1e-3)


def test_rejects_shape_mismatch():
    a, _, _ = make_inputs()
    bad = torch.randn(3, 3, device="cuda")
    with pytest.raises(RuntimeError):
        flash_train.torch.gemm(a, bad)


def test_rejects_cpu_tensor():
    a, b, _ = make_inputs()
    with pytest.raises(RuntimeError):
        flash_train.torch.gemm(a.cpu(), b)


def test_rejects_float64():
    a, b, _ = make_inputs()
    with pytest.raises(RuntimeError):
        flash_train.torch.gemm(a.to(torch.float64), b)
