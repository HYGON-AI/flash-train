# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Run the GEMM logging example twice and check against PyTorch.

Requires an HCU environment and a flash-train build containing the logger.
Select a free physical device and logging level before starting the process.
This script prints numerical-check results to stdout; the library logs stderr.
"""

import argparse

import torch
import flash_train.torch as ftrain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--invalid-shape", action="store_true", help="trigger the binding's shape check"
    )
    args = parser.parse_args()

    torch.manual_seed(20261016)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn(128, 64, device="cuda", dtype=torch.float32)
    b = torch.randn(
        32 if args.invalid_shape else 64, 96, device="cuda", dtype=torch.float32
    )
    c = torch.randn(128, 96, device="cuda", dtype=torch.float32)

    for call in range(1, 3):
        out = ftrain.gemm(a, b, c, alpha=1.5, beta=0.25)
        reference = 1.5 * (a @ b) + 0.25 * c
        torch.testing.assert_close(out, reference, rtol=1e-4, atol=1e-5)
        max_error = (out - reference).abs().max().item()
        print(
            f"[example] call={call} shape={tuple(out.shape)} "
            f"max_abs_error={max_error:.8g} check=passed",
            flush=True,
        )


if __name__ == "__main__":
    main()
