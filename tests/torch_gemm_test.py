"""Black-box check of the ftrain_torch binding against torch.matmul."""

import sys

import torch

import ftrain_torch


def main() -> int:
    if not torch.cuda.is_available():
        print("no visible device; skipping", file=sys.stderr)
        return 0
    device = "cuda"
    generator = torch.Generator(device=device).manual_seed(7)

    a = torch.randn(128, 64, device=device, generator=generator)
    b = torch.randn(64, 96, device=device, generator=generator)
    c = torch.randn(128, 96, device=device, generator=generator)

    alpha, beta = 1.5, 0.25
    d = ftrain_torch.gemm(a, b, c, alpha=alpha, beta=beta)
    torch.testing.assert_close(d, alpha * (a @ b) + beta * c, rtol=1e-4, atol=1e-3)

    d_no_c = ftrain_torch.gemm(a, b, alpha=alpha)
    torch.testing.assert_close(d_no_c, alpha * (a @ b), rtol=1e-4, atol=1e-3)

    defaults = ftrain_torch.gemm(a, b)
    torch.testing.assert_close(defaults, a @ b, rtol=1e-4, atol=1e-3)

    # Shape and placement rules surface as python errors.
    for bad_call in (
        lambda: ftrain_torch.gemm(a, torch.randn(3, 3, device=device)),
        lambda: ftrain_torch.gemm(a.cpu(), b),
        lambda: ftrain_torch.gemm(a.to(torch.float64), b),
    ):
        try:
            bad_call()
        except RuntimeError:
            pass
        else:
            print("expected a rejection", file=sys.stderr)
            return 1

    print("ftrain_torch.gemm matches torch.matmul")
    return 0


if __name__ == "__main__":
    sys.exit(main())
